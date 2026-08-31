"""Task-conditioned semantic graph replay for Spotify's public web player.

The graph is parameterized by artist and track rank. It intentionally performs
read-only navigation and extraction: no playback, follows, likes, or playlist
mutations. A task can terminate at the canonical artist node or continue to an
ordered Popular-track extraction node.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator
from uuid_extensions import uuid7str

from browser_use import BrowserProfile, BrowserSession, Tools
from browser_use.browser.profile import ViewportSize
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode

SPOTIFY_ORIGIN = 'https://open.spotify.com'
GRAPH_SCHEMA_VERSION = '0.1.0'
SPOTIFY_GRAPH_KILL_SWITCH = 'BROWSER_USE_MEMORABLE_SPOTIFY_GRAPH'


def _normalize(value: Any) -> str:
	return re.sub(r'\s+', ' ', str(value or '').strip()).casefold()


def search_url_matches_artist(url: str, artist: str) -> bool:
	path = urlsplit(url).path
	if not path.startswith('/search/'):
		return False
	return _normalize(unquote(path.removeprefix('/search/'))) == _normalize(artist)


class SpotifyGoal(str, Enum):
	CANONICAL_ARTIST = 'canonical_artist'
	POPULAR_TRACK = 'popular_track'


class GraphAction(str, Enum):
	INPUT = 'input'
	CLICK = 'click'
	EXTRACT_RANKED = 'extract_ranked'


class GraphLocator(BaseModel):
	fixed: dict[str, str] = Field(default_factory=dict)
	dynamic: dict[str, str] = Field(default_factory=dict)
	href_prefix: str | None = None
	ancestor_role: str | None = None
	ancestor_name: str | None = None


class GraphNode(BaseModel):
	id: str
	description: str
	keywords: list[str]
	terminal: bool = False
	required_parameters: list[str] = Field(default_factory=list)
	state_guard: str


class GraphEdge(BaseModel):
	id: str
	source: str
	target: str
	action: GraphAction
	locator: GraphLocator | None = None
	value_from: str | None = None
	postcondition: str


class SpotifyProcedureGraph(BaseModel):
	model_config = ConfigDict(extra='forbid')

	schema_version: Literal['0.1.0'] = GRAPH_SCHEMA_VERSION
	graph_id: str
	entry_node: str
	parameters: list[str]
	nodes: list[GraphNode]
	edges: list[GraphEdge]
	training: dict[str, Any]

	@model_validator(mode='after')
	def validate_graph(self) -> SpotifyProcedureGraph:
		node_ids = {node.id for node in self.nodes}
		if self.entry_node not in node_ids:
			raise ValueError('entry node is missing')
		for edge in self.edges:
			if edge.source not in node_ids or edge.target not in node_ids:
				raise ValueError(f'edge {edge.id} references a missing node')
		return self

	def node(self, node_id: str) -> GraphNode:
		return next(node for node in self.nodes if node.id == node_id)

	def path_to(self, target: str) -> list[GraphEdge]:
		queue: deque[tuple[str, list[GraphEdge]]] = deque([(self.entry_node, [])])
		visited: set[str] = set()
		while queue:
			node_id, path = queue.popleft()
			if node_id == target:
				return path
			if node_id in visited:
				continue
			visited.add(node_id)
			for edge in self.edges:
				if edge.source == node_id:
					queue.append((edge.target, [*path, edge]))
		raise ValueError(f'no path from {self.entry_node} to {target}')

	def write(self, path: str | Path) -> Path:
		output = Path(path).expanduser().resolve()
		output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
		temporary = output.with_name(f'.{output.name}.partial')
		temporary.write_text(self.model_dump_json(indent=2) + '\n')
		os.chmod(temporary, 0o600)
		temporary.replace(output)
		return output

	@classmethod
	def read(cls, path: str | Path) -> SpotifyProcedureGraph:
		return cls.model_validate_json(Path(path).read_text())


class SpotifyTrack(BaseModel):
	rank: int = Field(ge=1)
	name: str
	url: str


class SpotifyTrace(BaseModel):
	trace_id: str
	created_at: str
	artist: str
	terminal_node: str
	search_locator: dict[str, str]
	artist_result_locator: dict[str, str]
	artist_url: str
	popular_tracks: list[SpotifyTrack]
	model_calls: Literal[0] = 0
	verified: bool


class SpotifyTaskIntent(BaseModel):
	task: str
	artist: str
	goal_node: SpotifyGoal
	track_rank: int | None = Field(default=None, ge=1, le=10)
	node_scores: dict[str, int]
	path: list[str]


class GraphEvent(BaseModel):
	edge_id: str
	source: str
	target: str
	status: Literal['executed', 'extracted', 'needs_recovery']
	selector_index: int | None = None
	evidence: dict[str, Any] = Field(default_factory=dict)
	reason: str | None = None


class SpotifyGraphReport(BaseModel):
	run_id: str
	status: Literal['completed', 'needs_recovery', 'disabled']
	intent: SpotifyTaskIntent | None = None
	terminal_node: str | None = None
	artist_url: str | None = None
	track: SpotifyTrack | None = None
	model_calls: Literal[0] = 0
	events: list[GraphEvent] = Field(default_factory=list)
	reason: str | None = None
	trace_path: str | None = None


class SpotifyTaskRouter:
	"""Retrieve a graph terminal from task text and bind artist/rank entities."""

	ORDINALS = {
		'first': 1,
		'second': 2,
		'third': 3,
		'fourth': 4,
		'fifth': 5,
	}

	def route(
		self,
		task: str,
		graph: SpotifyProcedureGraph,
		*,
		artist: str | None = None,
		track_rank: int | None = None,
	) -> SpotifyTaskIntent:
		bound_artist = (artist or self._artist_from_task(task)).strip().strip(' “”"\'.,;:!?')
		if not bound_artist:
			raise ValueError('could not bind an artist from the task')
		tokens = set(re.findall(r"[a-z0-9']+", task.casefold()))
		scores = {
			node.id: sum(1 for keyword in node.keywords if keyword.casefold() in tokens) for node in graph.nodes if node.terminal
		}
		has_track_intent = bool(tokens & {'track', 'song', 'popular', 'first', 'second', 'third', 'fourth', 'fifth'})
		goal = SpotifyGoal.POPULAR_TRACK if has_track_intent else SpotifyGoal.CANONICAL_ARTIST
		rank = (track_rank or self._rank_from_task(task)) if goal == SpotifyGoal.POPULAR_TRACK else None
		if rank is not None and not 1 <= rank <= 10:
			raise ValueError('track rank must be between 1 and 10')
		path = graph.path_to(goal.value)
		return SpotifyTaskIntent(
			task=task,
			artist=bound_artist,
			goal_node=goal,
			track_rank=rank,
			node_scores=scores,
			path=[graph.entry_node, *(edge.target for edge in path)],
		)

	@classmethod
	def _rank_from_task(cls, task: str) -> int:
		lowered = task.casefold()
		for word, value in cls.ORDINALS.items():
			if re.search(rf'\b{re.escape(word)}\b', lowered):
				return value
		match = re.search(r'\b(\d{1,2})(?:[a-z]{2})?\s+(?:visible\s+)?(?:track|song)\b', lowered)
		return int(match.group(1)) if match else 1

	@staticmethod
	def _artist_from_task(task: str) -> str:
		quoted = re.search(r'[“"]([^”"]+)[”"]', task)
		if quoted:
			return quoted.group(1)
		patterns = (
			r'(?:search(?:\s+spotify)?\s+for)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)',
			r'(?:artist\s+result\s+for|artist\s+page\s+for)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)',
			r'(?:navigate|go|open)(?:\s+spotify)?\s+to\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)',
			r'(?:song|track|music)\s+(?:by|from)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)',
			r'(?:best|top|popular)\s+(?:song|track)\s+(?:for|by)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)',
		)
		for pattern in patterns:
			match = re.search(pattern, task, flags=re.IGNORECASE)
			if match:
				return match.group(1).strip()
		return ''


def spotify_graph_template(training: dict[str, Any] | None = None) -> SpotifyProcedureGraph:
	nodes = [
		GraphNode(
			id='spotify_home',
			description='Spotify is open and the artist search control is available.',
			keywords=['spotify', 'home', 'search'],
			state_guard='unique search combobox',
		),
		GraphNode(
			id='search_results',
			description='Search results for the requested artist are visible.',
			keywords=['search', 'results', 'artist'],
			required_parameters=['artist'],
			state_guard='URL path starts /search/ and Top result region exists',
		),
		GraphNode(
			id='canonical_artist',
			description='Open the verified canonical artist result and stop on its artist page.',
			keywords=['open', 'verified', 'canonical', 'artist', 'result', 'page', 'stop'],
			terminal=True,
			required_parameters=['artist'],
			state_guard='canonical /artist/ URL and exact artist heading and Popular section',
		),
		GraphNode(
			id='popular_track',
			description='Identify a ranked visible track in the artist Popular section.',
			keywords=['identify', 'find', 'get', 'track', 'song', 'popular', 'first', 'second', 'third'],
			terminal=True,
			required_parameters=['artist', 'track_rank'],
			state_guard='ordered unique /track/ links under exact Popular heading',
		),
	]
	edges = [
		GraphEdge(
			id='search_artist',
			source='spotify_home',
			target='search_results',
			action=GraphAction.INPUT,
			locator=GraphLocator(
				fixed={
					'node_name': 'input',
					'ax_role': 'combobox',
					'ax_name': 'what do you want to play?',
					'attributes.type': 'search',
					'attributes.data-testid': 'search-input',
				}
			),
			value_from='artist',
			postcondition='search URL contains encoded artist and Top result region exists',
		),
		GraphEdge(
			id='open_canonical_artist',
			source='search_results',
			target='canonical_artist',
			action=GraphAction.CLICK,
			locator=GraphLocator(
				fixed={'node_name': 'a', 'ax_role': 'link'},
				dynamic={'ax_name': 'artist'},
				href_prefix='/artist/',
				ancestor_role='region',
				ancestor_name='top result',
			),
			postcondition='canonical /artist/ URL + exact artist heading + Popular heading',
		),
		GraphEdge(
			id='read_ranked_popular_track',
			source='canonical_artist',
			target='popular_track',
			action=GraphAction.EXTRACT_RANKED,
			value_from='track_rank',
			postcondition='requested rank resolves to one visible /track/ link under Popular',
		),
	]
	identity = json.dumps(
		{'nodes': [node.model_dump(mode='json') for node in nodes], 'edges': [edge.model_dump(mode='json') for edge in edges]},
		sort_keys=True,
	)
	return SpotifyProcedureGraph(
		graph_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
		entry_node='spotify_home',
		parameters=['artist', 'track_rank'],
		nodes=nodes,
		edges=edges,
		training=training or {'trace_count': 0, 'model_calls': 0, 'status': 'template_only'},
	)


def compile_spotify_graph(trace_root: str | Path, minimum_artists: int = 3) -> SpotifyProcedureGraph:
	root = Path(trace_root).expanduser().resolve()
	traces: list[SpotifyTrace] = []
	for path in sorted(root.glob('*.json')):
		try:
			trace = SpotifyTrace.model_validate_json(path.read_text())
		except Exception:
			continue
		if trace.verified:
			traces.append(trace)
	artists = {_normalize(trace.artist) for trace in traces}
	if len(artists) < minimum_artists:
		raise ValueError(f'need {minimum_artists} distinct verified artists; found {len(artists)}')
	if any(not trace.popular_tracks for trace in traces):
		raise ValueError('every training trace must contain Popular-track evidence')
	return spotify_graph_template(
		training={
			'trace_count': len(traces),
			'distinct_artists': len(artists),
			'artists': sorted(trace.artist for trace in traces),
			'model_calls': 0,
			'compiler': 'deterministic_parameterized_graph_aggregation',
			'verified_predicate': 'exact_name + canonical_artist_url + artist_heading + popular_section',
			'captured_values_used_as_runtime_locators': False,
			'samples': [
				{
					'artist': trace.artist,
					'terminal_node': trace.terminal_node,
					'artist_url': trace.artist_url,
					'popular_tracks': [track.model_dump(mode='json') for track in trace.popular_tracks],
				}
				for trace in sorted(traces, key=lambda item: _normalize(item.artist))
			],
		},
	)


class SpotifyGraphExecutor:
	def __init__(
		self,
		graph: SpotifyProcedureGraph,
		*,
		tools: Tools[Any] | None = None,
		trace_dir: Path | None = None,
	):
		self.graph = graph
		self.tools = tools or Tools()
		self.trace_dir = trace_dir
		self.action_model = self.tools.registry.create_action_model(include_actions=['input', 'click'])

	async def run(
		self,
		browser_session: BrowserSession,
		task: str,
		*,
		artist: str | None = None,
		track_rank: int | None = None,
	) -> SpotifyGraphReport:
		run_id = uuid7str()
		if os.getenv(SPOTIFY_GRAPH_KILL_SWITCH, '1').casefold() in {'0', 'false', 'off'}:
			return SpotifyGraphReport(run_id=run_id, status='disabled', reason=f'{SPOTIFY_GRAPH_KILL_SWITCH} is disabled')
		try:
			intent = SpotifyTaskRouter().route(task, self.graph, artist=artist, track_rank=track_rank)
		except ValueError as exc:
			return SpotifyGraphReport(run_id=run_id, status='needs_recovery', reason=str(exc))
		report = SpotifyGraphReport(run_id=run_id, status='completed', intent=intent)
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		if urlsplit(state.url).netloc != 'open.spotify.com':
			report.status = 'needs_recovery'
			report.reason = f'current page is outside Spotify scope: {state.url}'
			return report

		for edge in self.graph.path_to(intent.goal_node.value):
			if edge.action == GraphAction.INPUT:
				resolved = await self._wait_for_resolution(browser_session, edge.locator, {'artist': intent.artist})
				resolution = resolved[1] if resolved else []
				if len(resolution) != 1:
					return self._refuse(report, edge, f'search locator resolved to {len(resolution)} candidates')
				index, _ = resolution[0]
				input_attempts = 0
				state = None
				for input_attempts in range(1, 3):
					if input_attempts > 1:
						resolved = await self._wait_for_resolution(browser_session, edge.locator, {'artist': intent.artist})
						resolution = resolved[1] if resolved else []
						if len(resolution) != 1:
							break
						index, _ = resolution[0]
					result = await self.tools.act(
						self.action_model.model_validate({'input': {'index': index, 'text': intent.artist, 'clear': True}}),
						browser_session,
						action_timeout=15,
					)
					if result.error:
						return self._refuse(report, edge, f'Browser Use input failed: {result.error}', index)
					state = await self._wait_for_search_query(browser_session, intent.artist)
					if state is not None:
						break
				if state is None:
					return self._refuse(report, edge, 'full Spotify search query did not stabilize after 2 attempts', index)
				report.events.append(
					GraphEvent(
						edge_id=edge.id,
						source=edge.source,
						target=edge.target,
						status='executed',
						selector_index=index,
						evidence={'url_path': urlsplit(state.url).path, 'input_attempts': input_attempts},
					)
				)
			elif edge.action == GraphAction.CLICK:
				resolved = await self._wait_for_resolution(browser_session, edge.locator, {'artist': intent.artist}, timeout=15)
				resolution = resolved[1] if resolved else []
				if len(resolution) != 1:
					return self._refuse(report, edge, f'canonical artist result resolved to {len(resolution)} candidates')
				index, node = resolution[0]
				href = str((node.attributes or {}).get('href') or '')
				result = await self.tools.act(
					self.action_model.model_validate({'click': {'index': index}}), browser_session, action_timeout=15
				)
				if result.error:
					return self._refuse(report, edge, f'Browser Use click failed: {result.error}', index)
				artist_page = await self._wait_for_artist_page(browser_session, intent.artist)
				if artist_page is None:
					return self._refuse(report, edge, 'canonical artist page predicate did not pass', index)
				report.artist_url = artist_page['url']
				report.events.append(
					GraphEvent(
						edge_id=edge.id,
						source=edge.source,
						target=edge.target,
						status='executed',
						selector_index=index,
						evidence={
							'candidate_href': href,
							'artist_url': artist_page['url'],
							'exact_heading': True,
							'popular_section': True,
						},
					)
				)
			elif edge.action == GraphAction.EXTRACT_RANKED:
				page = await self._read_artist_page(browser_session)
				rank = intent.track_rank or 1
				tracks = [SpotifyTrack.model_validate(track) for track in page.get('popular_tracks', [])] if page else []
				if rank > len(tracks):
					return self._refuse(report, edge, f'Popular exposes {len(tracks)} visible tracks; rank {rank} is unavailable')
				report.track = tracks[rank - 1]
				report.events.append(
					GraphEvent(
						edge_id=edge.id,
						source=edge.source,
						target=edge.target,
						status='extracted',
						evidence={
							'rank': rank,
							'visible_track_count': len(tracks),
							'track_name': report.track.name,
							'track_url': report.track.url,
						},
					)
				)
		report.terminal_node = intent.goal_node.value
		if self.trace_dir and report.artist_url:
			report.trace_path = str(await self._write_trace(report, browser_session))
		return report

	@staticmethod
	def _refuse(
		report: SpotifyGraphReport, edge: GraphEdge, reason: str, selector_index: int | None = None
	) -> SpotifyGraphReport:
		report.status = 'needs_recovery'
		report.reason = f'{edge.id}: {reason}'
		report.events.append(
			GraphEvent(
				edge_id=edge.id,
				source=edge.source,
				target=edge.target,
				status='needs_recovery',
				selector_index=selector_index,
				reason=reason,
			)
		)
		return report

	def _resolve(
		self,
		locator: GraphLocator | None,
		state: BrowserStateSummary,
		parameters: dict[str, str],
	) -> list[tuple[int, EnhancedDOMTreeNode]]:
		if locator is None:
			return []
		matches: list[tuple[int, EnhancedDOMTreeNode]] = []
		for index, node in state.dom_state.selector_map.items():
			if self._node_matches(locator, node, parameters):
				matches.append((index, node))
		return matches

	@staticmethod
	def _node_matches(locator: GraphLocator, node: EnhancedDOMTreeNode, parameters: dict[str, str]) -> bool:
		ax = node.ax_node
		values: dict[str, Any] = {
			'node_name': node.node_name,
			'ax_role': ax.role if ax else None,
			'ax_name': ax.name if ax else None,
		}
		values.update({f'attributes.{key}': value for key, value in (node.attributes or {}).items()})
		if any(_normalize(values.get(field)) != _normalize(expected) for field, expected in locator.fixed.items()):
			return False
		if any(
			_normalize(values.get(field)) != _normalize(parameters.get(parameter)) for field, parameter in locator.dynamic.items()
		):
			return False
		if locator.href_prefix and not str((node.attributes or {}).get('href') or '').startswith(locator.href_prefix):
			return False
		if locator.ancestor_role or locator.ancestor_name:
			parent = node.parent_node
			ancestor_match = False
			while parent is not None:
				parent_ax = parent.ax_node
				role_match = not locator.ancestor_role or _normalize(parent_ax.role if parent_ax else None) == _normalize(
					locator.ancestor_role
				)
				name_match = not locator.ancestor_name or _normalize(parent_ax.name if parent_ax else None) == _normalize(
					locator.ancestor_name
				)
				if role_match and name_match:
					ancestor_match = True
					break
				parent = parent.parent_node
			if not ancestor_match:
				return False
		return node.is_visible is not False

	async def _wait_for_state(
		self, browser_session: BrowserSession, predicate: Any, timeout: float = 8
	) -> BrowserStateSummary | None:
		deadline = time.monotonic() + timeout
		while time.monotonic() < deadline:
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			if predicate(state):
				return state
			await asyncio.sleep(0.2)
		return None

	async def _wait_for_search_query(
		self, browser_session: BrowserSession, artist: str, timeout: float = 10
	) -> BrowserStateSummary | None:
		return await self._wait_for_state(
			browser_session, lambda state: search_url_matches_artist(state.url, artist), timeout=timeout
		)

	async def _wait_for_resolution(
		self,
		browser_session: BrowserSession,
		locator: GraphLocator | None,
		parameters: dict[str, str],
		timeout: float = 10,
	) -> tuple[BrowserStateSummary, list[tuple[int, EnhancedDOMTreeNode]]] | None:
		"""Wait for a unique semantic target, not merely a navigation URL."""
		deadline = time.monotonic() + timeout
		while time.monotonic() < deadline:
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			resolution = self._resolve(locator, state, parameters)
			if len(resolution) == 1:
				return state, resolution
			await asyncio.sleep(0.2)
		return None

	async def _wait_for_artist_page(
		self, browser_session: BrowserSession, artist: str, timeout: float = 10
	) -> dict[str, Any] | None:
		deadline = time.monotonic() + timeout
		while time.monotonic() < deadline:
			page = await self._read_artist_page(browser_session)
			if (
				page
				and re.fullmatch(r'/artist/[A-Za-z0-9]+', urlsplit(page['url']).path)
				and _normalize(artist) in {_normalize(heading) for heading in page.get('headings', [])}
				and page.get('has_popular')
			):
				return page
			await asyncio.sleep(0.2)
		return None

	@staticmethod
	async def _read_artist_page(browser_session: BrowserSession) -> dict[str, Any] | None:
		try:
			page = await browser_session.must_get_current_page()
			payload = await page.evaluate(
				"""() => JSON.stringify((() => {
					const headings = Array.from(document.querySelectorAll('h1, h2'));
					const h1s = headings.filter(node => node.tagName === 'H1' && node.offsetParent).map(node => node.textContent.trim()).filter(Boolean);
					const popular = headings.find(node => node.tagName === 'H2' && node.textContent.trim().toLowerCase() === 'popular');
					let container = popular;
					while (container && container.querySelectorAll('a[href^="/track/"]').length === 0) container = container.parentElement;
					const seen = new Set();
					const tracks = container ? Array.from(container.querySelectorAll('a[href^="/track/"]')).filter(link => {
						if (!link.offsetParent || seen.has(link.getAttribute('href'))) return false;
						seen.add(link.getAttribute('href')); return true;
					}).map((link, index) => ({rank: index + 1, name: link.textContent.trim(), url: new URL(link.getAttribute('href'), location.origin).href})) : [];
					return {url: location.href, headings: h1s, has_popular: Boolean(popular), popular_tracks: tracks};
				})())"""
			)
			return json.loads(payload)
		except Exception:
			return None

	async def _write_trace(self, report: SpotifyGraphReport, browser_session: BrowserSession) -> Path:
		assert report.intent and report.artist_url
		assert self.trace_dir is not None
		page = await self._read_artist_page(browser_session)
		tracks = [SpotifyTrack.model_validate(track) for track in (page or {}).get('popular_tracks', [])]
		trace = SpotifyTrace(
			trace_id=report.run_id,
			created_at=datetime.now(timezone.utc).isoformat(),
			artist=report.intent.artist,
			terminal_node=report.terminal_node or report.intent.goal_node.value,
			search_locator=self.graph.edges[0].locator.fixed if self.graph.edges[0].locator else {},
			artist_result_locator=self.graph.edges[1].locator.fixed if self.graph.edges[1].locator else {},
			artist_url=report.artist_url,
			popular_tracks=tracks,
			verified=report.status == 'completed' and bool(tracks),
		)
		root = self.trace_dir.expanduser().resolve()
		root.mkdir(parents=True, exist_ok=True, mode=0o700)
		os.chmod(root, 0o700)
		path = root / f'{trace.trace_id}.json'
		path.write_text(trace.model_dump_json(indent=2) + '\n')
		os.chmod(path, 0o600)
		return path


async def run_spotify_task(
	task: str,
	*,
	graph: SpotifyProcedureGraph | None = None,
	trace_dir: Path | None = None,
	artist: str | None = None,
	track_rank: int | None = None,
	headless: bool = True,
	viewport_width: int = 1280,
	viewport_height: int = 800,
	linger_seconds: float = 0,
) -> SpotifyGraphReport:
	"""Run one graph task in a fresh Browser Use Chromium session."""
	procedure_graph = graph or spotify_graph_template()
	profile = BrowserProfile(
		headless=headless,
		user_data_dir=None,
		window_size=ViewportSize(width=viewport_width, height=viewport_height),
	)
	browser_session = BrowserSession(browser_profile=profile)
	await browser_session.start()
	try:
		await browser_session.navigate_to(SPOTIFY_ORIGIN)
		report = await SpotifyGraphExecutor(procedure_graph, trace_dir=trace_dir).run(
			browser_session,
			task,
			artist=artist,
			track_rank=track_rank,
		)
		if linger_seconds > 0:
			await asyncio.sleep(linger_seconds)
		return report
	finally:
		await browser_session.kill()


async def _main_async(args: argparse.Namespace) -> int:
	graph = SpotifyProcedureGraph.read(args.graph) if args.graph else spotify_graph_template()
	report = await run_spotify_task(
		args.task,
		graph=graph,
		trace_dir=args.trace_dir,
		artist=args.artist,
		track_rank=args.track_rank,
		headless=args.headless,
		viewport_width=args.viewport_width,
		viewport_height=args.viewport_height,
		linger_seconds=args.linger_seconds,
	)
	print(report.model_dump_json(indent=2))
	return 0 if report.status == 'completed' else 2


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('task', nargs='?')
	parser.add_argument('--graph', type=Path)
	parser.add_argument('--compile-traces', type=Path)
	parser.add_argument('--output-graph', type=Path)
	parser.add_argument('--artist')
	parser.add_argument('--track-rank', type=int)
	parser.add_argument('--trace-dir', type=Path, default=Path('./tmp/spotify-traces'))
	parser.add_argument('--viewport-width', type=int, default=1280)
	parser.add_argument('--viewport-height', type=int, default=800)
	parser.add_argument('--linger-seconds', type=float, default=0)
	parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
	args = parser.parse_args()
	if args.compile_traces:
		if not args.output_graph:
			parser.error('--compile-traces requires --output-graph')
		compiled = compile_spotify_graph(args.compile_traces)
		output = compiled.write(args.output_graph)
		print(json.dumps({'status': 'compiled', 'graph': str(output), 'training': compiled.training}, indent=2))
		return
	if not args.task:
		parser.error('task is required unless --compile-traces is used')
	raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == '__main__':
	main()
