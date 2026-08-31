"""Task-conditioned semantic graph replay for Spotify's public web player.

The graph is parameterized by artist and track rank. A task can stop on the
canonical artist, extract a ranked Popular track, or continue to verified
playback. Playback requires a signed-in Spotify session; follows, likes, and
playlist mutations remain out of scope.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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
	PLAY_TRACK = 'play_track'


class GraphAction(str, Enum):
	INPUT = 'input'
	CLICK = 'click'
	EXTRACT_RANKED = 'extract_ranked'
	PLAY_RANKED = 'play_ranked'


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


class StateEvidence(BaseModel):
	"""Browser-native evidence captured after one graph transition."""

	captured_at: str
	url: str | None = None
	title: str | None = None
	document: dict[str, Any] = Field(default_factory=dict)
	viewport: dict[str, Any] = Field(default_factory=dict)
	dom_sha256: str | None = None
	semantic_dom_sha256: str | None = None
	semantic_dom_chars: int = 0
	selector_count: int = 0
	screenshot_url: str | None = None
	screenshot_sha256: str | None = None
	browser_error_count: int = 0
	state_error: str | None = None
	capture_error: str | None = None
	delta_from_previous: dict[str, Any] = Field(default_factory=dict)


class GraphEvent(BaseModel):
	edge_id: str
	source: str
	target: str
	status: Literal['executed', 'extracted', 'needs_recovery']
	selector_index: int | None = None
	duration_ms: int = Field(default=0, ge=0)
	action_duration_ms: int = Field(default=0, ge=0)
	capture_duration_ms: int = Field(default=0, ge=0)
	evidence: dict[str, Any] = Field(default_factory=dict)
	state: StateEvidence | None = None
	reason: str | None = None


class SpotifyGraphReport(BaseModel):
	run_id: str
	status: Literal['completed', 'needs_recovery', 'disabled']
	intent: SpotifyTaskIntent | None = None
	terminal_node: str | None = None
	artist_url: str | None = None
	track: SpotifyTrack | None = None
	playback_started: bool = False
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
		has_play_intent = 'play' in tokens
		has_track_intent = has_play_intent or bool(
			tokens & {'track', 'song', 'popular', 'first', 'second', 'third', 'fourth', 'fifth'}
		)
		goal = (
			SpotifyGoal.PLAY_TRACK
			if has_play_intent
			else SpotifyGoal.POPULAR_TRACK
			if has_track_intent
			else SpotifyGoal.CANONICAL_ARTIST
		)
		rank = (track_rank or self._rank_from_task(task)) if has_track_intent else None
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
		GraphNode(
			id='play_track',
			description='Play the selected ranked track and verify Spotify entered a playing state.',
			keywords=['play', 'listen', 'start'],
			terminal=True,
			required_parameters=['artist', 'track_rank'],
			state_guard='selected Popular row exposes Pause after the click',
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
		GraphEdge(
			id='play_ranked_popular_track',
			source='popular_track',
			target='play_track',
			action=GraphAction.PLAY_RANKED,
			value_from='track_rank',
			postcondition='selected Popular row play control changes to Pause',
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
		self.tools.set_coordinate_clicking(True)
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
		run_started = time.monotonic()
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		if urlsplit(state.url).netloc != 'open.spotify.com':
			report.status = 'needs_recovery'
			report.reason = f'current page is outside Spotify scope: {state.url}'
			return report
		await self._record_event(
			report,
			browser_session,
			edge_id='verify_spotify_scope',
			source='spotify_home',
			target='spotify_home',
			status='executed',
			started_at=run_started,
			evidence={'origin': SPOTIFY_ORIGIN, 'scope_verified': True},
		)

		for edge in self.graph.path_to(intent.goal_node.value):
			edge_started = time.monotonic()
			if edge.action == GraphAction.INPUT:
				resolved = await self._wait_for_resolution(browser_session, edge.locator, {'artist': intent.artist})
				resolution = resolved[1] if resolved else []
				if len(resolution) != 1:
					return await self._refuse(
						report, browser_session, edge, f'search locator resolved to {len(resolution)} candidates', edge_started
					)
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
						return await self._refuse(
							report, browser_session, edge, f'Browser Use input failed: {result.error}', edge_started, index
						)
					state = await self._wait_for_search_query(browser_session, intent.artist)
					if state is not None:
						break
				if state is None:
					return await self._refuse(
						report,
						browser_session,
						edge,
						'full Spotify search query did not stabilize after 2 attempts',
						edge_started,
						index,
					)
				await self._record_event(
					report,
					browser_session,
					edge_id=edge.id,
					source=edge.source,
					target=edge.target,
					status='executed',
					started_at=edge_started,
					selector_index=index,
					evidence={'url_path': urlsplit(state.url).path, 'input_attempts': input_attempts},
				)
			elif edge.action == GraphAction.CLICK:
				resolved = await self._wait_for_resolution(browser_session, edge.locator, {'artist': intent.artist}, timeout=15)
				resolution = resolved[1] if resolved else []
				if len(resolution) != 1:
					return await self._refuse(
						report,
						browser_session,
						edge,
						f'canonical artist result resolved to {len(resolution)} candidates',
						edge_started,
					)
				index, node = resolution[0]
				href = str((node.attributes or {}).get('href') or '')
				result = await self.tools.act(
					self.action_model.model_validate({'click': {'index': index}}), browser_session, action_timeout=15
				)
				if result.error:
					return await self._refuse(
						report, browser_session, edge, f'Browser Use click failed: {result.error}', edge_started, index
					)
				artist_page = await self._wait_for_artist_page(browser_session, intent.artist)
				if artist_page is None:
					return await self._refuse(
						report,
						browser_session,
						edge,
						'canonical artist page predicate did not pass',
						edge_started,
						index,
					)
				report.artist_url = artist_page['url']
				await self._record_event(
					report,
					browser_session,
					edge_id=edge.id,
					source=edge.source,
					target=edge.target,
					status='executed',
					started_at=edge_started,
					selector_index=index,
					evidence={
						'candidate_href': href,
						'artist_url': artist_page['url'],
						'exact_heading': True,
						'popular_section': True,
					},
				)
			elif edge.action == GraphAction.EXTRACT_RANKED:
				page = await self._read_artist_page(browser_session)
				rank = intent.track_rank or 1
				tracks = [SpotifyTrack.model_validate(track) for track in page.get('popular_tracks', [])] if page else []
				if rank > len(tracks):
					return await self._refuse(
						report,
						browser_session,
						edge,
						f'Popular exposes {len(tracks)} visible tracks; rank {rank} is unavailable',
						edge_started,
					)
				report.track = tracks[rank - 1]
				await self._record_event(
					report,
					browser_session,
					edge_id=edge.id,
					source=edge.source,
					target=edge.target,
					status='extracted',
					started_at=edge_started,
					evidence={
						'rank': rank,
						'visible_track_count': len(tracks),
						'track_name': report.track.name,
						'track_url': report.track.url,
					},
				)
			elif edge.action == GraphAction.PLAY_RANKED:
				if report.track is None:
					return await self._refuse(
						report, browser_session, edge, 'ranked track was not extracted before playback', edge_started
					)
				target = await self._wait_for_play_target(browser_session, report.track)
				if target is None:
					return await self._refuse(
						report, browser_session, edge, 'play control did not resolve to one rendered target', edge_started
					)
				result = await self.tools.act(
					self.action_model.model_validate({'click': {'coordinate_x': target['x'], 'coordinate_y': target['y']}}),
					browser_session,
					action_timeout=15,
				)
				if result.error:
					return await self._refuse(
						report, browser_session, edge, f'Browser Use play click failed: {result.error}', edge_started
					)
				playback = await self._wait_for_playback(browser_session, report.track)
				if playback.get('auth_required'):
					return await self._refuse(
						report,
						browser_session,
						edge,
						'Spotify requires a signed-in session for playback; sign in once in the persistent demo profile and retry',
						edge_started,
					)
				if not playback.get('playing'):
					return await self._refuse(
						report, browser_session, edge, 'playback state did not change to Pause', edge_started
					)
				report.playback_started = True
				await self._record_event(
					report,
					browser_session,
					edge_id=edge.id,
					source=edge.source,
					target=edge.target,
					status='executed',
					started_at=edge_started,
					evidence={
						'rank': report.track.rank,
						'track_name': report.track.name,
						'track_url': report.track.url,
						'playback_started': True,
						'row_control': 'Pause',
						'current_geometry': {'x': target['x'], 'y': target['y']},
					},
				)
		report.terminal_node = intent.goal_node.value
		if self.trace_dir and report.artist_url:
			report.trace_path = str(await self._write_trace(report, browser_session))
		return report

	async def _record_event(
		self,
		report: SpotifyGraphReport,
		browser_session: BrowserSession,
		*,
		edge_id: str,
		source: str,
		target: str,
		status: Literal['executed', 'extracted', 'needs_recovery'],
		started_at: float,
		selector_index: int | None = None,
		evidence: dict[str, Any] | None = None,
		reason: str | None = None,
	) -> None:
		action_duration_ms = max(0, round((time.monotonic() - started_at) * 1000))
		capture_started = time.monotonic()
		state = await self._capture_state_evidence(report.run_id, target, len(report.events), browser_session)
		capture_duration_ms = max(0, round((time.monotonic() - capture_started) * 1000))
		previous = report.events[-1].state if report.events else None
		if previous is not None:
			state.delta_from_previous = {
				'previous_state': report.events[-1].target,
				'url_changed': state.url != previous.url,
				'dom_changed': state.dom_sha256 != previous.dom_sha256,
				'semantic_dom_changed': state.semantic_dom_sha256 != previous.semantic_dom_sha256,
				'screenshot_changed': state.screenshot_sha256 != previous.screenshot_sha256,
				'selectors_added': state.selector_count - previous.selector_count,
			}
		report.events.append(
			GraphEvent(
				edge_id=edge_id,
				source=source,
				target=target,
				status=status,
				selector_index=selector_index,
				duration_ms=action_duration_ms + capture_duration_ms,
				action_duration_ms=action_duration_ms,
				capture_duration_ms=capture_duration_ms,
				evidence=evidence or {},
				state=state,
				reason=reason,
			)
		)

	async def _capture_state_evidence(
		self,
		run_id: str,
		state_id: str,
		ordinal: int,
		browser_session: BrowserSession,
	) -> StateEvidence:
		captured_at = datetime.now(timezone.utc).isoformat()
		try:
			state = await browser_session.get_browser_state_summary(include_screenshot=True)
		except Exception as exc:
			return StateEvidence(captured_at=captured_at, capture_error=f'browser state unavailable: {type(exc).__name__}')

		semantic_dom = state.dom_state.llm_representation()
		evidence = StateEvidence(
			captured_at=captured_at,
			url=state.url,
			title=state.title,
			semantic_dom_sha256=hashlib.sha256(semantic_dom.encode()).hexdigest(),
			semantic_dom_chars=len(semantic_dom),
			selector_count=len(state.dom_state.selector_map),
			browser_error_count=len(state.browser_errors),
			state_error=state.state_error,
		)
		if state.page_info:
			evidence.viewport = state.page_info.model_dump(mode='json')

		try:
			page = await browser_session.must_get_current_page()
			payload_text = await page.evaluate(
				"""() => JSON.stringify({
					document: {
						ready_state: document.readyState,
						visibility_state: document.visibilityState,
						content_type: document.contentType,
						language: document.documentElement.lang || null,
					},
					viewport: {
						width: window.innerWidth,
						height: window.innerHeight,
						device_pixel_ratio: window.devicePixelRatio,
						scroll_x: window.scrollX,
						scroll_y: window.scrollY,
						document_width: document.documentElement.scrollWidth,
						document_height: document.documentElement.scrollHeight,
					},
					outer_html: document.documentElement.outerHTML,
				})"""
			)
			payload = json.loads(payload_text)
			outer_html = str(payload.pop('outer_html', ''))
			evidence.document = payload.get('document', {})
			evidence.viewport = {**evidence.viewport, **payload.get('viewport', {})}
			evidence.dom_sha256 = hashlib.sha256(outer_html.encode()).hexdigest()
		except Exception as exc:
			evidence.capture_error = f'rendered DOM metadata unavailable: {type(exc).__name__}'

		if state.screenshot and self.trace_dir:
			try:
				screenshot = base64.b64decode(state.screenshot, validate=True)
				filename = f'state-{ordinal:02d}-{re.sub(r"[^a-z0-9_-]+", "-", state_id.casefold())}.png'
				root = self.trace_dir.expanduser().resolve()
				run_dir = root / run_id
				run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
				os.chmod(root, 0o700)
				os.chmod(run_dir, 0o700)
				path = run_dir / filename
				path.write_bytes(screenshot)
				os.chmod(path, 0o600)
				evidence.screenshot_sha256 = hashlib.sha256(screenshot).hexdigest()
				evidence.screenshot_url = f'/api/spotify/artifacts/{run_id}/{filename}'
			except Exception as exc:
				evidence.capture_error = f'screenshot unavailable: {type(exc).__name__}'
		return evidence

	async def _refuse(
		self,
		report: SpotifyGraphReport,
		browser_session: BrowserSession,
		edge: GraphEdge,
		reason: str,
		started_at: float,
		selector_index: int | None = None,
	) -> SpotifyGraphReport:
		report.status = 'needs_recovery'
		report.reason = f'{edge.id}: {reason}'
		await self._record_event(
			report,
			browser_session,
			edge_id=edge.id,
			source=edge.source,
			target=edge.target,
			status='needs_recovery',
			started_at=started_at,
			selector_index=selector_index,
			reason=reason,
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

	@staticmethod
	async def _play_target(browser_session: BrowserSession, track: SpotifyTrack) -> dict[str, int] | None:
		try:
			page = await browser_session.must_get_current_page()
			payload = await page.evaluate(
				"""(trackPath) => JSON.stringify((() => {
					const visible = node => Boolean(node && node.offsetParent);
					const popular = Array.from(document.querySelectorAll('h2')).find(node =>
						visible(node) && node.textContent.trim().toLowerCase() === 'popular'
					);
					let container = popular;
					while (container && container.querySelectorAll('a[href^="/track/"]').length === 0) {
						container = container.parentElement;
					}
					const links = Array.from(container?.querySelectorAll('a[href^="/track/"]') || []).filter(node =>
						new URL(node.getAttribute('href'), location.origin).pathname === trackPath && visible(node)
					);
					const rows = [...new Set(links.map(link => link.closest('[role="row"]')).filter(Boolean))];
					const targets = rows.map(row => Array.from(row.querySelectorAll('button')).find(button =>
						/^play\\b/i.test(button.getAttribute('aria-label') || '')
					)).filter(Boolean);
					if (targets.length !== 1) return null;
					const rect = targets[0].getBoundingClientRect();
					if (rect.width <= 0 || rect.height <= 0) return null;
					return {x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2)};
				})())""",
				urlsplit(track.url).path,
			)
			return json.loads(payload)
		except Exception:
			return None

	async def _wait_for_play_target(
		self, browser_session: BrowserSession, track: SpotifyTrack, timeout: float = 10
	) -> dict[str, int] | None:
		deadline = time.monotonic() + timeout
		while time.monotonic() < deadline:
			target = await self._play_target(browser_session, track)
			if target is not None:
				return target
			await asyncio.sleep(0.2)
		return None

	@staticmethod
	async def _playback_state(browser_session: BrowserSession, track: SpotifyTrack) -> dict[str, bool]:
		try:
			page = await browser_session.must_get_current_page()
			payload = await page.evaluate(
				"""(trackPath, trackName) => JSON.stringify((() => {
					const visible = node => Boolean(node && node.offsetParent);
					const link = Array.from(document.querySelectorAll('a[href^="/track/"]'))
						.find(node => new URL(node.getAttribute('href'), location.origin).pathname === trackPath && visible(node));
					const row = link?.closest('[role="row"]');
					const rowControl = row ? Array.from(row.querySelectorAll('button')).find(button =>
						/^(play|pause)\\b/i.test(button.getAttribute('aria-label') || '')
					) : null;
					const label = rowControl?.getAttribute('aria-label') || '';
					const nowPlaying = document.querySelector('[aria-label^="Now playing:"]');
					const nowPlayingLabel = nowPlaying?.getAttribute('aria-label') || '';
					const globalPause = document.querySelector('[data-testid="control-button-playpause"][aria-label="Pause"]');
					const text = document.body.innerText.toLowerCase();
					return {
						playing: /^pause\\b/i.test(label) || Boolean(
							globalPause && !globalPause.disabled && nowPlayingLabel.toLowerCase().includes(trackName.toLowerCase())
						),
						auth_required: text.includes('start listening with a free spotify account')
							|| text.includes('sign up to start listening'),
					};
				})())""",
				urlsplit(track.url).path,
				track.name,
			)
			return json.loads(payload)
		except Exception:
			return {'playing': False, 'auth_required': False}

	async def _wait_for_playback(
		self, browser_session: BrowserSession, track: SpotifyTrack, timeout: float = 8
	) -> dict[str, bool]:
		deadline = time.monotonic() + timeout
		last = {'playing': False, 'auth_required': False}
		while time.monotonic() < deadline:
			last = await self._playback_state(browser_session, track)
			if last.get('playing') or last.get('auth_required'):
				return last
			await asyncio.sleep(0.2)
		return last

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
	user_data_dir: Path | None = None,
) -> SpotifyGraphReport:
	"""Run one graph task in a fresh Browser Use Chromium session."""
	procedure_graph = graph or spotify_graph_template()
	profile = BrowserProfile(
		headless=headless,
		user_data_dir=user_data_dir,
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
