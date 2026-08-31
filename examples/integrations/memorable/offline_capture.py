"""Capture high-fidelity Browser Use runs for deterministic offline analysis.

The collector is deliberately outside the agent core. It composes the existing
pre-step and post-step hooks, writes only to a caller-selected local directory,
and never lets a capture failure change the agent's result.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str

from browser_use.agent.views import ActionResult, AgentHistoryList, AgentOutput
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import DOMInteractedElement, EnhancedDOMTreeNode

if TYPE_CHECKING:
	from browser_use import Agent

logger = logging.getLogger(__name__)

CAPTURE_SCHEMA_VERSION = '0.3.0'
KILL_SWITCH = 'BROWSER_USE_OFFLINE_CAPTURE'
SECRET_PATTERN = re.compile(
	r'(?i)(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|'
	r'(?:bearer|token|password|api[_-]?key)[=:\s]+[^\s"\']{12,})'
)
DYNAMIC_IDENTIFIER_PATTERN = re.compile(r'(?:^|[-_])[a-f0-9]{8,}(?:$|[-_])', re.IGNORECASE)


class OfflineCaptureOptions(BaseModel):
	"""Controls which private artifacts are retained for one run."""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	output_dir: Path
	include_screenshots: bool = True
	include_dom_text: bool = True
	include_eval_dom: bool = True
	include_full_dom: bool = True
	include_candidates: bool = True
	include_rendered_page: bool = True
	include_conversations: bool = False
	include_har: bool = False
	include_video: bool = False
	include_downloads: bool = False
	post_state_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
	max_candidate_text_chars: int = Field(default=2000, ge=0, le=100_000)
	max_rendered_page_chars: int = Field(default=5_000_000, ge=10_000, le=50_000_000)
	expected_success_text: str | None = None
	run_label: str | None = None
	chaos_seed: str | None = None


class ArtifactRecord(BaseModel):
	"""Integrity metadata for an artifact in a completed bundle."""

	path: str
	bytes: int
	sha256: str


class RunManifest(BaseModel):
	"""Self-describing metadata written last when a run bundle is finalized."""

	schema_version: str = CAPTURE_SCHEMA_VERSION
	run_id: str
	state: str
	created_at: str
	finished_at: str | None = None
	duration_seconds: float | None = None
	run_label: str | None = None
	chaos_seed: str | None = None
	agent_id: str | None = None
	session_id: str | None = None
	task_id: str | None = None
	task: str | None = None
	model: str | None = None
	browser_use_version: str | None = None
	python_version: str
	platform: str
	max_actions_per_step: int | None = None
	capture_options: dict[str, Any]
	terminal: dict[str, Any] = Field(default_factory=dict)
	counts: dict[str, int] = Field(default_factory=dict)
	exception: dict[str, str] | None = None
	capture_errors: list[dict[str, Any]] = Field(default_factory=list)
	privacy: dict[str, Any] = Field(default_factory=dict)
	artifacts: list[ArtifactRecord] = Field(default_factory=list)


class _CapturedState(BaseModel):
	"""Small in-memory index used to derive transitions after the run."""

	step: int
	phase: str
	state_path: str
	url: str
	title: str
	dom_sha256: str | None = None
	screenshot_sha256: str | None = None
	html_sha256: str | None = None
	rendered_text_sha256: str | None = None
	expected_success_text_present: bool | None = None
	viewport: dict[str, Any] | None = None
	candidates: list[dict[str, Any]] = Field(default_factory=list)
	controls: list[dict[str, Any]] = Field(default_factory=list)


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
	if isinstance(value, BaseModel):
		return value.model_dump(mode='json', exclude_none=True)
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, Enum):
		return value.value
	if is_dataclass(value) and not isinstance(value, type):
		return asdict(value)
	if isinstance(value, (set, frozenset, tuple)):
		return list(value)
	if hasattr(value, '__json__'):
		return getattr(value, '__json__')()
	return repr(value)


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
	return json.dumps(
		value,
		default=_json_default,
		ensure_ascii=False,
		indent=2 if pretty else None,
		sort_keys=pretty,
	).encode('utf-8')


def _json_safe(value: Any) -> Any:
	return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))


def _sha256(data: bytes) -> str:
	return hashlib.sha256(data).hexdigest()


def _callable_result(callback: Callable[..., Any], *args: Any) -> Awaitable[Any] | None:
	result = callback(*args)
	return result if inspect.isawaitable(result) else None


def _action_name(action: dict[str, Any]) -> str:
	return next(iter(action), '') if action else ''


def _action_index(action: dict[str, Any]) -> int | None:
	if not action:
		return None
	params = action.get(_action_name(action))
	if isinstance(params, dict) and isinstance(params.get('index'), int):
		return params['index']
	return None


def _result_status(result: dict[str, Any] | None) -> str:
	if not result:
		return 'outcome_unavailable'
	if result.get('error'):
		return 'reported_error'
	if result.get('is_done') is True:
		return 'terminal_success' if result.get('success') is True else 'terminal_failure'
	return 'no_reported_error'


def _semantic_key(candidate: dict[str, Any]) -> str:
	attributes = candidate.get('attributes') or {}
	parts = [
		candidate.get('node_name'),
		candidate.get('ax_role'),
		candidate.get('ax_name') or candidate.get('meaningful_text'),
		attributes.get('name'),
		attributes.get('type'),
		attributes.get('placeholder'),
		attributes.get('href'),
	]
	return '|'.join(str(part or '').strip().lower() for part in parts)


def _normalized_identity(value: Any) -> str:
	return re.sub(r'\s+', ' ', str(value or '').strip().lower())


def _target_alignment_base(step: dict[str, Any]) -> str:
	"""Choose a cross-run action slot without depending on absolute step number."""
	action_name = _normalized_identity(step.get('action_name')) or 'unknown'
	candidate = step.get('selected_candidate') or {}
	attributes = candidate.get('attributes') or {}
	for field in ('autocomplete', 'name'):
		value = _normalized_identity(attributes.get(field))
		if value:
			return f'{action_name}|{field}:{value}'
	input_type = _normalized_identity(attributes.get('type'))
	if input_type and (candidate.get('node_name') == 'INPUT' or input_type == 'submit'):
		return f'{action_name}|type:{input_type}'
	ax_role = _normalized_identity(candidate.get('ax_role'))
	ax_name = _normalized_identity(candidate.get('ax_name'))
	if ax_role or ax_name:
		return f'{action_name}|ax:{ax_role}|{ax_name}'
	return f'{action_name}|unresolved'


def _control_identity(control: dict[str, Any], ordinal: int) -> str:
	for field in ('name', 'id', 'ariaLabel'):
		value = _normalized_identity(control.get(field))
		if value:
			return f'{field}:{value}'
	return '|'.join(
		[
			_normalized_identity(control.get('tag')),
			_normalized_identity(control.get('type')),
			f'ordinal:{ordinal}',
		]
	)


def _control_state_diff(pre: _CapturedState | None, post: _CapturedState | None) -> list[dict[str, Any]]:
	if pre is None or post is None:
		return []
	pre_controls = {_control_identity(control, index): control for index, control in enumerate(pre.controls)}
	post_controls = {_control_identity(control, index): control for index, control in enumerate(post.controls)}
	changes: list[dict[str, Any]] = []
	for identity in sorted(pre_controls.keys() & post_controls.keys()):
		before = pre_controls[identity]
		after = post_controls[identity]
		changed_fields = {
			field: {'before': before.get(field), 'after': after.get(field)}
			for field in ('value', 'checked', 'disabled', 'hidden')
			if before.get(field) != after.get(field)
		}
		if changed_fields:
			changes.append({'control': identity, 'changes': changed_fields})
	return changes


def _identity_quality(candidate: dict[str, Any] | None, same_name_count: int) -> dict[str, Any]:
	if not candidate:
		return {'score': 0.0, 'evidence': [], 'risks': ['target_not_in_pre_candidates']}
	attributes = candidate.get('attributes') or {}
	evidence: list[str] = []
	risks: list[str] = []
	score = 0.0
	if candidate.get('ax_role'):
		evidence.append('accessibility_role')
		score += 0.18
	if candidate.get('ax_name'):
		evidence.append('accessibility_name')
		score += 0.26
	if attributes.get('name'):
		evidence.append('name_attribute')
		score += 0.16
	if attributes.get('aria-label'):
		evidence.append('aria_label')
		score += 0.16
	if attributes.get('placeholder'):
		evidence.append('placeholder')
		score += 0.08
	if candidate.get('ancestor_context'):
		evidence.append('semantic_ancestors')
		score += 0.12
	identifier = attributes.get('id', '')
	if identifier and not DYNAMIC_IDENTIFIER_PATTERN.search(identifier):
		evidence.append('apparently_stable_id')
		score += 0.08
	if same_name_count > 1:
		risks.append(f'duplicate_accessible_name:{same_name_count}')
		score -= min(0.24, 0.08 * (same_name_count - 1))
	if not candidate.get('bounds'):
		risks.append('missing_rendered_bounds')
	return {'score': round(max(0.0, min(1.0, score)), 3), 'evidence': evidence, 'risks': risks}


class OfflineRunCapture:
	"""Collect one Browser Use run into an atomic local bundle."""

	def __init__(self, options: OfflineCaptureOptions):
		self.options = options
		self.enabled = os.environ.get(KILL_SWITCH, '1') != '0'
		self.run_id = uuid7str()
		self.output_dir = options.output_dir.expanduser().resolve()
		self.partial_dir = self.output_dir / f'.{self.run_id}.partial'
		self.final_dir = self.output_dir / self.run_id
		self.created_at = _utc_now()
		self.started_monotonic: float | None = None
		self._agent: Agent | None = None
		self._history: AgentHistoryList[Any] | None = None
		self._states: dict[int, dict[str, _CapturedState]] = defaultdict(dict)
		self._capture_errors: list[dict[str, Any]] = []
		self._original_new_step_callback: Callable[..., Any] | None = None
		self._original_conversation_path: Any = None
		self._original_har_path: Any = None
		self._original_video_dir: Any = None
		self._original_downloads_path: Any = None
		self._finalized = False

	async def run_agent(self, agent: Agent, **run_kwargs: Any) -> AgentHistoryList[Any]:
		"""Run an agent with capture installed, preserving caller hooks and exceptions."""
		if not self.enabled:
			return await agent.run(**run_kwargs)

		self._agent = agent
		self._begin(agent)
		self._install_agent_capture(agent)
		caller_step_end = run_kwargs.pop('on_step_end', None)

		async def combined_step_end(active_agent: Agent) -> None:
			await self._safe_capture_post_step(active_agent)
			if caller_step_end is not None:
				awaitable = _callable_result(caller_step_end, active_agent)
				if awaitable is not None:
					await awaitable

		history: AgentHistoryList[Any] | None = None
		run_exception: BaseException | None = None
		try:
			history = await agent.run(on_step_end=combined_step_end, **run_kwargs)
			self._history = history
			return history
		except BaseException as exc:
			run_exception = exc
			self._history = getattr(agent, 'history', None)
			raise
		finally:
			self._restore_agent(agent)
			try:
				self._finalize(agent, self._history, run_exception)
			except Exception as exc:
				self._record_capture_error('finalize', exc)
				logger.warning('Offline Browser Use capture could not be finalized: %s', exc)

	def _begin(self, agent: Agent) -> None:
		self.started_monotonic = time.monotonic()
		self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
		os.chmod(self.output_dir, 0o700)
		self.partial_dir.mkdir(mode=0o700)
		self._write_json(
			'manifest.partial.json',
			{
				'schema_version': CAPTURE_SCHEMA_VERSION,
				'run_id': self.run_id,
				'state': 'running',
				'created_at': self.created_at,
				'task': getattr(agent, 'task', None),
			},
		)

	def _install_agent_capture(self, agent: Agent) -> None:
		self._original_new_step_callback = agent.register_new_step_callback

		async def combined_new_step_callback(
			browser_state: BrowserStateSummary, model_output: AgentOutput, step_number: int
		) -> None:
			await self._safe_capture_pre_step(browser_state, model_output, step_number)
			if self._original_new_step_callback is not None:
				awaitable = _callable_result(self._original_new_step_callback, browser_state, model_output, step_number)
				if awaitable is not None:
					await awaitable

		agent.register_new_step_callback = combined_new_step_callback

		self._original_conversation_path = agent.settings.save_conversation_path
		if self.options.include_conversations:
			conversation_dir = self.partial_dir / 'conversations'
			conversation_dir.mkdir(mode=0o700)
			agent.settings.save_conversation_path = conversation_dir

		profile = agent.browser_session.browser_profile
		self._original_har_path = profile.record_har_path
		self._original_video_dir = profile.record_video_dir
		self._original_downloads_path = profile.downloads_path
		if self.options.include_har:
			profile.record_har_path = self.partial_dir / 'network.har'
		if self.options.include_video:
			video_dir = self.partial_dir / 'video'
			video_dir.mkdir(mode=0o700)
			profile.record_video_dir = video_dir
		if self.options.include_downloads:
			downloads_dir = self.partial_dir / 'downloads'
			downloads_dir.mkdir(mode=0o700)
			profile.downloads_path = downloads_dir

	def _restore_agent(self, agent: Agent) -> None:
		agent.register_new_step_callback = self._original_new_step_callback
		agent.settings.save_conversation_path = self._original_conversation_path
		profile = agent.browser_session.browser_profile
		profile.record_har_path = self._original_har_path
		profile.record_video_dir = self._original_video_dir
		profile.downloads_path = self._original_downloads_path

	async def _safe_capture_pre_step(
		self, browser_state: BrowserStateSummary, model_output: AgentOutput, step_number: int
	) -> None:
		try:
			rendered_page = await self._safe_capture_rendered_page(step_number, 'pre')
			captured = self._capture_state(browser_state, step_number, 'pre', rendered_page)
			self._states[step_number]['pre'] = captured
			self._write_json(
				f'steps/{step_number:03d}/pre.model_output.json',
				model_output.model_dump(mode='json', exclude_none=True),
			)
		except Exception as exc:
			self._record_capture_error('pre_step', exc, step_number)

	async def _safe_capture_post_step(self, agent: Agent) -> None:
		step_number = max(1, int(agent.state.n_steps) - 1)
		try:
			browser_state = await asyncio.wait_for(
				agent.browser_session.get_browser_state_summary(
					include_screenshot=self.options.include_screenshots,
					include_recent_events=agent.include_recent_events,
				),
				timeout=self.options.post_state_timeout_seconds,
			)
			rendered_page = await self._safe_capture_rendered_page(step_number, 'post')
			captured = self._capture_state(browser_state, step_number, 'post', rendered_page)
			self._states[step_number]['post'] = captured
			if agent.state.last_result is not None:
				self._write_json(
					f'steps/{step_number:03d}/post.results.json',
					[result.model_dump(mode='json', exclude_none=True) for result in agent.state.last_result],
				)
		except Exception as exc:
			self._record_capture_error('post_step', exc, step_number)

	async def _safe_capture_rendered_page(self, step: int, phase: str) -> dict[str, Any] | None:
		"""Capture browser-native page evidence omitted by the simplified DOM tree."""
		if not self.options.include_rendered_page or self._agent is None:
			return None
		try:
			page = await self._agent.browser_session.must_get_current_page()
			payload_text = await asyncio.wait_for(
				page.evaluate(
					"""(maxChars) => {
						const root = document.documentElement;
						const body = document.body;
						const html = root ? root.outerHTML : '';
						const renderedText = body ? body.innerText : '';
						const rect = (element) => {
							const value = element.getBoundingClientRect();
							return {x: value.x, y: value.y, width: value.width, height: value.height};
						};
						const describe = (element) => element ? {
							tag: element.tagName,
							id: element.id || null,
							name: element.getAttribute('name'),
							type: element.getAttribute('type'),
							ariaLabel: element.getAttribute('aria-label'),
							role: element.getAttribute('role'),
							text: (element.innerText || '').slice(0, 500),
							value: 'value' in element ? element.value : null,
							checked: 'checked' in element ? element.checked : null,
							disabled: 'disabled' in element ? element.disabled : null,
							hidden: element.hidden,
							rect: rect(element),
						} : null;
						return {
							capturedAt: new Date().toISOString(),
							document: {
								url: location.href,
								title: document.title,
								readyState: document.readyState,
								visibilityState: document.visibilityState,
								contentType: document.contentType,
								characterSet: document.characterSet,
							},
							viewport: {
								width: window.innerWidth,
								height: window.innerHeight,
								outerWidth: window.outerWidth,
								outerHeight: window.outerHeight,
								devicePixelRatio: window.devicePixelRatio,
								scrollX: window.scrollX,
								scrollY: window.scrollY,
								documentWidth: root ? root.scrollWidth : null,
								documentHeight: root ? root.scrollHeight : null,
							},
							activeElement: describe(document.activeElement),
							controls: Array.from(document.querySelectorAll(
								'input, select, textarea, button, [contenteditable="true"]'
							)).map(describe),
							html: {
								content: html.slice(0, maxChars),
								sourceChars: html.length,
								truncated: html.length > maxChars,
							},
							renderedText: {
								content: renderedText.slice(0, maxChars),
								sourceChars: renderedText.length,
								truncated: renderedText.length > maxChars,
							},
						};
					}""",
					self.options.max_rendered_page_chars,
				),
				timeout=self.options.post_state_timeout_seconds,
			)
			payload = json.loads(payload_text)
			html_record = payload.pop('html')
			text_record = payload.pop('renderedText')
			html = str(html_record.pop('content', ''))
			rendered_text = str(text_record.pop('content', ''))
			html_bytes = html.encode('utf-8')
			text_bytes = rendered_text.encode('utf-8')
			self._write_bytes(f'steps/{step:03d}/{phase}.html.gz', gzip.compress(html_bytes, compresslevel=6))
			self._write_bytes(f'steps/{step:03d}/{phase}.rendered.txt', text_bytes)
			metadata = {
				**payload,
				'html': {**html_record, 'capturedBytes': len(html_bytes), 'sha256': _sha256(html_bytes)},
				'renderedText': {**text_record, 'capturedBytes': len(text_bytes), 'sha256': _sha256(text_bytes)},
			}
			if self.options.expected_success_text is not None:
				metadata['expectedSuccessTextPresent'] = self.options.expected_success_text in rendered_text
			self._write_json(f'steps/{step:03d}/{phase}.page.json', metadata)
			return metadata
		except Exception as exc:
			self._record_capture_error(f'{phase}_rendered_page', exc, step)
			return None

	def _capture_state(
		self, state: BrowserStateSummary, step: int, phase: str, rendered_page: dict[str, Any] | None = None
	) -> _CapturedState:
		step_dir = self.partial_dir / 'steps' / f'{step:03d}'
		step_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

		dom_text: str | None = None
		dom_sha256: str | None = None
		if self.options.include_dom_text:
			dom_text = state.dom_state.llm_representation()
			dom_bytes = dom_text.encode('utf-8')
			dom_sha256 = _sha256(dom_bytes)
			self._write_bytes(f'steps/{step:03d}/{phase}.dom.txt', dom_bytes)

		if self.options.include_eval_dom:
			eval_text = state.dom_state.eval_representation()
			self._write_bytes(f'steps/{step:03d}/{phase}.eval.dom.txt', eval_text.encode('utf-8'))

		if self.options.include_full_dom and state.dom_state._root is not None:
			full_dom = state.dom_state._root.__json__()
			self._write_gzip_json(f'steps/{step:03d}/{phase}.full.dom.json.gz', full_dom)

		candidates: list[dict[str, Any]] = []
		if self.options.include_candidates:
			for selector_index, node in sorted(state.dom_state.selector_map.items()):
				candidates.append(self._candidate_record(selector_index, node))
			self._write_json(f'steps/{step:03d}/{phase}.candidates.json', candidates)

		screenshot_sha256: str | None = None
		if self.options.include_screenshots and state.screenshot:
			screenshot = base64.b64decode(state.screenshot)
			screenshot_sha256 = _sha256(screenshot)
			self._write_bytes(f'steps/{step:03d}/{phase}.png', screenshot)

		page_info = state.page_info.model_dump(mode='json') if state.page_info else None
		state_record = {
			'captured_at': _utc_now(),
			'step': step,
			'phase': phase,
			'url': state.url,
			'title': state.title,
			'tabs': [tab.model_dump(mode='json') for tab in state.tabs],
			'page_info': page_info,
			'legacy_scroll': {'pixels_above': state.pixels_above, 'pixels_below': state.pixels_below},
			'browser_errors': list(state.browser_errors),
			'recent_events': state.recent_events,
			'pending_network_requests': [_json_safe(request) for request in state.pending_network_requests],
			'pagination_buttons': [_json_safe(button) for button in state.pagination_buttons],
			'closed_popup_messages': list(state.closed_popup_messages),
			'is_pdf_viewer': state.is_pdf_viewer,
			'state_error': state.state_error,
			'dom_sha256': dom_sha256,
			'screenshot_sha256': screenshot_sha256,
			'candidate_count': len(candidates),
			'rendered_page': rendered_page,
		}
		state_path = f'steps/{step:03d}/{phase}.json'
		self._write_json(state_path, state_record)
		return _CapturedState(
			step=step,
			phase=phase,
			state_path=state_path,
			url=state.url,
			title=state.title,
			dom_sha256=dom_sha256,
			screenshot_sha256=screenshot_sha256,
			html_sha256=(rendered_page or {}).get('html', {}).get('sha256'),
			rendered_text_sha256=(rendered_page or {}).get('renderedText', {}).get('sha256'),
			expected_success_text_present=(rendered_page or {}).get('expectedSuccessTextPresent'),
			viewport=page_info,
			candidates=candidates,
			controls=(rendered_page or {}).get('controls', []),
		)

	def _candidate_record(self, selector_index: int, node: EnhancedDOMTreeNode) -> dict[str, Any]:
		interacted = DOMInteractedElement.load_from_enhanced_dom_tree(node).to_dict()
		ax_node = node.ax_node
		snapshot = node.snapshot_node
		meaningful_text = node.get_meaningful_text_for_llm()[: self.options.max_candidate_text_chars]
		ancestors: list[dict[str, Any]] = []
		parent = node.parent_node
		while parent is not None and len(ancestors) < 6:
			parent_attributes = parent.attributes or {}
			ancestors.append(
				{
					'node_name': parent.node_name,
					'ax_role': parent.ax_node.role if parent.ax_node else None,
					'ax_name': parent.ax_node.name if parent.ax_node else None,
					'id': parent_attributes.get('id'),
					'class': parent_attributes.get('class'),
					'aria_label': parent_attributes.get('aria-label'),
				}
			)
			parent = parent.parent_node
		return {
			'selector_index': selector_index,
			**interacted,
			'meaningful_text': meaningful_text,
			'ax_role': ax_node.role if ax_node else None,
			'ax_description': ax_node.description if ax_node else None,
			'ax_ignored': ax_node.ignored if ax_node else None,
			'ax_properties': _json_safe(ax_node.properties) if ax_node and ax_node.properties else None,
			'is_visible': node.is_visible,
			'is_scrollable': node.is_scrollable,
			'is_actually_scrollable': node.is_actually_scrollable,
			'has_js_click_listener': node.has_js_click_listener,
			'absolute_position': node.absolute_position.to_dict() if node.absolute_position else None,
			'client_rect': snapshot.clientRects.to_dict() if snapshot and snapshot.clientRects else None,
			'scroll_rect': snapshot.scrollRects.to_dict() if snapshot and snapshot.scrollRects else None,
			'computed_styles': snapshot.computed_styles if snapshot else None,
			'paint_order': snapshot.paint_order if snapshot else None,
			'stacking_contexts': snapshot.stacking_contexts if snapshot else None,
			'ancestor_context': ancestors,
		}

	def _finalize(self, agent: Agent, history: AgentHistoryList[Any] | None, run_exception: BaseException | None) -> None:
		if self._finalized or not self.partial_dir.exists():
			return
		self._finalized = True

		if history is not None:
			self._write_json('history.json', history.model_dump(sensitive_data=agent.sensitive_data))
			usage = history.usage.model_dump(mode='json') if history.usage else None
			self._write_json('usage.json', usage)
			self._write_events(history)
			self._write_json('derived.json', self._derive_run(history))
		else:
			self._write_json('history.json', {'history': []})
			self._write_json('usage.json', None)
			self._write_bytes('events.jsonl', b'')
			self._write_json('derived.json', {'steps': [], 'warnings': ['history_unavailable']})

		try:
			self._write_json('agent_state.json', agent.state.model_dump(mode='json', exclude_none=True))
		except Exception as exc:
			self._record_capture_error('agent_state', exc)

		self._harden_permissions()
		secret_scan = self._secret_scan()
		duration = time.monotonic() - self.started_monotonic if self.started_monotonic is not None else None
		manifest = RunManifest(
			run_id=self.run_id,
			state='failed' if run_exception else 'completed',
			created_at=self.created_at,
			finished_at=_utc_now(),
			duration_seconds=round(duration, 6) if duration is not None else None,
			run_label=self.options.run_label,
			chaos_seed=self.options.chaos_seed,
			agent_id=str(getattr(agent, 'id', '')) or None,
			session_id=str(getattr(agent, 'session_id', '')) or None,
			task_id=str(getattr(agent, 'task_id', '')) or None,
			task=getattr(agent, 'task', None),
			model=str(getattr(getattr(agent, 'llm', None), 'model', '') or '') or None,
			browser_use_version=self._browser_use_version(),
			python_version=sys.version.split()[0],
			platform=platform.platform(),
			max_actions_per_step=getattr(agent.settings, 'max_actions_per_step', None),
			capture_options=self.options.model_dump(mode='json'),
			terminal=self._terminal_record(history),
			counts=self._count_history(history),
			exception={
				'type': type(run_exception).__name__,
				'message': str(run_exception),
			}
			if run_exception
			else None,
			capture_errors=self._capture_errors,
			privacy={
				'tier': 'private_raw',
				'file_mode': '0600',
				'directory_mode': '0700',
				'secret_pattern_matches': secret_scan,
				'network_egress_by_collector': False,
			},
			artifacts=self._artifact_inventory(),
		)
		self._write_json('manifest.json', manifest.model_dump(mode='json', exclude_none=True))
		partial_manifest = self.partial_dir / 'manifest.partial.json'
		if partial_manifest.exists():
			partial_manifest.unlink()
		self._harden_permissions()
		if self.final_dir.exists():
			raise FileExistsError(f'Capture target already exists: {self.final_dir}')
		self.partial_dir.rename(self.final_dir)

	def _write_events(self, history: AgentHistoryList[Any]) -> None:
		lines: list[bytes] = []
		for ordinal, item in enumerate(history.history):
			step = item.metadata.step_number if item.metadata else ordinal
			actions = (
				[action.model_dump(mode='json', exclude_none=True) for action in item.model_output.action]
				if item.model_output
				else []
			)
			results = [result.model_dump(mode='json', exclude_none=True) for result in item.result]
			interacted = [element.to_dict() if element else None for element in (item.state.interacted_element or [])]
			action_events = []
			for action_ordinal in range(max(len(actions), len(results), len(interacted))):
				action = actions[action_ordinal] if action_ordinal < len(actions) else {}
				result = results[action_ordinal] if action_ordinal < len(results) else None
				target = interacted[action_ordinal] if action_ordinal < len(interacted) else None
				action_events.append(
					{
						'ordinal': action_ordinal,
						'name': _action_name(action),
						'action': action,
						'result': result,
						'outcome_status': _result_status(result),
						'interacted_element': target,
					}
				)
			event = {
				'type': 'agent_step',
				'step': step,
				'history_ordinal': ordinal,
				'pre_state': self._state_ref(step, 'pre'),
				'post_state': self._state_ref(step, 'post'),
				'model_output': item.model_output.model_dump(mode='json', exclude_none=True) if item.model_output else None,
				'actions': action_events,
				'metadata': item.metadata.model_dump(mode='json') if item.metadata else None,
				'saved_state': item.state.to_dict(),
				'state_message': item.state_message,
			}
			lines.append(_json_bytes(event, pretty=False) + b'\n')
		self._write_bytes('events.jsonl', b''.join(lines))

	def _derive_run(self, history: AgentHistoryList[Any]) -> dict[str, Any]:
		derived_steps: list[dict[str, Any]] = []
		route: list[str] = []
		warnings: list[str] = []
		if self._agent is not None and getattr(self._agent.settings, 'max_actions_per_step', 1) != 1:
			warnings.append('multi_action_steps_blur_action_level_dom_transitions')

		for ordinal, item in enumerate(history.history):
			step = item.metadata.step_number if item.metadata else ordinal
			pre = self._states.get(step, {}).get('pre')
			post = self._states.get(step, {}).get('post')
			pre_candidates = pre.candidates if pre else []
			post_candidates = post.candidates if post else []
			pre_keys = Counter(_semantic_key(candidate) for candidate in pre_candidates)
			post_keys = Counter(_semantic_key(candidate) for candidate in post_candidates)
			added = list((post_keys - pre_keys).elements())
			removed = list((pre_keys - post_keys).elements())
			control_changes = _control_state_diff(pre, post)

			actions = item.model_output.action if item.model_output else []
			for action_ordinal, action_model in enumerate(actions):
				action = action_model.model_dump(mode='json', exclude_none=True)
				name = _action_name(action)
				route.append(name)
				result_model: ActionResult | None = item.result[action_ordinal] if action_ordinal < len(item.result) else None
				result = result_model.model_dump(mode='json', exclude_none=True) if result_model else None
				selector_index = _action_index(action)
				selected_candidate = next(
					(candidate for candidate in pre_candidates if candidate.get('selector_index') == selector_index), None
				)
				selected_name = selected_candidate.get('ax_name') if selected_candidate else None
				same_name_count = (
					sum(1 for candidate in pre_candidates if selected_name and candidate.get('ax_name') == selected_name)
					if selected_name
					else 0
				)

				facts = {
					'url_changed': bool(pre and post and pre.url != post.url),
					'title_changed': bool(pre and post and pre.title != post.title),
					'dom_changed': bool(pre and post and pre.dom_sha256 != post.dom_sha256),
					'html_changed': bool(pre and post and pre.html_sha256 != post.html_sha256),
					'rendered_text_changed': bool(pre and post and pre.rendered_text_sha256 != post.rendered_text_sha256),
					'expected_success_text_present_after': (post.expected_success_text_present if post is not None else None),
					'screenshot_changed': bool(pre and post and pre.screenshot_sha256 != post.screenshot_sha256),
					'candidate_count_before': len(pre_candidates) if pre else None,
					'candidate_count_after': len(post_candidates) if post else None,
					'candidate_semantics_added': added[:20],
					'candidate_semantics_removed': removed[:20],
					'control_state_changes': control_changes,
				}
				transition_type = self._transition_type(result, facts, len(added), len(removed))
				postcondition_hints = self._postcondition_hints(pre, post, selected_candidate, added, removed, control_changes)
				derived_steps.append(
					{
						'step': step,
						'action_ordinal': action_ordinal,
						'action_name': name,
						'outcome_status': _result_status(result),
						'selected_candidate': selected_candidate,
						'target_identity_quality': _identity_quality(selected_candidate, same_name_count),
						'observed_facts': facts,
						'inference': {
							'transition_type': transition_type,
							'postcondition_hints': postcondition_hints,
							'provenance': 'deterministic_diff_of_captured_pre_and_post_state',
						},
					}
				)

		scores = [step['target_identity_quality']['score'] for step in derived_steps if step['selected_candidate'] is not None]
		return {
			'schema_version': CAPTURE_SCHEMA_VERSION,
			'run_id': self.run_id,
			'task_fingerprint': hashlib.sha256((getattr(self._agent, 'task', '') or '').strip().encode()).hexdigest()[:16],
			'route_signature': route,
			'route_hash': hashlib.sha256(json.dumps(route).encode()).hexdigest()[:16],
			'run_outcome': self._terminal_record(history),
			'offline_verification': self._offline_verification(history),
			'memory_readiness': {
				'target_identity_mean': round(sum(scores) / len(scores), 3) if scores else None,
				'actions_with_resolved_candidate': len(scores),
				'actions_total': len(derived_steps),
				'checkpoint_hints': sum(len(step['inference']['postcondition_hints']) for step in derived_steps),
			},
			'steps': derived_steps,
			'warnings': warnings,
		}

	@staticmethod
	def _transition_type(result: dict[str, Any] | None, facts: dict[str, Any], added: int, removed: int) -> str:
		if result and result.get('error'):
			return 'reported_error'
		if result and result.get('is_done'):
			return 'terminal'
		if facts['url_changed']:
			return 'navigation'
		if facts['control_state_changes'] and added == 0 and removed == 0:
			return 'form_state_mutation'
		if facts['dom_changed'] and added > removed:
			return 'progressive_disclosure'
		if facts['dom_changed'] and removed > added:
			return 'collapse_or_navigation_within_page'
		if facts['dom_changed']:
			return 'dom_mutation'
		if facts['rendered_text_changed'] or facts['html_changed']:
			return 'rendered_page_mutation'
		if facts['screenshot_changed']:
			return 'visual_only_change'
		return 'no_observed_change'

	def _postcondition_hints(
		self,
		pre: _CapturedState | None,
		post: _CapturedState | None,
		selected_candidate: dict[str, Any] | None,
		added: list[str],
		removed: list[str],
		control_changes: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		hints: list[dict[str, Any]] = []
		if pre and post and pre.url != post.url:
			hints.append({'kind': 'url_equals', 'value': post.url, 'evidence': 'observed_url_change'})
		if added:
			hints.append({'kind': 'semantic_elements_appear', 'values': added[:5], 'evidence': 'candidate_set_diff'})
		if selected_candidate:
			selected_key = _semantic_key(selected_candidate)
			if selected_key in removed:
				hints.append({'kind': 'target_disappears', 'value': selected_key, 'evidence': 'candidate_set_diff'})
			attributes = selected_candidate.get('attributes') or {}
			target_identities = {
				f'name:{_normalized_identity(attributes.get("name"))}',
				f'id:{_normalized_identity(attributes.get("id"))}',
				f'ariaLabel:{_normalized_identity(attributes.get("aria-label"))}',
			}
			target_identities.discard('name:')
			target_identities.discard('id:')
			target_identities.discard('ariaLabel:')
			locator = {
				'node_name': selected_candidate.get('node_name'),
				'ax_role': selected_candidate.get('ax_role'),
				'ax_name': selected_candidate.get('ax_name'),
				'attributes': {
					field: attributes[field] for field in ('name', 'type', 'autocomplete', 'aria-label') if attributes.get(field)
				},
			}
			for change in control_changes:
				if change['control'] not in target_identities:
					continue
				for field, values in change['changes'].items():
					if field in {'value', 'checked', 'disabled'}:
						hints.append(
							{
								'kind': f'control_{field}_equals',
								'locator': locator,
								'value': values['after'],
								'evidence': 'browser_native_form_control_state_diff',
							}
						)
		if pre and post and pre.expected_success_text_present is not True and post.expected_success_text_present is True:
			hints.append(
				{
					'kind': 'expected_success_text_appears',
					'value': self.options.expected_success_text,
					'evidence': 'browser_document_body_innerText',
				}
			)
		return hints

	def _offline_verification(self, history: AgentHistoryList[Any]) -> dict[str, Any]:
		if self.options.expected_success_text is None:
			return {
				'enabled': False,
				'provenance': 'no_expected_success_text_configured',
			}
		observations = [
			{
				'step': step,
				'phase': phase,
				'present': state.expected_success_text_present,
			}
			for step, phases in sorted(self._states.items())
			for phase, state in sorted(phases.items(), key=lambda item: 0 if item[0] == 'pre' else 1)
			if state.expected_success_text_present is not None
		]
		observed = any(item['present'] is True for item in observations) if observations else None
		agent_reported_success = history.is_successful()
		return {
			'enabled': True,
			'expected_text': self.options.expected_success_text,
			'observed_in_rendered_page': observed,
			'first_observation': next((item for item in observations if item['present'] is True), None),
			'last_observation': observations[-1] if observations else None,
			'agent_reported_success': agent_reported_success,
			'agreement_with_agent': (
				agent_reported_success == observed if agent_reported_success is not None and observed is not None else None
			),
			'provenance': 'exact_substring_check_against_browser_document_body_innerText',
		}

	def _state_ref(self, step: int, phase: str) -> str | None:
		state = self._states.get(step, {}).get(phase)
		return state.state_path if state else None

	def _record_capture_error(self, phase: str, exc: Exception, step: int | None = None) -> None:
		self._capture_errors.append(
			{
				'at': _utc_now(),
				'phase': phase,
				'step': step,
				'type': type(exc).__name__,
				'message': str(exc),
			}
		)
		logger.warning('Offline capture %s failed%s: %s', phase, f' at step {step}' if step is not None else '', exc)

	def _write_json(self, relative_path: str, value: Any) -> None:
		self._write_bytes(relative_path, _json_bytes(value))

	def _write_gzip_json(self, relative_path: str, value: Any) -> None:
		self._write_bytes(relative_path, gzip.compress(_json_bytes(value, pretty=False), compresslevel=6))

	def _write_bytes(self, relative_path: str, data: bytes) -> None:
		path = self.partial_dir / relative_path
		path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
		temporary_path = path.with_name(f'.{path.name}.{uuid7str()}.tmp')
		with temporary_path.open('wb') as file:
			file.write(data)
			file.flush()
			os.fsync(file.fileno())
		os.chmod(temporary_path, 0o600)
		temporary_path.replace(path)

	def _artifact_inventory(self) -> list[ArtifactRecord]:
		artifacts = []
		for path in sorted(self.partial_dir.rglob('*')):
			if not path.is_file() or path.name in {'manifest.json', 'manifest.partial.json'}:
				continue
			data = path.read_bytes()
			artifacts.append(
				ArtifactRecord(
					path=str(path.relative_to(self.partial_dir)),
					bytes=len(data),
					sha256=_sha256(data),
				)
			)
		return artifacts

	def _harden_permissions(self) -> None:
		"""Normalize permissions for artifacts written by Browser Use itself."""
		for path in self.partial_dir.rglob('*'):
			os.chmod(path, 0o700 if path.is_dir() else 0o600)
		os.chmod(self.partial_dir, 0o700)

	def _secret_scan(self) -> dict[str, Any]:
		matches_by_file: dict[str, int] = {}
		for path in self.partial_dir.rglob('*'):
			if not path.is_file() or path.suffix in {'.png', '.mp4', '.gz'}:
				continue
			try:
				text = path.read_text(encoding='utf-8', errors='ignore')
			except OSError:
				continue
			count = len(SECRET_PATTERN.findall(text))
			if count:
				matches_by_file[str(path.relative_to(self.partial_dir))] = count
		return {'count': sum(matches_by_file.values()), 'files': matches_by_file}

	@staticmethod
	def _browser_use_version() -> str | None:
		try:
			return version('browser-use')
		except PackageNotFoundError:
			return None

	@staticmethod
	def _terminal_record(history: AgentHistoryList[Any] | None) -> dict[str, Any]:
		if history is None:
			return {'is_done': False, 'is_successful': None, 'has_errors': None}
		return {
			'is_done': history.is_done(),
			'is_successful': history.is_successful(),
			'has_errors': history.has_errors(),
			'final_result': history.final_result(),
			'judgement': history.judgement(),
		}

	@staticmethod
	def _count_history(history: AgentHistoryList[Any] | None) -> dict[str, int]:
		if history is None:
			return {'history_steps': 0, 'actions': 0, 'results': 0, 'errors': 0}
		results = history.action_results()
		return {
			'history_steps': len(history.history),
			'actions': len(history.model_actions()),
			'results': len(results),
			'errors': sum(1 for result in results if result.error),
		}


def analyze_capture_corpus(output_dir: str | Path) -> dict[str, Any]:
	"""Infer repeated routes and target feature stability across completed bundles."""
	root = Path(output_dir).expanduser().resolve()
	runs: list[dict[str, Any]] = []
	for derived_path in sorted(root.glob('*/derived.json')):
		manifest_path = derived_path.parent / 'manifest.json'
		if not manifest_path.exists():
			continue
		try:
			derived = json.loads(derived_path.read_text())
			manifest = json.loads(manifest_path.read_text())
		except (OSError, json.JSONDecodeError):
			continue
		runs.append({'directory': derived_path.parent.name, 'derived': derived, 'manifest': manifest})

	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for run in runs:
		groups[run['derived'].get('task_fingerprint', 'unknown')].append(run)

	task_groups = []
	for task_fingerprint, task_runs in sorted(groups.items()):
		route_counts = Counter(tuple(run['derived'].get('route_signature', [])) for run in task_runs)
		agent_successful_runs = [run for run in task_runs if run['manifest'].get('terminal', {}).get('is_successful') is True]
		verifiable_runs = [run for run in task_runs if run['derived'].get('offline_verification', {}).get('enabled') is True]
		evidence_verified_runs = [
			run
			for run in agent_successful_runs
			if run['derived'].get('offline_verification', {}).get('observed_in_rendered_page') is True
		]
		outcome_disagreements = [
			run['directory']
			for run in verifiable_runs
			if run['derived'].get('offline_verification', {}).get('agreement_with_agent') is False
		]
		successful_runs = evidence_verified_runs if verifiable_runs else agent_successful_runs
		positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for run in successful_runs:
			alignment_counts: Counter[str] = Counter()
			for step in run['derived'].get('steps', []):
				if step.get('selected_candidate'):
					base = _target_alignment_base(step)
					alignment_counts[base] += 1
					alignment_key = f'{base}|occurrence:{alignment_counts[base]}'
					positions[alignment_key].append(
						{
							'candidate': step['selected_candidate'],
							'run_directory': run['directory'],
							'step': step['step'],
						}
					)

		stable_targets = []
		for alignment_key, observations in sorted(positions.items()):
			if len(observations) < 2:
				continue
			candidates = [observation['candidate'] for observation in observations]
			stable_fields = _stable_candidate_fields(candidates)
			locator_uniqueness = _locator_uniqueness(root, observations, stable_fields)
			stable_targets.append(
				{
					'alignment_key': alignment_key,
					'observations': len(candidates),
					'stable_fields': stable_fields,
					'recommended_locator_fields': [field['field'] for field in stable_fields if field['recommended']],
					'volatile_fields': [field['field'] for field in stable_fields if field['stability'] < 0.8],
					'locator_uniqueness': locator_uniqueness,
					'geometry': _geometry_summary(candidates),
				}
			)

		dominant_route_count = route_counts.most_common(1)[0][1] if route_counts else 0
		task_groups.append(
			{
				'task_fingerprint': task_fingerprint,
				'runs': len(task_runs),
				'successful_runs': len(successful_runs),
				'agent_reported_successful_runs': len(agent_successful_runs),
				'evidence_verified_successful_runs': len(evidence_verified_runs),
				'outcome_disagreements': outcome_disagreements,
				'route_variants': [{'actions': list(route), 'runs': count} for route, count in route_counts.most_common()],
				'route_consensus': round(dominant_route_count / len(task_runs), 3) if task_runs else None,
				'stable_targets': stable_targets,
				'inference': _memory_readiness_inference(
					task_runs, successful_runs, route_counts, stable_targets, outcome_disagreements
				),
			}
		)

	return {
		'schema_version': CAPTURE_SCHEMA_VERSION,
		'analyzed_at': _utc_now(),
		'root': str(root),
		'run_count': len(runs),
		'task_groups': task_groups,
	}


def _stable_candidate_fields(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
	field_names = (
		'node_name',
		'ax_role',
		'ax_name',
		'attributes.autocomplete',
		'attributes.name',
		'attributes.type',
		'attributes.aria-label',
		'attributes.placeholder',
		'attributes.id',
		'x_path',
		'stable_hash',
	)
	stable = []
	for field_name in field_names:
		values = [_candidate_field_value(candidate, field_name) for candidate in candidates]
		populated = [value for value in values if value not in (None, '')]
		if not populated:
			continue
		value_counts = Counter(json.dumps(value, sort_keys=True, default=_json_default) for value in populated)
		winning_value, winning_count = value_counts.most_common(1)[0]
		stability = winning_count / len(candidates)
		stable.append(
			{
				'field': field_name,
				'value': json.loads(winning_value),
				'coverage': round(len(populated) / len(candidates), 3),
				'stability': round(stability, 3),
				'recommended': stability >= 0.8 and field_name not in {'x_path', 'attributes.id', 'stable_hash'},
			}
		)
	return sorted(stable, key=lambda item: (-item['stability'], item['field']))


def _candidate_field_value(candidate: dict[str, Any], field_name: str) -> Any:
	if field_name.startswith('attributes.'):
		return (candidate.get('attributes') or {}).get(field_name.removeprefix('attributes.'))
	return candidate.get(field_name)


def _locator_uniqueness(root: Path, observations: list[dict[str, Any]], stable_fields: list[dict[str, Any]]) -> dict[str, Any]:
	locator = {field['field']: field['value'] for field in stable_fields if field['recommended']}
	checks: list[dict[str, Any]] = []
	for observation in observations:
		candidate_path = root / observation['run_directory'] / 'steps' / f'{int(observation["step"]):03d}' / 'pre.candidates.json'
		try:
			candidates = json.loads(candidate_path.read_text())
		except (OSError, json.JSONDecodeError):
			continue
		match_count = sum(
			1
			for candidate in candidates
			if all(_candidate_field_value(candidate, field) == value for field, value in locator.items())
		)
		checks.append(
			{
				'run_directory': observation['run_directory'],
				'step': observation['step'],
				'candidate_count': len(candidates),
				'locator_match_count': match_count,
			}
		)
	unique = all(check['locator_match_count'] == 1 for check in checks) if checks and locator else None
	return {
		'locator': locator,
		'evaluated_states': len(checks),
		'unique_in_all_evaluated_states': unique,
		'checks': checks,
		'provenance': 'exact_match_against_each_captured_pre_action_candidate_set',
	}


def _geometry_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
	boxes: list[dict[str, Any]] = []
	for candidate in candidates:
		box = candidate.get('bounds')
		if isinstance(box, dict):
			boxes.append(box)
	if not boxes:
		return {'observations': 0, 'layout_shift_detected': None}
	x_values = [float(box['x']) for box in boxes]
	y_values = [float(box['y']) for box in boxes]
	width_values = [float(box['width']) for box in boxes]
	height_values = [float(box['height']) for box in boxes]
	max_x_shift = max(x_values) - min(x_values)
	max_y_shift = max(y_values) - min(y_values)
	max_width_delta = max(width_values) - min(width_values)
	max_height_delta = max(height_values) - min(height_values)
	return {
		'observations': len(boxes),
		'boxes': boxes,
		'max_x_shift_pixels': round(max_x_shift, 3),
		'max_y_shift_pixels': round(max_y_shift, 3),
		'max_width_delta_pixels': round(max_width_delta, 3),
		'max_height_delta_pixels': round(max_height_delta, 3),
		'layout_shift_detected': max(max_x_shift, max_y_shift, max_width_delta, max_height_delta) > 4,
	}


def _memory_readiness_inference(
	task_runs: list[dict[str, Any]],
	successful_runs: list[dict[str, Any]],
	route_counts: Counter[tuple[str, ...]],
	stable_targets: list[dict[str, Any]],
	outcome_disagreements: list[str],
) -> dict[str, Any]:
	reasons: list[str] = []
	risks: list[str] = []
	score = 0.0
	if successful_runs:
		score += min(0.25, 0.125 * len(successful_runs))
		reasons.append(f'{len(successful_runs)} successful captured run(s)')
	else:
		risks.append('no_successful_runs')
	if len(successful_runs) >= 2:
		reasons.append('cross_run_target_stability_is_measurable')
		score += 0.2
	if len(successful_runs) < 3:
		risks.append('small_sample_size:fewer_than_3_evidence_backed_successes')
	if route_counts:
		consensus = route_counts.most_common(1)[0][1] / len(task_runs)
		score += 0.25 * consensus
		reasons.append(f'route_consensus={consensus:.3f}')
		if consensus < 0.75:
			risks.append('route_divergence_requires_branching_or_optional_steps')
	if stable_targets:
		score += min(0.25, 0.05 * len(stable_targets))
		reasons.append(f'{len(stable_targets)} cross-run semantic target slot(s)')
	else:
		risks.append('insufficient_repeated_target_observations')
	ambiguous_locators = sum(
		1 for target in stable_targets if target.get('locator_uniqueness', {}).get('unique_in_all_evaluated_states') is False
	)
	if ambiguous_locators:
		risks.append(f'ambiguous_stable_target_locators:{ambiguous_locators}')
		score -= min(0.25, 0.1 * ambiguous_locators)
	if any(run['manifest'].get('capture_errors') for run in task_runs):
		risks.append('one_or_more_runs_have_capture_errors')
		score -= 0.1
	if outcome_disagreements:
		risks.append(f'agent_page_outcome_disagreement:{len(outcome_disagreements)}')
		score -= 0.2
	if len(successful_runs) < 3:
		score = min(score, 0.79)
	if len(successful_runs) >= 3 and len(stable_targets) >= 2 and not outcome_disagreements and not ambiguous_locators:
		label = 'ready_for_replay_prototype'
	elif len(successful_runs) >= 2 and stable_targets and not outcome_disagreements:
		label = 'ready_for_adaptive_replay_experiment'
	else:
		label = 'needs_more_evidence'
	return {
		'score': round(max(0.0, min(1.0, score)), 3),
		'label': label,
		'reasons': reasons,
		'risks': risks,
		'provenance': 'deterministic_aggregation_of_local_run_bundles',
	}


def write_corpus_report(output_dir: str | Path, destination: str | Path | None = None) -> Path:
	"""Analyze an output directory and write an integrity-protected JSON report."""
	root = Path(output_dir).expanduser().resolve()
	report_path = Path(destination).expanduser().resolve() if destination else root / 'corpus_report.json'
	report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
	report = analyze_capture_corpus(root)
	temporary_path = report_path.with_name(f'.{report_path.name}.{uuid7str()}.tmp')
	temporary_path.write_bytes(_json_bytes(report))
	os.chmod(temporary_path, 0o600)
	temporary_path.replace(report_path)
	return report_path


def remove_partial_capture(output_dir: str | Path, run_id: str) -> bool:
	"""Remove one explicitly named unfinished bundle; completed runs are never touched."""
	root = Path(output_dir).expanduser().resolve()
	partial = root / f'.{run_id}.partial'
	if partial.parent != root or not partial.name.endswith('.partial') or not partial.exists():
		return False
	shutil.rmtree(partial)
	return True
