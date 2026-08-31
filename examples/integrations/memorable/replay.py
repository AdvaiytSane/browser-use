"""Guarded semantic replay against fresh Browser Use DOM state, without an LLM."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

from browser_use import BrowserProfile, BrowserSession, Tools
from browser_use.browser.profile import ViewportSize
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode
from examples.integrations.memorable.procedure import (
	BrowserProcedure,
	PostconditionKind,
	ProcedureStep,
	ReplayAction,
	SemanticLocator,
	ValueBinding,
	locator_matches,
)

REPLAY_KILL_SWITCH = 'BROWSER_USE_MEMORABLE_SEMANTIC_REPLAY'


class ResolutionStatus(str, Enum):
	RESOLVED = 'resolved'
	ABSENT = 'absent'
	AMBIGUOUS = 'ambiguous'
	NOT_ACTIONABLE = 'not_actionable'


class ReplayStatus(str, Enum):
	COMPLETED = 'completed'
	NEEDS_RECOVERY = 'needs_recovery'
	ACTION_FAILED = 'action_failed'
	DISABLED = 'disabled'


class StepStatus(str, Enum):
	EXECUTED = 'executed'
	SKIPPED_OPTIONAL = 'skipped_optional'
	ALREADY_SATISFIED = 'already_satisfied'
	NEEDS_RECOVERY = 'needs_recovery'
	ACTION_FAILED = 'action_failed'


class CandidateSnapshot(BaseModel):
	selector_index: int
	node_name: str | None = None
	ax_role: str | None = None
	ax_name: str | None = None
	attributes: dict[str, str] = Field(default_factory=dict)
	is_visible: bool | None = None
	bounds: dict[str, float] | None = None

	def as_match_record(self) -> dict[str, Any]:
		return {
			'node_name': self.node_name,
			'ax_role': self.ax_role,
			'ax_name': self.ax_name,
			'attributes': self.attributes,
		}


class Resolution(BaseModel):
	status: ResolutionStatus
	selector_index: int | None = None
	match_count: int = 0
	reason: str
	matches: list[CandidateSnapshot] = Field(default_factory=list)


class Verification(BaseModel):
	verified: bool
	kind: str
	evidence: dict[str, Any] = Field(default_factory=dict)


class ReplayEvent(BaseModel):
	ordinal: int
	step_id: str
	action: str
	status: StepStatus
	selector_index: int | None = None
	current_bounds: dict[str, float] | None = None
	locator: dict[str, str]
	resolution_status: str
	resolution_match_count: int
	parameter_name: str | None = None
	action_result: dict[str, Any] | None = None
	postcondition: Verification | None = None
	duration_seconds: float
	reason: str | None = None


class ReplayReport(BaseModel):
	run_id: str
	procedure_id: str
	status: ReplayStatus
	started_at: str
	finished_at: str
	initial_url: str
	final_url: str | None = None
	model_calls: int = 0
	actions_attempted: int = 0
	events: list[ReplayEvent] = Field(default_factory=list)
	reason: str | None = None
	audit_dir: str | None = None


class ReplayOptions(BaseModel):
	output_dir: Path
	action_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
	poll_interval_seconds: float = Field(default=0.1, gt=0, le=2)
	optional_wait_seconds: float = Field(default=0.5, ge=0, le=5)
	max_actions: int = Field(default=30, ge=1, le=1000)


class SemanticResolver:
	"""Resolve exact semantic conjunctions and refuse ties."""

	def resolve(
		self,
		locator: SemanticLocator,
		state: BrowserStateSummary,
		action: ReplayAction,
	) -> Resolution:
		matches = [
			_candidate_snapshot(index, node)
			for index, node in sorted(state.dom_state.selector_map.items())
			if locator_matches(locator.required, _candidate_snapshot(index, node).as_match_record())
		]
		if not matches:
			return Resolution(
				status=ResolutionStatus.ABSENT,
				reason='no current DOM candidate matches every stable semantic field',
			)
		if len(matches) > 1:
			return Resolution(
				status=ResolutionStatus.AMBIGUOUS,
				match_count=len(matches),
				reason='multiple current DOM candidates satisfy the exact locator; ranking is forbidden',
				matches=matches,
			)
		candidate = matches[0]
		unsafe_reason = _not_actionable_reason(candidate, action)
		if unsafe_reason:
			return Resolution(
				status=ResolutionStatus.NOT_ACTIONABLE,
				selector_index=candidate.selector_index,
				match_count=1,
				reason=unsafe_reason,
				matches=matches,
			)
		return Resolution(
			status=ResolutionStatus.RESOLVED,
			selector_index=candidate.selector_index,
			match_count=1,
			reason='exact semantic locator resolved uniquely and passed actionability gates',
			matches=matches,
		)


class DeterministicReplayer:
	"""Execute a compiled procedure with Browser Use actions and live verification."""

	def __init__(
		self,
		procedure: BrowserProcedure,
		options: ReplayOptions,
		*,
		tools: Tools[Any] | None = None,
	):
		self.procedure = procedure
		self.options = options
		self.tools = tools or Tools()
		self.resolver = SemanticResolver()
		self.action_model = self.tools.registry.create_action_model(include_actions=[action.value for action in ReplayAction])

	async def run(self, browser_session: BrowserSession, parameters: dict[str, str]) -> ReplayReport:
		"""Replay against an already-navigated browser session."""

		if os.getenv(REPLAY_KILL_SWITCH, '1').casefold() in {'0', 'false', 'off'}:
			now = _utc_now()
			return ReplayReport(
				run_id=uuid7str(),
				procedure_id=self.procedure.procedure_id,
				status=ReplayStatus.DISABLED,
				started_at=now,
				finished_at=now,
				initial_url='',
				reason=f'{REPLAY_KILL_SWITCH} is disabled; no browser state was read and no audit files were written',
			)

		run_id = uuid7str()
		started_at = _utc_now()
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		initial_url = state.url
		events: list[ReplayEvent] = []
		status = ReplayStatus.COMPLETED
		reason: str | None = None
		actions_attempted = 0
		parameter_error = self._parameter_error(parameters)
		if parameter_error:
			status = ReplayStatus.NEEDS_RECOVERY
			reason = parameter_error
		elif not self.procedure.site_scope.permits(initial_url):
			status = ReplayStatus.NEEDS_RECOVERY
			reason = f'initial URL is outside compiled site scope: {initial_url}'

		if status == ReplayStatus.COMPLETED:
			for ordinal, step in enumerate(self.procedure.steps):
				if actions_attempted >= self.options.max_actions:
					status = ReplayStatus.NEEDS_RECOVERY
					reason = f'maximum deterministic action count reached ({self.options.max_actions})'
					break
				event_started = time.monotonic()
				state = await browser_session.get_browser_state_summary(include_screenshot=False)
				if not self.procedure.site_scope.permits(state.url):
					status = ReplayStatus.NEEDS_RECOVERY
					reason = f'current URL left compiled site scope before {step.id}: {state.url}'
					events.append(self._refusal_event(ordinal, step, event_started, reason, ResolutionStatus.NOT_ACTIONABLE))
					break

				resolution = self.resolver.resolve(step.locator, state, step.action)
				if resolution.status == ResolutionStatus.ABSENT and step.optional:
					resolution, state = await self._wait_for_optional(browser_session, step)
					if resolution.status == ResolutionStatus.ABSENT and self._downstream_is_available(state, ordinal):
						events.append(
							ReplayEvent(
								ordinal=ordinal,
								step_id=step.id,
								action=step.action.value,
								status=StepStatus.SKIPPED_OPTIONAL,
								locator=step.locator.required,
								resolution_status=resolution.status.value,
								resolution_match_count=0,
								duration_seconds=round(time.monotonic() - event_started, 6),
								reason='optional target is absent while a later required state is uniquely available',
							)
						)
						continue

				if resolution.status != ResolutionStatus.RESOLVED or resolution.selector_index is None:
					status = ReplayStatus.NEEDS_RECOVERY
					reason = f'{step.id}: {resolution.status.value}: {resolution.reason}'
					events.append(self._resolution_failure_event(ordinal, step, event_started, resolution, reason))
					break

				preverification = await self._check_postcondition(browser_session, step, parameters)
				if preverification.verified and step.postcondition.kind in {
					PostconditionKind.CONTROL_VALUE_EQUALS,
					PostconditionKind.CONTROL_CHECKED_EQUALS,
				}:
					events.append(
						ReplayEvent(
							ordinal=ordinal,
							step_id=step.id,
							action=step.action.value,
							status=StepStatus.ALREADY_SATISFIED,
							selector_index=resolution.selector_index,
							current_bounds=resolution.matches[0].bounds,
							locator=step.locator.required,
							resolution_status=resolution.status.value,
							resolution_match_count=resolution.match_count,
							parameter_name=_parameter_name(step.value),
							postcondition=preverification,
							duration_seconds=round(time.monotonic() - event_started, 6),
							reason='live control state already equals the requested value',
						)
					)
					continue

				payload = self._action_payload(step, resolution.selector_index, parameters)
				action = self.action_model.model_validate(payload)
				actions_attempted += 1
				result = await self.tools.act(
					action,
					browser_session,
					action_timeout=self.options.action_timeout_seconds,
				)
				result_summary = _action_result_summary(result)
				if result.error:
					status = ReplayStatus.ACTION_FAILED
					reason = f'{step.id}: Browser Use action reported an error: {result.error}'
					events.append(
						ReplayEvent(
							ordinal=ordinal,
							step_id=step.id,
							action=step.action.value,
							status=StepStatus.ACTION_FAILED,
							selector_index=resolution.selector_index,
							current_bounds=resolution.matches[0].bounds,
							locator=step.locator.required,
							resolution_status=resolution.status.value,
							resolution_match_count=resolution.match_count,
							parameter_name=_parameter_name(step.value),
							action_result=result_summary,
							duration_seconds=round(time.monotonic() - event_started, 6),
							reason=reason,
						)
					)
					break

				verification = await self._wait_for_postcondition(browser_session, step, parameters)
				if not verification.verified:
					status = ReplayStatus.NEEDS_RECOVERY
					reason = f'{step.id}: Browser Use returned no error, but the live postcondition was not verified'
					step_status = StepStatus.NEEDS_RECOVERY
				else:
					step_status = StepStatus.EXECUTED
				events.append(
					ReplayEvent(
						ordinal=ordinal,
						step_id=step.id,
						action=step.action.value,
						status=step_status,
						selector_index=resolution.selector_index,
						current_bounds=resolution.matches[0].bounds,
						locator=step.locator.required,
						resolution_status=resolution.status.value,
						resolution_match_count=resolution.match_count,
						parameter_name=_parameter_name(step.value),
						action_result=result_summary,
						postcondition=verification,
						duration_seconds=round(time.monotonic() - event_started, 6),
						reason=reason if not verification.verified else None,
					)
				)
				if not verification.verified:
					break

		try:
			final_state = await browser_session.get_browser_state_summary(include_screenshot=False)
			final_url = final_state.url
		except Exception:
			final_url = None
		report = ReplayReport(
			run_id=run_id,
			procedure_id=self.procedure.procedure_id,
			status=status,
			started_at=started_at,
			finished_at=_utc_now(),
			initial_url=initial_url,
			final_url=final_url,
			model_calls=0,
			actions_attempted=actions_attempted,
			events=events,
			reason=reason,
		)
		report.audit_dir = str(_write_audit_bundle(self.options.output_dir, report, self.procedure))
		return report

	def _parameter_error(self, parameters: dict[str, str]) -> str | None:
		expected = {parameter.name for parameter in self.procedure.parameters if parameter.required}
		missing = sorted(expected - parameters.keys())
		extra = sorted(parameters.keys() - {parameter.name for parameter in self.procedure.parameters})
		empty = sorted(name for name in expected if not parameters.get(name))
		if missing:
			return f'missing required runtime parameters: {missing}'
		if empty:
			return f'required runtime parameters are empty: {empty}'
		if extra:
			return f'unknown runtime parameters: {extra}'
		return None

	async def _wait_for_optional(
		self, browser_session: BrowserSession, step: ProcedureStep
	) -> tuple[Resolution, BrowserStateSummary]:
		deadline = time.monotonic() + self.options.optional_wait_seconds
		while True:
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			resolution = self.resolver.resolve(step.locator, state, step.action)
			if resolution.status != ResolutionStatus.ABSENT or time.monotonic() >= deadline:
				return resolution, state
			await asyncio.sleep(self.options.poll_interval_seconds)

	def _downstream_is_available(self, state: BrowserStateSummary, ordinal: int) -> bool:
		for downstream in self.procedure.steps[ordinal + 1 :]:
			if downstream.optional:
				continue
			resolution = self.resolver.resolve(downstream.locator, state, downstream.action)
			return resolution.status == ResolutionStatus.RESOLVED
		return False

	def _action_payload(self, step: ProcedureStep, selector_index: int, parameters: dict[str, str]) -> dict[str, Any]:
		if step.action == ReplayAction.INPUT:
			return {
				'input': {
					'index': selector_index,
					'text': _binding_value(step.value, parameters),
					'clear': step.clear,
				}
			}
		if step.action == ReplayAction.SELECT_DROPDOWN:
			return {
				'select_dropdown': {
					'index': selector_index,
					'text': _binding_value(step.value, parameters),
				}
			}
		return {'click': {'index': selector_index}}

	async def _wait_for_postcondition(
		self,
		browser_session: BrowserSession,
		step: ProcedureStep,
		parameters: dict[str, str],
	) -> Verification:
		deadline = time.monotonic() + step.postcondition.timeout_seconds
		last = Verification(verified=False, kind=step.postcondition.kind.value)
		while True:
			last = await self._check_postcondition(browser_session, step, parameters)
			if last.verified or time.monotonic() >= deadline:
				return last
			await asyncio.sleep(self.options.poll_interval_seconds)

	async def _check_postcondition(
		self,
		browser_session: BrowserSession,
		step: ProcedureStep,
		parameters: dict[str, str],
	) -> Verification:
		postcondition = step.postcondition
		if postcondition.kind == PostconditionKind.TARGET_DISAPPEARS:
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			resolution = self.resolver.resolve(step.locator, state, step.action)
			return Verification(
				verified=resolution.status == ResolutionStatus.ABSENT,
				kind=postcondition.kind.value,
				evidence={
					'current_match_status': resolution.status.value,
					'current_match_count': resolution.match_count,
				},
			)
		page = await _read_live_page(browser_session)
		if page is None:
			return Verification(
				verified=False,
				kind=postcondition.kind.value,
				evidence={'browser_native_page_state': 'unavailable'},
			)
		if postcondition.kind == PostconditionKind.EXPECTED_TEXT_APPEARS:
			text = postcondition.text or ''
			return Verification(
				verified=text in page['rendered_text'],
				kind=postcondition.kind.value,
				evidence={
					'exact_text_present': text in page['rendered_text'],
					'expected_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
					'provenance': 'document.body.innerText exact substring check',
				},
			)
		controls = [control for control in page['controls'] if _control_matches(step.locator.required, control)]
		if len(controls) != 1:
			return Verification(
				verified=False,
				kind=postcondition.kind.value,
				evidence={
					'live_control_match_count': len(controls),
					'provenance': 'browser-native form control state',
				},
			)
		expected = _binding_value(postcondition.value, parameters)
		control = controls[0]
		if postcondition.kind == PostconditionKind.CONTROL_CHECKED_EQUALS:
			observed = bool(control.get('checked'))
			verified = observed is bool(expected)
			evidence = {
				'live_control_match_count': 1,
				'observed_checked': observed,
				'expected_checked': bool(expected),
				'provenance': 'browser-native checked property',
			}
		else:
			observed_value = str(control.get('value') or '')
			expected_value = str(expected)
			verified = observed_value == expected_value
			evidence = {
				'live_control_match_count': 1,
				'observed_value_sha256': hashlib.sha256(observed_value.encode()).hexdigest(),
				'expected_value_sha256': hashlib.sha256(expected_value.encode()).hexdigest(),
				'observed_value_length': len(observed_value),
				'expected_value_length': len(expected_value),
				'provenance': 'browser-native value property',
			}
		return Verification(verified=verified, kind=postcondition.kind.value, evidence=evidence)

	@staticmethod
	def _resolution_failure_event(
		ordinal: int,
		step: ProcedureStep,
		started: float,
		resolution: Resolution,
		reason: str,
	) -> ReplayEvent:
		return ReplayEvent(
			ordinal=ordinal,
			step_id=step.id,
			action=step.action.value,
			status=StepStatus.NEEDS_RECOVERY,
			selector_index=resolution.selector_index,
			current_bounds=resolution.matches[0].bounds if len(resolution.matches) == 1 else None,
			locator=step.locator.required,
			resolution_status=resolution.status.value,
			resolution_match_count=resolution.match_count,
			parameter_name=_parameter_name(step.value),
			duration_seconds=round(time.monotonic() - started, 6),
			reason=reason,
		)

	@staticmethod
	def _refusal_event(
		ordinal: int,
		step: ProcedureStep,
		started: float,
		reason: str,
		resolution_status: ResolutionStatus,
	) -> ReplayEvent:
		return ReplayEvent(
			ordinal=ordinal,
			step_id=step.id,
			action=step.action.value,
			status=StepStatus.NEEDS_RECOVERY,
			locator=step.locator.required,
			resolution_status=resolution_status.value,
			resolution_match_count=0,
			parameter_name=_parameter_name(step.value),
			duration_seconds=round(time.monotonic() - started, 6),
			reason=reason,
		)


def _candidate_snapshot(index: int, node: EnhancedDOMTreeNode) -> CandidateSnapshot:
	ax_node = node.ax_node
	bounds: dict[str, float] | None = None
	if node.absolute_position is not None:
		raw_bounds = node.absolute_position.to_dict()
		bounds = {
			key: float(raw_bounds[key]) for key in ('x', 'y', 'width', 'height') if isinstance(raw_bounds.get(key), int | float)
		}
	return CandidateSnapshot(
		selector_index=index,
		node_name=node.node_name,
		ax_role=ax_node.role if ax_node else None,
		ax_name=ax_node.name if ax_node else None,
		attributes={str(key): str(value) for key, value in (node.attributes or {}).items()},
		is_visible=node.is_visible,
		bounds=bounds,
	)


def _not_actionable_reason(candidate: CandidateSnapshot, action: ReplayAction) -> str | None:
	if candidate.is_visible is False:
		return 'unique candidate is not visible'
	if not candidate.bounds:
		return 'unique candidate has no rendered bounds'
	if candidate.bounds.get('width', 0) <= 0 or candidate.bounds.get('height', 0) <= 0:
		return 'unique candidate has an empty rendered box'
	attributes = candidate.attributes
	if 'disabled' in attributes or attributes.get('aria-disabled', '').casefold() == 'true':
		return 'unique candidate is disabled'
	node_name = (candidate.node_name or '').casefold()
	input_type = attributes.get('type', '').casefold()
	if action == ReplayAction.INPUT and (node_name not in {'input', 'textarea'} or input_type == 'checkbox'):
		return f'unique target is incompatible with input: {node_name or "unknown"}/{input_type or "default"}'
	if action == ReplayAction.SELECT_DROPDOWN and node_name != 'select':
		return f'unique target is incompatible with select_dropdown: {node_name or "unknown"}'
	return None


def _control_matches(required: dict[str, str], control: dict[str, Any]) -> bool:
	dom_fields = {field: value for field, value in required.items() if field == 'node_name' or field.startswith('attributes.')}
	if len(dom_fields) <= ('node_name' in dom_fields):
		return False
	return locator_matches(dom_fields, control)


async def _read_live_page(browser_session: BrowserSession) -> dict[str, Any] | None:
	try:
		page = await browser_session.must_get_current_page()
		payload = await page.evaluate(
			"""() => JSON.stringify({
				url: location.href,
				rendered_text: document.body ? document.body.innerText : '',
				controls: Array.from(document.querySelectorAll(
					'input, select, textarea, button, [contenteditable=\"true\"]'
				)).map(element => ({
					node_name: element.tagName,
					attributes: {
						name: element.getAttribute('name'),
						type: element.getAttribute('type'),
						autocomplete: element.getAttribute('autocomplete'),
						'aria-label': element.getAttribute('aria-label'),
						placeholder: element.getAttribute('placeholder'),
						'data-action': element.getAttribute('data-action')
					},
					value: 'value' in element ? element.value : null,
					checked: 'checked' in element ? element.checked : null,
					disabled: 'disabled' in element ? element.disabled : null,
					hidden: element.hidden
				}))
			})"""
		)
		return json.loads(payload)
	except Exception:
		return None


def _binding_value(binding: ValueBinding | None, parameters: dict[str, str]) -> str | bool:
	if binding is None:
		raise ValueError('Missing value binding')
	if binding.parameter is not None:
		return parameters[binding.parameter]
	if isinstance(binding.literal, str | bool):
		return binding.literal
	raise ValueError('Value binding did not contain a supported value')


def _parameter_name(binding: ValueBinding | None) -> str | None:
	return binding.parameter if binding else None


def _action_result_summary(result: Any) -> dict[str, Any]:
	content = str(result.extracted_content or '')
	return {
		'reported_error': result.error is not None,
		'error': result.error,
		'extracted_content_sha256': hashlib.sha256(content.encode()).hexdigest(),
		'extracted_content_length': len(content),
		'provenance': 'Browser Use Tools.act ActionResult',
	}


def _write_audit_bundle(
	output_root: Path,
	report: ReplayReport,
	procedure: BrowserProcedure,
) -> Path:
	root = output_root.expanduser().resolve()
	root.mkdir(parents=True, exist_ok=True, mode=0o700)
	os.chmod(root, 0o700)
	partial = root / f'.{report.run_id}.partial'
	final = root / report.run_id
	partial.mkdir(mode=0o700)
	report.audit_dir = str(final)
	files = {
		'report.json': report.model_dump_json(indent=2) + '\n',
		'events.jsonl': ''.join(event.model_dump_json() + '\n' for event in report.events),
		'procedure.snapshot.json': procedure.model_dump_json(indent=2) + '\n',
	}
	for name, content in files.items():
		path = partial / name
		path.write_text(content)
		os.chmod(path, 0o600)
	manifest = {
		'schema_version': '0.1.0',
		'run_id': report.run_id,
		'status': report.status.value,
		'model_calls': 0,
		'privacy': {
			'tier': 'private_local_audit',
			'runtime_parameter_values_retained': False,
			'action_result_content_retained': False,
		},
		'artifacts': [
			{
				'path': name,
				'bytes': len(content.encode()),
				'sha256': hashlib.sha256(content.encode()).hexdigest(),
			}
			for name, content in files.items()
		],
	}
	manifest_path = partial / 'manifest.json'
	manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
	os.chmod(manifest_path, 0o600)
	partial.replace(final)
	return final


def _parse_parameters(items: list[str]) -> dict[str, str]:
	parameters: dict[str, str] = {}
	for item in items:
		if '=' not in item:
			raise ValueError(f'Runtime parameter must use KEY=VALUE syntax: {item!r}')
		key, value = item.split('=', 1)
		if not key:
			raise ValueError(f'Runtime parameter name cannot be empty: {item!r}')
		if key in parameters:
			raise ValueError(f'Duplicate runtime parameter: {key}')
		parameters[key] = value
	return parameters


async def _main_async(args: argparse.Namespace) -> int:
	procedure = BrowserProcedure.read(args.procedure)
	parameters = _parse_parameters(args.parameter)
	profile = BrowserProfile(
		headless=args.headless,
		user_data_dir=None,
		window_size=ViewportSize(width=args.viewport_width, height=args.viewport_height),
	)
	browser_session = BrowserSession(browser_profile=profile)
	await browser_session.start()
	try:
		await browser_session.navigate_to(args.url)
		replayer = DeterministicReplayer(
			procedure,
			ReplayOptions(output_dir=args.output_dir),
		)
		report = await replayer.run(browser_session, parameters)
		print(report.model_dump_json(indent=2))
		return 0 if report.status == ReplayStatus.COMPLETED else 2
	finally:
		await browser_session.kill()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('procedure', type=Path)
	parser.add_argument('url')
	parser.add_argument('-p', '--parameter', action='append', default=[], metavar='KEY=VALUE')
	parser.add_argument('--output-dir', type=Path, default=Path('./tmp/memorable-replays'))
	parser.add_argument('--viewport-width', type=int, default=1280)
	parser.add_argument('--viewport-height', type=int, default=800)
	parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
	args = parser.parse_args()
	raise SystemExit(asyncio.run(_main_async(args)))


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


if __name__ == '__main__':
	main()
