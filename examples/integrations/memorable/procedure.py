"""Compile private Browser Use captures into a deterministic replay procedure.

The compiler intentionally drops coordinates, XPath, generated IDs, model text,
and observed form values.  A procedure keeps only cross-run semantic identity,
runtime parameter names, route coverage, and evidence-backed postconditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROCEDURE_SCHEMA_VERSION = '0.1.0'
SUPPORTED_ACTIONS = {'input', 'click', 'select_dropdown'}
SAFE_LOCATOR_FIELDS = (
	'node_name',
	'ax_role',
	'ax_name',
	'attributes.autocomplete',
	'attributes.name',
	'attributes.type',
	'attributes.aria-label',
	'attributes.placeholder',
	'attributes.data-action',
)
VOLATILE_FIELDS = ('attributes.id', 'x_path', 'stable_hash', 'bounds')


class ProcedureCompilationError(ValueError):
	"""Raised when the captured evidence is unsafe to compile."""


class ReplayAction(str, Enum):
	INPUT = 'input'
	CLICK = 'click'
	SELECT_DROPDOWN = 'select_dropdown'


class ParameterKind(str, Enum):
	TEXT = 'text'
	OPTION = 'option'


class PostconditionKind(str, Enum):
	CONTROL_VALUE_EQUALS = 'control_value_equals'
	CONTROL_CHECKED_EQUALS = 'control_checked_equals'
	TARGET_DISAPPEARS = 'target_disappears'
	EXPECTED_TEXT_APPEARS = 'expected_text_appears'


class ProcedureParameter(BaseModel):
	name: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
	kind: ParameterKind
	required: bool = True
	source_locator_field: str


class SemanticLocator(BaseModel):
	"""An exact conjunction of normalized, layout-independent DOM facts."""

	required: dict[str, str] = Field(min_length=1)
	training_observations: int = Field(ge=1)
	training_states_checked: int = Field(ge=1)
	unique_in_all_training_states: bool
	excluded_volatile_fields: list[str] = Field(default_factory=lambda: list(VOLATILE_FIELDS))

	@model_validator(mode='after')
	def validate_fields(self) -> SemanticLocator:
		unsupported = sorted(set(self.required) - set(SAFE_LOCATOR_FIELDS))
		if unsupported:
			raise ValueError(f'Unsupported semantic locator fields: {unsupported}')
		if not self.unique_in_all_training_states:
			raise ValueError('A replay locator must be unique in every evaluated training state')
		return self


class ValueBinding(BaseModel):
	parameter: str | None = None
	literal: str | bool | None = None

	@model_validator(mode='after')
	def exactly_one_source(self) -> ValueBinding:
		if (self.parameter is None) == (self.literal is None):
			raise ValueError('Exactly one of parameter or literal must be set')
		return self


class ProcedurePostcondition(BaseModel):
	kind: PostconditionKind
	value: ValueBinding | None = None
	text: str | None = None
	timeout_seconds: float = Field(default=3.0, gt=0, le=30)

	@model_validator(mode='after')
	def validate_payload(self) -> ProcedurePostcondition:
		if self.kind in {PostconditionKind.CONTROL_VALUE_EQUALS, PostconditionKind.CONTROL_CHECKED_EQUALS}:
			if self.value is None:
				raise ValueError(f'{self.kind.value} requires a value binding')
		elif self.kind == PostconditionKind.EXPECTED_TEXT_APPEARS:
			if not self.text:
				raise ValueError('expected_text_appears requires text')
		elif self.value is not None or self.text is not None:
			raise ValueError(f'{self.kind.value} does not accept value or text')
		return self


class ProcedureStep(BaseModel):
	id: str
	action: ReplayAction
	locator: SemanticLocator
	optional: bool
	route_coverage: float = Field(ge=0, le=1)
	order_score: float = Field(ge=0)
	value: ValueBinding | None = None
	clear: bool = True
	postcondition: ProcedurePostcondition
	source_runs: list[str]

	@model_validator(mode='after')
	def validate_action_value(self) -> ProcedureStep:
		if self.action in {ReplayAction.INPUT, ReplayAction.SELECT_DROPDOWN} and self.value is None:
			raise ValueError(f'{self.action.value} requires a value binding')
		if self.action == ReplayAction.CLICK and self.value is not None:
			target_type = self.locator.required.get('attributes.type')
			if not (target_type == 'checkbox' and isinstance(self.value.literal, bool)):
				raise ValueError('Only checkbox clicks can have a boolean literal value')
		return self


class SiteScope(BaseModel):
	allowed_origins: list[str] = Field(min_length=1)
	allowed_paths: list[str] = Field(min_length=1)

	def permits(self, url: str) -> bool:
		parts = urlsplit(url)
		origin = f'{parts.scheme.lower()}://{parts.netloc.lower()}'
		return origin in self.allowed_origins and parts.path in self.allowed_paths


class BrowserProcedure(BaseModel):
	model_config = ConfigDict(extra='forbid')

	schema_version: Literal['0.1.0'] = PROCEDURE_SCHEMA_VERSION
	procedure_id: str
	compiled_at: str
	task_fingerprint: str
	site_scope: SiteScope
	expected_success_text: str | None = None
	parameters: list[ProcedureParameter]
	steps: list[ProcedureStep] = Field(min_length=1)
	training: dict[str, Any]

	@model_validator(mode='after')
	def validate_bindings(self) -> BrowserProcedure:
		parameter_names = {parameter.name for parameter in self.parameters}
		if len(parameter_names) != len(self.parameters):
			raise ValueError('Procedure parameter names must be unique')
		for step in self.steps:
			bindings = [step.value, step.postcondition.value]
			for binding in bindings:
				if binding and binding.parameter and binding.parameter not in parameter_names:
					raise ValueError(f'Step {step.id} references unknown parameter {binding.parameter!r}')
		return self

	def write(self, path: str | Path) -> Path:
		output = Path(path).expanduser().resolve()
		output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
		payload = (self.model_dump_json(indent=2) + '\n').encode()
		temporary = output.with_name(f'.{output.name}.partial')
		temporary.write_bytes(payload)
		os.chmod(temporary, 0o600)
		temporary.replace(output)
		return output

	@classmethod
	def read(cls, path: str | Path) -> BrowserProcedure:
		return cls.model_validate_json(Path(path).read_text())


class _Observation(BaseModel):
	run_directory: str
	step: int
	action_ordinal: int
	action_name: str
	position: int
	candidate: dict[str, Any]
	action_args: dict[str, Any]
	pre_candidates: list[dict[str, Any]]


class _CapturedRun(BaseModel):
	directory: Path
	manifest: dict[str, Any]
	derived: dict[str, Any]
	history: dict[str, Any]


def compile_procedure(
	capture_root: str | Path,
	*,
	task_fingerprint: str | None = None,
	minimum_successful_runs: int = 2,
) -> BrowserProcedure:
	"""Compile one task group of evidence-backed captures into a procedure."""

	root = Path(capture_root).expanduser().resolve()
	runs = _load_eligible_runs(root)
	if not runs:
		raise ProcedureCompilationError(f'No completed successful capture bundles found in {root}')

	groups: dict[str, list[_CapturedRun]] = defaultdict(list)
	for run in runs:
		groups[str(run.derived.get('task_fingerprint') or 'unknown')].append(run)
	if task_fingerprint is None:
		if len(groups) != 1:
			raise ProcedureCompilationError(f'Capture root contains {len(groups)} task groups; pass task_fingerprint explicitly')
		task_fingerprint = next(iter(groups))
	selected_runs = groups.get(task_fingerprint, [])
	if len(selected_runs) < minimum_successful_runs:
		raise ProcedureCompilationError(
			f'Need at least {minimum_successful_runs} successful runs for {task_fingerprint}; found {len(selected_runs)}'
		)

	slots: dict[str, list[_Observation]] = defaultdict(list)
	for run in selected_runs:
		action_payloads = _history_action_payloads(run.history)
		position = 0
		occurrences: Counter[str] = Counter()
		for step in run.derived.get('steps', []):
			action_name = str(step.get('action_name') or '')
			candidate = step.get('selected_candidate')
			if action_name not in SUPPORTED_ACTIONS or not isinstance(candidate, dict):
				continue
			if step.get('outcome_status') == 'reported_error':
				continue
			base = _alignment_base(action_name, candidate)
			occurrences[base] += 1
			slot_id = f'{base}|occurrence:{occurrences[base]}'
			step_number = int(step.get('step', 0))
			action_ordinal = int(step.get('action_ordinal', 0))
			args = action_payloads.get((step_number, action_ordinal, action_name))
			if args is None:
				raise ProcedureCompilationError(
					f'Missing history payload for {run.directory.name} step {step_number} action {action_ordinal}'
				)
			pre_path = run.directory / 'steps' / f'{step_number:03d}' / 'pre.candidates.json'
			try:
				pre_candidates = json.loads(pre_path.read_text())
			except (OSError, json.JSONDecodeError) as exc:
				raise ProcedureCompilationError(f'Cannot read training candidates at {pre_path}: {exc}') from exc
			slots[slot_id].append(
				_Observation(
					run_directory=run.directory.name,
					step=step_number,
					action_ordinal=action_ordinal,
					action_name=action_name,
					position=position,
					candidate=candidate,
					action_args=args,
					pre_candidates=pre_candidates,
				)
			)
			position += 1

	if not slots:
		raise ProcedureCompilationError('No supported, resolved actions were available to compile')

	parameter_map: dict[str, ProcedureParameter] = {}
	steps = [
		_compile_step(slot_id, observations, len(selected_runs), parameter_map) for slot_id, observations in sorted(slots.items())
	]
	steps.sort(key=lambda step: (step.order_score, step.id))

	origins, paths = _site_scope(selected_runs)
	expected_success_text = _consensus_expected_success_text(selected_runs)
	_submit_postcondition(steps, expected_success_text)

	training = {
		'capture_root': str(root),
		'successful_runs': len(selected_runs),
		'run_directories': sorted(run.directory.name for run in selected_runs),
		'route_variants': _route_variants(selected_runs),
		'compiler': 'deterministic_cross_run_semantic_aggregation',
		'values_retained_from_actions': False,
		'model_calls_during_compile': 0,
		'volatile_fields_excluded': list(VOLATILE_FIELDS),
	}
	compiled_at = datetime.now(timezone.utc).isoformat()
	identity_payload = {
		'task_fingerprint': task_fingerprint,
		'site_scope': {'allowed_origins': origins, 'allowed_paths': paths},
		'expected_success_text': expected_success_text,
		'parameters': [parameter.model_dump(mode='json') for parameter in sorted(parameter_map.values(), key=lambda p: p.name)],
		'steps': [step.model_dump(mode='json') for step in steps],
		'training_runs': training['run_directories'],
	}
	procedure_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:20]
	return BrowserProcedure(
		procedure_id=procedure_id,
		compiled_at=compiled_at,
		task_fingerprint=task_fingerprint,
		site_scope=SiteScope(allowed_origins=origins, allowed_paths=paths),
		expected_success_text=expected_success_text,
		parameters=sorted(parameter_map.values(), key=lambda parameter: parameter.name),
		steps=steps,
		training=training,
	)


def _load_eligible_runs(root: Path) -> list[_CapturedRun]:
	runs: list[_CapturedRun] = []
	for derived_path in sorted(root.glob('*/derived.json')):
		directory = derived_path.parent
		try:
			manifest = json.loads((directory / 'manifest.json').read_text())
			derived = json.loads(derived_path.read_text())
			history = json.loads((directory / 'history.json').read_text())
		except (OSError, json.JSONDecodeError):
			continue
		if manifest.get('state') != 'completed':
			continue
		if (manifest.get('terminal') or {}).get('is_successful') is not True:
			continue
		if manifest.get('capture_errors'):
			continue
		verification = derived.get('offline_verification') or {}
		if verification.get('enabled') is True and verification.get('observed_in_rendered_page') is not True:
			continue
		runs.append(_CapturedRun(directory=directory, manifest=manifest, derived=derived, history=history))
	return runs


def _history_action_payloads(history: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
	payloads: dict[tuple[int, int, str], dict[str, Any]] = {}
	for ordinal, item in enumerate(history.get('history', [])):
		metadata = item.get('metadata') or {}
		step_number = int(metadata.get('step_number', ordinal))
		model_output = item.get('model_output') or {}
		for action_ordinal, action in enumerate(model_output.get('action') or []):
			if not isinstance(action, dict) or not action:
				continue
			name = next(iter(action))
			args = action.get(name)
			if isinstance(args, dict):
				payloads[(step_number, action_ordinal, name)] = args
	return payloads


def _compile_step(
	slot_id: str,
	observations: list[_Observation],
	run_count: int,
	parameter_map: dict[str, ProcedureParameter],
) -> ProcedureStep:
	action_name = observations[0].action_name
	if any(observation.action_name != action_name for observation in observations):
		raise ProcedureCompilationError(f'Action mismatch inside slot {slot_id}')
	locator_fields = _stable_locator_fields([observation.candidate for observation in observations])
	if not locator_fields:
		raise ProcedureCompilationError(f'No stable semantic locator fields for {slot_id}')

	checks: list[int] = []
	for observation in observations:
		checks.append(sum(1 for candidate in observation.pre_candidates if locator_matches(locator_fields, candidate)))
	if not checks or any(match_count != 1 for match_count in checks):
		raise ProcedureCompilationError(f'Locator for {slot_id} is not unique in all training states: match counts {checks}')

	parameter: ProcedureParameter | None = None
	value: ValueBinding | None = None
	candidate = observations[0].candidate
	attributes = candidate.get('attributes') or {}
	if action_name in {'input', 'select_dropdown'}:
		parameter = _parameter_for(candidate, action_name)
		existing = parameter_map.get(parameter.name)
		if existing is not None and existing != parameter:
			raise ProcedureCompilationError(f'Conflicting parameter inference for {parameter.name}')
		parameter_map[parameter.name] = parameter
		value = ValueBinding(parameter=parameter.name)
	elif action_name == 'click' and _normalize(attributes.get('type')) == 'checkbox':
		value = ValueBinding(literal=True)

	postcondition = _postcondition_for(action_name, locator_fields, value)
	return ProcedureStep(
		id=slot_id,
		action=ReplayAction(action_name),
		locator=SemanticLocator(
			required=locator_fields,
			training_observations=len(observations),
			training_states_checked=len(checks),
			unique_in_all_training_states=True,
		),
		optional=len({observation.run_directory for observation in observations}) < run_count,
		route_coverage=round(len({observation.run_directory for observation in observations}) / run_count, 3),
		order_score=round(sum(observation.position for observation in observations) / len(observations), 3),
		value=value,
		clear=all(bool(observation.action_args.get('clear', True)) for observation in observations),
		postcondition=postcondition,
		source_runs=sorted({observation.run_directory for observation in observations}),
	)


def _stable_locator_fields(candidates: list[dict[str, Any]]) -> dict[str, str]:
	required: dict[str, str] = {}
	for field in SAFE_LOCATOR_FIELDS:
		values = [_normalize(_candidate_field(candidate, field)) for candidate in candidates]
		populated = [value for value in values if value]
		if len(populated) != len(candidates):
			continue
		winning_value, winning_count = Counter(populated).most_common(1)[0]
		if winning_count / len(candidates) >= 0.8:
			required[field] = winning_value
	return required


def locator_matches(required: dict[str, str], candidate: dict[str, Any]) -> bool:
	return all(_normalize(_candidate_field(candidate, field)) == value for field, value in required.items())


def _candidate_field(candidate: dict[str, Any], field: str) -> Any:
	if field.startswith('attributes.'):
		return (candidate.get('attributes') or {}).get(field.removeprefix('attributes.'))
	return candidate.get(field)


def _alignment_base(action_name: str, candidate: dict[str, Any]) -> str:
	attributes = candidate.get('attributes') or {}
	for field in ('autocomplete', 'name'):
		value = _normalize(attributes.get(field))
		if value:
			return f'{action_name}|{field}:{value}'
	input_type = _normalize(attributes.get('type'))
	if input_type and (_normalize(candidate.get('node_name')) == 'input' or input_type == 'submit'):
		return f'{action_name}|type:{input_type}'
	role = _normalize(candidate.get('ax_role'))
	name = _normalize(candidate.get('ax_name'))
	if role or name:
		return f'{action_name}|ax:{role}|{name}'
	return f'{action_name}|unresolved'


def _parameter_for(candidate: dict[str, Any], action_name: str) -> ProcedureParameter:
	attributes = candidate.get('attributes') or {}
	for field in ('name', 'autocomplete', 'aria-label'):
		value = attributes.get(field)
		if value:
			return ProcedureParameter(
				name=_snake_case(str(value)),
				kind=ParameterKind.TEXT if action_name == 'input' else ParameterKind.OPTION,
				source_locator_field=f'attributes.{field}',
			)
	name = candidate.get('ax_name') or candidate.get('meaningful_text')
	if not name:
		raise ProcedureCompilationError('Cannot infer a safe runtime parameter name for an input action')
	return ProcedureParameter(
		name=_snake_case(str(name)),
		kind=ParameterKind.TEXT if action_name == 'input' else ParameterKind.OPTION,
		source_locator_field='ax_name',
	)


def _postcondition_for(action_name: str, locator_fields: dict[str, str], value: ValueBinding | None) -> ProcedurePostcondition:
	if action_name in {'input', 'select_dropdown'}:
		return ProcedurePostcondition(kind=PostconditionKind.CONTROL_VALUE_EQUALS, value=value)
	if locator_fields.get('attributes.type') == 'checkbox':
		return ProcedurePostcondition(
			kind=PostconditionKind.CONTROL_CHECKED_EQUALS,
			value=ValueBinding(literal=True),
		)
	return ProcedurePostcondition(kind=PostconditionKind.TARGET_DISAPPEARS)


def _submit_postcondition(steps: list[ProcedureStep], expected_text: str | None) -> None:
	if not expected_text:
		return
	for step in steps:
		if step.locator.required.get('attributes.type') == 'submit':
			step.postcondition = ProcedurePostcondition(
				kind=PostconditionKind.EXPECTED_TEXT_APPEARS,
				text=expected_text,
				timeout_seconds=5.0,
			)


def _site_scope(runs: list[_CapturedRun]) -> tuple[list[str], list[str]]:
	urls: set[str] = set()
	for run in runs:
		for item in run.history.get('history', []):
			for action in (item.get('model_output') or {}).get('action') or []:
				params = action.get('navigate') if isinstance(action, dict) else None
				if isinstance(params, dict) and isinstance(params.get('url'), str):
					urls.add(params['url'])
	if not urls:
		raise ProcedureCompilationError('Cannot infer site scope because training runs contain no navigate action')
	origins = sorted({f'{urlsplit(url).scheme.lower()}://{urlsplit(url).netloc.lower()}' for url in urls})
	paths = sorted({urlsplit(url).path for url in urls})
	if any(not origin.startswith(('http://', 'https://')) for origin in origins):
		raise ProcedureCompilationError(f'Only HTTP(S) training origins are supported: {origins}')
	return origins, paths


def _consensus_expected_success_text(runs: list[_CapturedRun]) -> str | None:
	values = [
		str((run.manifest.get('capture_options') or {}).get('expected_success_text'))
		for run in runs
		if (run.manifest.get('capture_options') or {}).get('expected_success_text')
	]
	if not values:
		return None
	value, count = Counter(values).most_common(1)[0]
	if count != len(runs):
		raise ProcedureCompilationError('Successful runs disagree on the expected success evidence')
	return value


def _route_variants(runs: list[_CapturedRun]) -> list[dict[str, Any]]:
	counts = Counter(tuple(run.derived.get('route_signature') or []) for run in runs)
	return [{'actions': list(route), 'runs': count} for route, count in counts.most_common()]


def _snake_case(value: str) -> str:
	value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', value)
	value = re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_').lower()
	if not value:
		raise ProcedureCompilationError('Cannot derive a runtime parameter name from an empty label')
	if value[0].isdigit():
		value = f'field_{value}'
	return value


def _normalize(value: Any) -> str:
	return re.sub(r'\s+', ' ', str(value or '').strip()).casefold()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('capture_root', type=Path)
	parser.add_argument('output', type=Path)
	parser.add_argument('--task-fingerprint')
	parser.add_argument('--minimum-successful-runs', type=int, default=2)
	args = parser.parse_args()
	procedure = compile_procedure(
		args.capture_root,
		task_fingerprint=args.task_fingerprint,
		minimum_successful_runs=args.minimum_successful_runs,
	)
	path = procedure.write(args.output)
	print(f'wrote {len(procedure.steps)} semantic steps to {path} ({procedure.procedure_id})')


if __name__ == '__main__':
	main()
