import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from examples.integrations.memorable import replay as replay_module
from examples.integrations.memorable.procedure import (
	BrowserProcedure,
	ParameterKind,
	PostconditionKind,
	ProcedureParameter,
	ProcedurePostcondition,
	ProcedureStep,
	ReplayAction,
	SemanticLocator,
	SiteScope,
	ValueBinding,
	_effective_action_steps,
	_stable_locator_fields,
	compile_procedure,
)
from examples.integrations.memorable.replay import (
	DeterministicReplayer,
	ReplayOptions,
	ReplayStatus,
	StepStatus,
)


def _locator(**required: str) -> SemanticLocator:
	return SemanticLocator(
		required=required,
		training_observations=3,
		training_states_checked=3,
		unique_in_all_training_states=True,
	)


def _procedure(url: str) -> BrowserProcedure:
	parts = urlsplit(url)
	origin = f'{parts.scheme}://{parts.netloc}'
	email = _locator(
		node_name='input',
		ax_role='textbox',
		ax_name='email address',
		**{
			'attributes.name': 'email',
			'attributes.type': 'email',
			'attributes.autocomplete': 'email',
		},
	)
	full_name = _locator(
		node_name='input',
		ax_role='textbox',
		ax_name='full name',
		**{'attributes.name': 'fullname', 'attributes.autocomplete': 'name'},
	)
	continue_button = SemanticLocator(
		required={
			'node_name': 'button',
			'ax_role': 'button',
			'ax_name': 'continue',
			'attributes.type': 'button',
		},
		training_observations=1,
		training_states_checked=1,
		unique_in_all_training_states=True,
	)
	country = _locator(
		node_name='select',
		ax_role='combobox',
		ax_name='country',
		**{'attributes.name': 'country'},
	)
	terms = _locator(
		node_name='input',
		ax_role='checkbox',
		ax_name='accept terms',
		**{'attributes.name': 'terms', 'attributes.type': 'checkbox'},
	)
	submit = _locator(
		node_name='button',
		ax_role='button',
		**{'attributes.type': 'submit'},
	)
	return BrowserProcedure(
		procedure_id='fixture-procedure',
		compiled_at='2026-08-31T00:00:00+00:00',
		task_fingerprint='fixture-registration',
		site_scope=SiteScope(
			allowed_origins=[origin],
			allowed_paths=['/variant-a.html', '/variant-b.html'],
		),
		expected_success_text='Registration complete. Success code: MEM-042',
		parameters=[
			ProcedureParameter(
				name='country',
				kind=ParameterKind.OPTION,
				source_locator_field='attributes.name',
			),
			ProcedureParameter(
				name='email',
				kind=ParameterKind.TEXT,
				source_locator_field='attributes.name',
			),
			ProcedureParameter(
				name='full_name',
				kind=ParameterKind.TEXT,
				source_locator_field='attributes.name',
			),
		],
		steps=[
			ProcedureStep(
				id='email',
				action=ReplayAction.INPUT,
				locator=email,
				optional=False,
				route_coverage=1,
				order_score=0,
				value=ValueBinding(parameter='email'),
				postcondition=ProcedurePostcondition(
					kind=PostconditionKind.CONTROL_VALUE_EQUALS,
					value=ValueBinding(parameter='email'),
				),
				source_runs=['a', 'b', 'c'],
			),
			ProcedureStep(
				id='full-name',
				action=ReplayAction.INPUT,
				locator=full_name,
				optional=False,
				route_coverage=1,
				order_score=1,
				value=ValueBinding(parameter='full_name'),
				postcondition=ProcedurePostcondition(
					kind=PostconditionKind.CONTROL_VALUE_EQUALS,
					value=ValueBinding(parameter='full_name'),
				),
				source_runs=['a', 'b', 'c'],
			),
			ProcedureStep(
				id='continue',
				action=ReplayAction.CLICK,
				locator=continue_button,
				optional=True,
				route_coverage=0.333,
				order_score=2,
				postcondition=ProcedurePostcondition(kind=PostconditionKind.TARGET_DISAPPEARS),
				source_runs=['a'],
			),
			ProcedureStep(
				id='country',
				action=ReplayAction.SELECT_DROPDOWN,
				locator=country,
				optional=False,
				route_coverage=1,
				order_score=3,
				value=ValueBinding(parameter='country'),
				postcondition=ProcedurePostcondition(
					kind=PostconditionKind.CONTROL_VALUE_EQUALS,
					value=ValueBinding(parameter='country'),
				),
				source_runs=['a', 'b', 'c'],
			),
			ProcedureStep(
				id='terms',
				action=ReplayAction.CLICK,
				locator=terms,
				optional=False,
				route_coverage=1,
				order_score=4,
				value=ValueBinding(literal=True),
				postcondition=ProcedurePostcondition(
					kind=PostconditionKind.CONTROL_CHECKED_EQUALS,
					value=ValueBinding(literal=True),
				),
				source_runs=['a', 'b', 'c'],
			),
			ProcedureStep(
				id='submit',
				action=ReplayAction.CLICK,
				locator=submit,
				optional=False,
				route_coverage=1,
				order_score=5,
				postcondition=ProcedurePostcondition(
					kind=PostconditionKind.EXPECTED_TEXT_APPEARS,
					text='Registration complete. Success code: MEM-042',
				),
				source_runs=['a', 'b', 'c'],
			),
		],
		training={'successful_runs': 3, 'model_calls_during_compile': 0},
	)


def _parameters() -> dict[str, str]:
	return {
		'email': 'grace.hopper@example.test',
		'full_name': 'Grace Hopper',
		'country': 'Canada',
	}


def _write_synthetic_run(
	root: Path,
	run_id: str,
	button_id: str,
	button_index: int,
) -> None:
	run = root / run_id
	(run / 'steps' / '001').mkdir(parents=True)
	candidate = {
		'selector_index': button_index,
		'node_name': 'BUTTON',
		'ax_role': 'button',
		'ax_name': 'Continue',
		'x_path': f'body/div[{button_index}]/button',
		'stable_hash': button_index * 123,
		'attributes': {'id': button_id, 'type': 'button'},
	}
	(run / 'steps' / '001' / 'pre.candidates.json').write_text(json.dumps([candidate]))
	(run / 'manifest.json').write_text(
		json.dumps(
			{
				'state': 'completed',
				'terminal': {'is_successful': True},
				'capture_errors': [],
				'capture_options': {},
			}
		)
	)
	(run / 'derived.json').write_text(
		json.dumps(
			{
				'run_id': run_id,
				'task_fingerprint': 'task-1',
				'route_signature': ['navigate', 'click', 'done'],
				'offline_verification': {'enabled': False},
				'steps': [
					{
						'step': 1,
						'action_ordinal': 0,
						'action_name': 'click',
						'outcome_status': 'no_reported_error',
						'selected_candidate': candidate,
					}
				],
			}
		)
	)
	(run / 'history.json').write_text(
		json.dumps(
			{
				'history': [
					{
						'metadata': {'step_number': 0},
						'model_output': {
							'action': [
								{
									'navigate': {
										'url': 'http://127.0.0.1:8765/fixture.html',
										'new_tab': False,
									}
								}
							]
						},
					},
					{
						'metadata': {'step_number': 1},
						'model_output': {'action': [{'click': {'index': button_index}}]},
					},
				]
			}
		)
	)


def test_compiler_retains_semantics_and_drops_volatile_inputs(tmp_path):
	_write_synthetic_run(tmp_path, 'run-a', 'random-a1b2c3d4', 7)
	_write_synthetic_run(tmp_path, 'run-b', 'random-deadbeef', 42)

	procedure = compile_procedure(tmp_path)

	assert len(procedure.steps) == 1
	step = procedure.steps[0]
	assert step.locator.required == {
		'node_name': 'button',
		'ax_role': 'button',
		'ax_name': 'continue',
		'attributes.type': 'button',
	}
	serialized = procedure.model_dump_json()
	assert 'random-a1b2c3d4' not in serialized
	assert 'selector_index' not in serialized
	assert 'x_path' not in step.locator.required
	assert 'stable_hash' not in step.locator.required
	assert procedure.training['model_calls_during_compile'] == 0


def test_compiler_can_migrate_legacy_capture_identity(tmp_path):
	_write_synthetic_run(tmp_path, 'run-a', 'random-a1b2c3d4', 7)
	_write_synthetic_run(tmp_path, 'run-b', 'random-deadbeef', 42)

	procedure = compile_procedure(tmp_path, procedure_task_fingerprint='canonical-task-fingerprint')

	assert procedure.task_fingerprint == 'canonical-task-fingerprint'
	assert procedure.training['capture_task_fingerprint'] == 'task-1'


def test_retry_recovery_collapses_to_terminal_control_actions():
	def step(action_name: str, name: str, input_type: str = 'text') -> dict:
		return {
			'action_name': action_name,
			'selected_candidate': {
				'node_name': 'input' if input_type != 'submit' else 'button',
				'attributes': {'name': name, 'type': input_type},
			},
		}

	raw = [
		step('input', 'email', 'email'),
		step('click', 'help', 'button'),
		step('input', 'email', 'email'),
		step('click', 'submit', 'submit'),
		step('click', 'submit', 'submit'),
	]

	effective = _effective_action_steps(raw)

	assert effective == [raw[1], raw[2], raw[4]]


def test_three_run_majority_can_select_unique_radio_value():
	candidates = [
		{
			'node_name': 'INPUT',
			'ax_role': 'radio',
			'ax_name': label,
			'attributes': {'name': 'contactMethod', 'type': 'radio', 'value': value},
		}
		for label, value in [('Email', 'email'), ('Phone', 'phone'), ('Phone', 'phone')]
	]

	locator = _stable_locator_fields(candidates)

	assert locator['attributes.value'] == 'phone'
	assert locator['ax_name'] == 'phone'


def test_file_upload_contract_needs_no_secret_value_in_postcondition():
	step = ProcedureStep(
		id='document',
		action=ReplayAction.UPLOAD_FILE,
		locator=_locator(
			node_name='input',
			ax_role='button',
			ax_name='document',
			**{'attributes.name': 'document', 'attributes.type': 'file'},
		),
		optional=False,
		route_coverage=1,
		order_score=0,
		value=ValueBinding(parameter='document'),
		postcondition=ProcedurePostcondition(kind=PostconditionKind.CONTROL_FILES_SELECTED),
		source_runs=['a', 'b', 'c'],
	)

	assert step.postcondition.value is None


@pytest.mark.asyncio
async def test_file_upload_postcondition_reads_browser_file_count_without_value_binding(monkeypatch, tmp_path):
	step = ProcedureStep(
		id='document',
		action=ReplayAction.UPLOAD_FILE,
		locator=_locator(
			node_name='input',
			ax_role='button',
			ax_name='document',
			**{'attributes.name': 'document', 'attributes.type': 'file'},
		),
		optional=False,
		route_coverage=1,
		order_score=0,
		value=ValueBinding(parameter='document'),
		postcondition=ProcedurePostcondition(kind=PostconditionKind.CONTROL_FILES_SELECTED),
		source_runs=['a', 'b', 'c'],
	)

	async def live_page(_browser_session):
		return {
			'rendered_text': '',
			'controls': [
				{
					'node_name': 'input',
					'ax_role': 'button',
					'ax_name': 'document',
					'attributes': {'name': 'document', 'type': 'file'},
					'file_count': 1,
				}
			],
		}

	monkeypatch.setattr(replay_module, '_read_live_page', live_page)
	verification = await DeterministicReplayer(
		_procedure('http://127.0.0.1:9000/variant-a.html'), ReplayOptions(output_dir=tmp_path)
	)._check_postcondition(None, step, {})  # type: ignore[arg-type]

	assert verification.verified is True
	assert verification.evidence['observed_file_count'] == 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
	('variant', 'expected_continue_status'),
	[
		('variant-a.html', StepStatus.EXECUTED),
		('variant-b.html?notice=0&delay=25', StepStatus.SKIPPED_OPTIONAL),
	],
)
async def test_real_browser_replays_reordered_variants_without_model(
	browser_session,
	httpserver,
	tmp_path,
	variant,
	expected_continue_status,
):
	sites = Path(__file__).parents[2] / 'examples' / 'integrations' / 'memorable' / 'sites'
	for filename in ('variant-a.html', 'variant-b.html'):
		httpserver.expect_request(f'/{filename}').respond_with_data(
			(sites / filename).read_text(),
			content_type='text/html',
		)
	path, _, query = variant.partition('?')
	url = str(httpserver.url_for(f'/{path}')) + (f'?{query}' if query else '')
	await browser_session.navigate_to(url)

	report = await DeterministicReplayer(
		_procedure(url),
		ReplayOptions(output_dir=tmp_path / 'audits', optional_wait_seconds=0.05),
	).run(browser_session, _parameters())

	assert report.status == ReplayStatus.COMPLETED, report.reason
	assert report.model_calls == 0
	assert next(event for event in report.events if event.step_id == 'continue').status == expected_continue_status
	assert Path(report.audit_dir or '').joinpath('manifest.json').exists()
	manifest = json.loads(Path(report.audit_dir or '').joinpath('manifest.json').read_text())
	assert manifest['privacy']['runtime_parameter_values_retained'] is False
	page = await browser_session.must_get_current_page()
	assert (
		await page.evaluate("() => document.body.innerText.includes('Registration complete. Success code: MEM-042')")
	).casefold() == 'true'


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_browser_refuses_ambiguous_submit_instead_of_guessing(
	browser_session,
	httpserver,
	tmp_path,
):
	site = Path(__file__).parents[2] / 'examples' / 'integrations' / 'memorable' / 'sites' / 'variant-b.html'
	httpserver.expect_request('/variant-b.html').respond_with_data(
		site.read_text(),
		content_type='text/html',
	)
	url = str(httpserver.url_for('/variant-b.html')) + '?notice=0&duplicate_submit=1&delay=0'
	await browser_session.navigate_to(url)

	report = await DeterministicReplayer(
		_procedure(url),
		ReplayOptions(output_dir=tmp_path / 'audits', optional_wait_seconds=0.05),
	).run(browser_session, _parameters())

	assert report.status == ReplayStatus.NEEDS_RECOVERY
	assert report.reason is not None and 'submit: ambiguous' in report.reason
	assert report.model_calls == 0
	assert report.actions_attempted == 4
	page = await browser_session.must_get_current_page()
	assert (
		await page.evaluate("() => document.body.innerText.includes('Registration complete. Success code: MEM-042')")
	).casefold() == 'false'


@pytest.mark.asyncio
async def test_kill_switch_prevents_browser_reads_and_audit_writes(monkeypatch, tmp_path):
	monkeypatch.setenv('BROWSER_USE_MEMORABLE_SEMANTIC_REPLAY', '0')
	report = await DeterministicReplayer(
		_procedure('http://127.0.0.1:9000/variant-a.html'),
		ReplayOptions(output_dir=tmp_path),
	).run(None, _parameters())  # type: ignore[arg-type]

	assert report.status == ReplayStatus.DISABLED
	assert list(tmp_path.iterdir()) == []
