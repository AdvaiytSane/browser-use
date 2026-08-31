import json
import os
import stat
from pathlib import Path

import pytest

from browser_use import Agent
from examples.integrations.memorable.offline_capture import (
	OfflineCaptureOptions,
	OfflineRunCapture,
	_result_status,
	analyze_capture_corpus,
)
from tests.ci.conftest import create_mock_llm


def _read_json(path: Path):
	return json.loads(path.read_text())


def test_result_status_does_not_invent_regular_action_success():
	assert _result_status(None) == 'outcome_unavailable'
	assert _result_status({}) == 'outcome_unavailable'
	assert _result_status({'error': 'not found'}) == 'reported_error'
	assert _result_status({'is_done': True, 'success': True}) == 'terminal_success'
	assert _result_status({'is_done': True, 'success': False}) == 'terminal_failure'
	assert _result_status({'long_term_memory': 'clicked'}) == 'no_reported_error'


@pytest.mark.asyncio
async def test_kill_switch_runs_agent_without_creating_capture(monkeypatch, tmp_path):
	monkeypatch.setenv('BROWSER_USE_OFFLINE_CAPTURE', '0')
	sentinel = object()

	class FakeAgent:
		async def run(self, **kwargs):
			return sentinel

	capture = OfflineRunCapture(OfflineCaptureOptions(output_dir=tmp_path))
	result = await capture.run_agent(FakeAgent())  # type: ignore[arg-type]

	assert result is sentinel
	assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_browser_agent_produces_joined_pre_post_bundle(browser_session, httpserver, tmp_path):
	httpserver.expect_request('/capture').respond_with_data(
		"""
		<!doctype html>
		<html>
			<head><title>Offline capture fixture</title></head>
			<body>
				<main aria-label="Checkout flow">
					<h1>Ready</h1>
					<label for="name-a1b2c3d4">Name</label>
					<input id="name-a1b2c3d4" name="fullName" autocomplete="name">
					<button id="button-a1b2c3d4" aria-label="Continue"
						onclick="this.hidden=true; document.querySelector('#success').hidden=false; document.querySelector('h1').textContent='Complete'">
						Continue
					</button>
					<p id="success" hidden>Success: dumbledore</p>
				</main>
			</body>
		</html>
		""",
		content_type='text/html',
	)
	url = httpserver.url_for('/capture')
	await browser_session.navigate_to(url)
	initial_state = await browser_session.get_browser_state_summary(include_screenshot=False)
	button_index = next(
		index
		for index, node in initial_state.dom_state.selector_map.items()
		if node.ax_node is not None and node.ax_node.name == 'Continue'
	)
	name_index = next(
		index for index, node in initial_state.dom_state.selector_map.items() if node.attributes.get('name') == 'fullName'
	)
	input_action = """
	{
		"thinking": "The name field should be filled before continuing.",
		"evaluation_previous_goal": "The page loaded.",
		"memory": "The Name field and Continue button are visible.",
		"next_goal": "Enter Ada Lovelace.",
		"action": [{"input": {"index": __INDEX__, "text": "Ada Lovelace", "clear": true}}]
	}
	""".replace('__INDEX__', str(name_index))
	click_action = """
	{
		"thinking": "The only button advances the fixture.",
		"evaluation_previous_goal": "The name was entered.",
		"memory": "Ada Lovelace is in the Name field and the Continue button is visible.",
		"next_goal": "Click Continue.",
		"action": [{"click": {"index": __INDEX__}}]
	}
	""".replace('__INDEX__', str(button_index))
	done_action = """
	{
		"thinking": "The success marker is visible.",
		"evaluation_previous_goal": "Continue revealed the success marker.",
		"memory": "Success: dumbledore is visible.",
		"next_goal": "Finish.",
		"action": [{"done": {"text": "dumbledore", "success": true}}]
	}
	"""
	agent = Agent(
		task='Click Continue and finish when dumbledore appears.',
		llm=create_mock_llm(actions=[input_action, click_action, done_action]),
		browser_session=browser_session,
		max_actions_per_step=1,
		use_judge=False,
		directly_open_url=False,
	)
	capture = OfflineRunCapture(
		OfflineCaptureOptions(
			output_dir=tmp_path,
			include_full_dom=True,
			include_screenshots=True,
			include_conversations=True,
			expected_success_text='Success: dumbledore',
			run_label='pytest-real-browser',
		)
	)

	history = await capture.run_agent(agent, max_steps=4)

	assert history.is_successful() is True
	assert capture.final_dir.exists()
	manifest = _read_json(capture.final_dir / 'manifest.json')
	derived = _read_json(capture.final_dir / 'derived.json')
	history_json = _read_json(capture.final_dir / 'history.json')
	usage = _read_json(capture.final_dir / 'usage.json')

	assert manifest['state'] == 'completed'
	assert manifest['terminal']['is_successful'] is True
	assert manifest['counts']['actions'] == manifest['counts']['results']
	assert manifest['privacy']['network_egress_by_collector'] is False
	assert history_json['history']
	assert usage is not None
	assert (capture.final_dir / 'events.jsonl').read_text().count('\n') == len(history.history)
	assert any((capture.final_dir / 'steps').glob('*/pre.candidates.json'))
	assert any((capture.final_dir / 'steps').glob('*/post.candidates.json'))
	assert any((capture.final_dir / 'steps').glob('*/pre.full.dom.json.gz'))
	assert any((capture.final_dir / 'steps').glob('*/post.html.gz'))
	assert any((capture.final_dir / 'steps').glob('*/post.rendered.txt'))
	assert derived['offline_verification']['observed_in_rendered_page'] is True
	assert derived['offline_verification']['agreement_with_agent'] is True
	assert any('Success: dumbledore' in path.read_text() for path in (capture.final_dir / 'steps').glob('*/post.rendered.txt'))

	click_steps = [step for step in derived['steps'] if step['action_name'] == 'click']
	assert len(click_steps) == 1
	click_step = click_steps[0]
	assert click_step['selected_candidate']['ax_name'] == 'Continue'
	assert click_step['observed_facts']['dom_changed'] is True
	assert click_step['inference']['transition_type'] in {
		'collapse_or_navigation_within_page',
		'dom_mutation',
	}
	assert click_step['outcome_status'] == 'no_reported_error'
	assert click_step['target_identity_quality']['score'] > 0
	input_step = next(step for step in derived['steps'] if step['action_name'] == 'input')
	assert input_step['inference']['transition_type'] == 'form_state_mutation'
	assert any(
		hint['kind'] == 'control_value_equals' and hint['value'] == 'Ada Lovelace'
		for hint in input_step['inference']['postcondition_hints']
	)

	assert stat.S_IMODE(capture.final_dir.stat().st_mode) == 0o700
	for directory in (path for path in capture.final_dir.rglob('*') if path.is_dir()):
		assert stat.S_IMODE(directory.stat().st_mode) == 0o700
	for artifact in manifest['artifacts']:
		assert stat.S_IMODE((capture.final_dir / artifact['path']).stat().st_mode) == 0o600


def test_corpus_analysis_infers_cross_run_stable_semantics(tmp_path):
	task_fingerprint = 'task-123'
	for ordinal, volatile_id in enumerate(['generated-a1b2c3d4', 'generated-deadbeef']):
		run_dir = tmp_path / f'run-{ordinal}'
		run_dir.mkdir()
		candidate = {
			'node_name': 'BUTTON',
			'ax_role': 'button',
			'ax_name': 'Continue',
			'x_path': f'body/div[{ordinal + 1}]/button',
			'stable_hash': 42,
			'attributes': {'id': volatile_id, 'type': 'button'},
		}
		step_dir = run_dir / 'steps' / '001'
		step_dir.mkdir(parents=True)
		(step_dir / 'pre.candidates.json').write_text(json.dumps([candidate]))
		(run_dir / 'manifest.json').write_text(json.dumps({'terminal': {'is_successful': True}, 'capture_errors': []}))
		(run_dir / 'derived.json').write_text(
			json.dumps(
				{
					'task_fingerprint': task_fingerprint,
					'route_signature': ['navigate', 'click', 'done'],
					'steps': [
						{
							'step': 1,
							'action_ordinal': 0,
							'selected_candidate': candidate,
						}
					],
				}
			)
		)

	report = analyze_capture_corpus(tmp_path)

	assert report['run_count'] == 2
	group = report['task_groups'][0]
	assert group['route_consensus'] == 1.0
	assert group['inference']['label'] == 'ready_for_adaptive_replay_experiment'
	assert 'small_sample_size:fewer_than_3_evidence_backed_successes' in group['inference']['risks']
	assert group['stable_targets'][0]['locator_uniqueness']['unique_in_all_evaluated_states'] is True
	fields = {field['field']: field for field in group['stable_targets'][0]['stable_fields']}
	assert fields['ax_name']['recommended'] is True
	assert fields['ax_role']['recommended'] is True
	assert fields['attributes.id']['stability'] == 0.5
	assert fields['attributes.id']['recommended'] is False
	assert fields['stable_hash']['recommended'] is False


def test_corpus_does_not_trust_agent_success_when_page_evidence_disagrees(tmp_path):
	run_dir = tmp_path / 'run-disagreement'
	run_dir.mkdir()
	(run_dir / 'manifest.json').write_text(json.dumps({'terminal': {'is_successful': True}, 'capture_errors': []}))
	(run_dir / 'derived.json').write_text(
		json.dumps(
			{
				'task_fingerprint': 'task-disagreement',
				'route_signature': ['done'],
				'offline_verification': {
					'enabled': True,
					'observed_in_rendered_page': False,
					'agreement_with_agent': False,
				},
				'steps': [],
			}
		)
	)

	group = analyze_capture_corpus(tmp_path)['task_groups'][0]

	assert group['agent_reported_successful_runs'] == 1
	assert group['evidence_verified_successful_runs'] == 0
	assert group['successful_runs'] == 0
	assert group['outcome_disagreements'] == ['run-disagreement']
	assert group['inference']['label'] == 'needs_more_evidence'


def test_capture_options_do_not_contain_provider_keys(tmp_path, monkeypatch):
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-example-secret-value-123456789')
	options = OfflineCaptureOptions(output_dir=tmp_path, include_conversations=True, include_har=True)
	dumped = json.dumps(options.model_dump(mode='json'))
	assert 'ANTHROPIC_API_KEY' not in dumped
	assert 'sk-ant-' not in dumped
	assert os.environ['ANTHROPIC_API_KEY'].startswith('sk-ant-')
