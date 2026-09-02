from datetime import date

import pytest
from pydantic import ValidationError

from examples.integrations.memorable.airbnb_demo_server import (
	STATIC_ROOT,
	AirbnbDemoRequest,
	AirbnbLiveRunStore,
	airbnb_demo_report_payload,
	airbnb_live_event_payload,
	airbnb_routing_error_payload,
)
from examples.integrations.memorable.airbnb_hybrid import (
	AgentListingSelection,
	AirbnbHybridReport,
	AirbnbSearchParameters,
	AirbnbStateEvidence,
	AirbnbTaskRoute,
	AirbnbWorkflow,
	ExecutionMode,
	HybridEvent,
	ListingCandidate,
	RouteDisposition,
	WorkflowStatus,
)


def test_demo_request_is_bounded_and_rejects_extra_fields() -> None:
	assert AirbnbDemoRequest(task='Find the cheapest stay').task == 'Find the cheapest stay'
	assert (
		AirbnbDemoRequest(
			task='Find the cheapest stay',
			client_run_id='12345678-1234-1234-1234-123456789abc',
		).client_run_id
		== '12345678-1234-1234-1234-123456789abc'
	)
	with pytest.raises(ValidationError):
		AirbnbDemoRequest.model_validate({'task': 'Find a stay', 'book_it': True})
	with pytest.raises(ValidationError):
		AirbnbDemoRequest(task='Find a stay', client_run_id='../trace.json')
	with pytest.raises(ValidationError):
		AirbnbDemoRequest(task='x' * 1001)


def test_demo_has_a_real_prefilled_task_not_a_placeholder_only() -> None:
	html = (STATIC_ROOT / 'airbnb.html').read_text()
	css = (STATIC_ROOT / 'airbnb.css').read_text()
	javascript = (STATIC_ROOT / 'airbnb.js').read_text()

	assert '>Find the least expensive Airbnb in Chicago' in html
	assert 'placeholder="Describe an Airbnb search"' in html
	assert 'id="browser-shot"' in html
	assert 'id="action-stream"' in html
	assert 'id="screen-mode"' in html
	assert 'Agent decision' in html
	assert 'Popup repair' in html
	assert 'data-objective' not in html
	for mode in ('deterministic', 'agent', 'repair', 'refuse'):
		assert f'.browser-frame[data-mode="{mode}"]' in css
	assert 'setScreenMode(latestEvents.at(-1)?.mode)' in javascript


def test_live_event_payload_is_sanitized_and_points_to_run_scoped_screenshot(tmp_path) -> None:
	screenshot = tmp_path / 'state.png'
	screenshot.write_bytes(b'png')
	event = HybridEvent(
		node='search_results',
		mode=ExecutionMode.DETERMINISTIC,
		status='verified',
		duration_ms=245,
		evidence={'candidate_count': 18, 'raw_candidates': [{'secret': 'not for the UI'}]},
		state=AirbnbStateEvidence(
			captured_at='2026-09-01T12:00:00+00:00',
			url='https://www.airbnb.com/s/Boston/homes',
			title='Boston stays',
			semantic_dom_sha256='a' * 64,
			selector_count=92,
			screenshot_path=str(screenshot),
		),
	)

	payload = airbnb_live_event_payload(event, index=2, live_run_id='run-1')

	assert payload['detail'] == 'Frozen 18 eligible cards from the initial DOM.'
	assert payload['state']['dom_signature'] == 'a' * 12
	assert payload['state']['screenshot_url'] == '/api/airbnb/live/screenshot?run_id=run-1&event=2'
	assert 'raw_candidates' not in payload


def test_live_run_store_keeps_screenshot_paths_private(tmp_path) -> None:
	screenshot = tmp_path / 'state.png'
	screenshot.write_bytes(b'png')
	event = HybridEvent(
		node='listing_details',
		mode=ExecutionMode.DETERMINISTIC,
		status='verified',
		state=AirbnbStateEvidence(
			captured_at='2026-09-01T12:00:00+00:00',
			screenshot_path=str(screenshot),
		),
	)
	store = AirbnbLiveRunStore()
	store.start('run-1', 'Find a stay')
	store.add_event('run-1', event)
	store.finish('run-1', status='completed', result={'status': 'completed'})

	snapshot = store.snapshot('run-1')

	assert snapshot is not None
	assert snapshot['status'] == 'completed'
	assert 'screenshots' not in snapshot
	assert str(screenshot) not in str(snapshot)
	assert store.screenshot('run-1', 1) == screenshot
	assert store.screenshot('wrong-run', 1) is None


def test_demo_payload_exposes_summary_without_raw_trace_evidence() -> None:
	parameters = AirbnbSearchParameters(
		city='Milwaukee',
		check_in=date(2026, 10, 16),
		check_out=date(2026, 10, 18),
		adults=2,
	)
	candidate = ListingCandidate(
		listing_id='3061452',
		href='/rooms/3061452?adults=2&check_in=2026-10-16&check_out=2026-10-18',
		title='Private room in a comfortable home',
		total_price=184,
		currency='$',
		position=8,
	)
	report = AirbnbHybridReport(
		run_id='test-run',
		status=WorkflowStatus.COMPLETED,
		task='Find the least expensive Airbnb in Milwaukee.',
		workflow_id=AirbnbWorkflow.CHEAPEST_LOADED_STAY,
		parameters=parameters,
		search_url='https://www.airbnb.com/s/Milwaukee/homes',
		candidates=[candidate],
		selected=AgentListingSelection(
			listing_id='3061452',
			total_price=184,
			currency='$',
			considered_count=1,
			reason='Lowest displayed total.',
		),
		listing_heading='Private room in a comfortable home',
		listing_url='https://www.airbnb.com/rooms/3061452',
		model_calls=3,
	)

	payload = airbnb_demo_report_payload(report)

	assert payload['selected']['title'] == candidate.title
	assert payload['selected']['total_price'] == 184
	assert payload['candidate_count'] == 1
	assert payload['model_calls'] == 3
	assert 'events' not in payload
	assert 'candidates' not in payload


def test_demo_routing_error_is_actionable_and_typed() -> None:
	route = AirbnbTaskRoute(
		task='Find the cheapest Airbnb in Chicago.',
		disposition=RouteDisposition.NEEDS_CLARIFICATION,
		workflow_id=AirbnbWorkflow.CHEAPEST_LOADED_STAY,
		confidence=0.45,
		reason='Check-in and check-out dates are missing.',
	)

	payload = airbnb_routing_error_payload(route)

	assert payload['status'] == 'needs_input'
	assert payload['detail'] == 'Check-in and check-out dates are missing.'
	assert payload['routing']['disposition'] == 'needs_clarification'
