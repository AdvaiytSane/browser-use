from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from examples.integrations.memorable.airbnb_hybrid import (
	AgentListingSelection,
	AirbnbSearchParameters,
	AirbnbTaskRouteDraft,
	AirbnbWorkflow,
	ExecutionMode,
	HybridEvent,
	ListingCandidate,
	RouteDisposition,
	TaskRoutingError,
	TaskRoutingEvidence,
	airbnb_destination_slug,
	build_agent_bridge_tools,
	build_airbnb_search_url,
	candidate_href_matches_parameters,
	canonicalize_agent_selection,
	cheapest_candidate,
	choose_execution_mode,
	listing_handoff_contract,
	parse_rating_reviews,
	parse_total_price,
	publish_live_event,
	route_airbnb_task,
	run_airbnb_task,
	search_results_contract,
	select_candidate,
	validate_task_route,
)


def _parameters(city: str = 'Detroit, MI') -> AirbnbSearchParameters:
	return AirbnbSearchParameters(
		city=city,
		check_in=date(2026, 10, 16),
		check_out=date(2026, 10, 18),
		adults=2,
	)


def _candidate(
	listing_id: str,
	total_price: float,
	position: int,
	*,
	currency: str = '$',
	rating: float | None = None,
	review_count: int | None = None,
) -> ListingCandidate:
	return ListingCandidate(
		listing_id=listing_id,
		href=f'/rooms/{listing_id}?check_in=2026-10-16&check_out=2026-10-18&adults=2',
		title=f'Listing {listing_id}',
		total_price=total_price,
		currency=currency,
		position=position,
		rating=rating,
		review_count=review_count,
	)


def _listing_url(listing_id: str) -> str:
	return f'https://www.airbnb.com/rooms/{listing_id}?adults=2&check_in=2026-10-16&check_out=2026-10-18'


def test_parameterized_search_url_does_not_require_captured_place_ids() -> None:
	url = build_airbnb_search_url(_parameters('Pittsburgh, PA'))

	assert '/s/Pittsburgh--PA/homes?' in url
	assert 'checkin=2026-10-16' in url
	assert 'checkout=2026-10-18' in url
	assert 'adults=2' in url
	assert 'place_id' not in url
	assert airbnb_destination_slug('New York, NY') == 'New-York--NY'


def test_parameters_refuse_paths_and_invalid_dates() -> None:
	with pytest.raises(ValidationError, match='place label'):
		_parameters('https://example.com')
	with pytest.raises(ValidationError, match='after check_in'):
		AirbnbSearchParameters(
			city='Detroit, MI',
			check_in=date(2026, 10, 18),
			check_out=date(2026, 10, 16),
		)


@pytest.mark.parametrize(
	('text', 'expected'),
	[
		('$753\n$522\nShow price breakdown\nfor 2 nights\n4.85 (34)', (522.0, '$')),
		('£1,104 for 4 nights', (1104.0, '£')),
		('€98.50\nfor 1 night', (98.5, '€')),
		('No price shown', None),
	],
)
def test_parse_total_price_uses_displayed_trip_total(text: str, expected: tuple[float, str] | None) -> None:
	assert parse_total_price(text) == expected


@pytest.mark.parametrize(
	('text', 'expected'),
	[
		('5.0 out of 5 average rating, 24 reviews\n5.0 (24)', (5.0, 24)),
		('4.92 out of 5 average rating, 1,204 reviews', (4.92, 1204)),
		('New place to stay\nNew', None),
	],
)
def test_parse_rating_reviews_requires_explicit_airbnb_evidence(
	text: str,
	expected: tuple[float, int] | None,
) -> None:
	assert parse_rating_reviews(text) == expected


def test_results_contract_requires_url_parameters_heading_and_candidates() -> None:
	parameters = _parameters()
	url = build_airbnb_search_url(parameters)

	passed = search_results_contract(url, ['Search results; Over 1,000 homes in Detroit'], 18, parameters)
	visible_heading_only = search_results_contract(url, ['Over 1,000 homes in Detroit'], 18, parameters)
	wrong_city = search_results_contract(url, ['Search results; homes in Chicago'], 18, parameters)
	missing_cards = search_results_contract(url, ['Search results; homes in Detroit'], 0, parameters)

	assert passed.passed is True
	assert visible_heading_only.passed is True
	assert wrong_city.passed is False
	assert 'requested city' in wrong_city.violations[0]
	assert missing_cards.passed is False


def test_switches_only_after_state_contract_and_only_when_comparison_is_needed() -> None:
	parameters = _parameters()
	url = build_airbnb_search_url(parameters)
	passed = search_results_contract(url, ['Search results; homes in Detroit'], 2, parameters)
	failed = search_results_contract('https://example.com', [], 0, parameters)
	one = [_candidate('1', 300, 1)]
	two = [*one, _candidate('2', 200, 2)]

	assert choose_execution_mode(failed, two).mode == ExecutionMode.REFUSE
	assert choose_execution_mode(passed, one).mode == ExecutionMode.DETERMINISTIC
	assert choose_execution_mode(passed, two).mode == ExecutionMode.AGENT
	assert choose_execution_mode(passed, two, AirbnbWorkflow.FIRST_VISIBLE_STAY).mode == ExecutionMode.DETERMINISTIC
	assert choose_execution_mode(passed, two, AirbnbWorkflow.HIGHEST_RATED_LOADED_STAY).mode == ExecutionMode.REFUSE


def test_cheapest_candidate_has_a_reproducible_tie_breaker() -> None:
	candidates = [
		_candidate('3', 250, 3),
		_candidate('2', 200, 2),
		_candidate('1', 200, 1),
	]

	assert cheapest_candidate(candidates).listing_id == '1'


def test_shared_candidate_scope_supports_four_distinct_selection_policies() -> None:
	candidates = [
		_candidate('1', 300, 1, rating=4.9, review_count=100),
		_candidate('2', 200, 2, rating=5.0, review_count=3),
		_candidate('3', 400, 3, rating=5.0, review_count=24),
	]

	assert select_candidate(candidates, AirbnbWorkflow.CHEAPEST_LOADED_STAY).listing_id == '2'
	assert select_candidate(candidates, AirbnbWorkflow.HIGHEST_RATED_LOADED_STAY).listing_id == '3'
	assert select_candidate(candidates, AirbnbWorkflow.MOST_REVIEWED_LOADED_STAY).listing_id == '1'
	assert select_candidate(candidates, AirbnbWorkflow.FIRST_VISIBLE_STAY).listing_id == '1'


def test_agent_choice_is_canonicalized_from_frozen_dom_facts() -> None:
	candidates = [
		_candidate('1', 300, 1, rating=4.9, review_count=100),
		_candidate('2', 200, 2, rating=5.0, review_count=24),
	]
	raw = AgentListingSelection(
		listing_id='2',
		total_price=999,
		currency='€',
		rating=1,
		review_count=1,
		considered_count=999,
		reason='Selected listing 2.',
	)

	selection, reason = canonicalize_agent_selection(raw, ['2'], candidates)

	assert reason is None
	assert selection is not None
	assert selection.total_price == 200
	assert selection.currency == '$'
	assert selection.rating == 5.0
	assert selection.review_count == 24
	assert selection.considered_count == 2


def test_handoff_accepts_only_opened_verified_minimum() -> None:
	candidates = [_candidate('1', 300, 1), _candidate('2', 200, 2)]
	selection = AgentListingSelection(
		listing_id='2',
		total_price=200,
		currency='$',
		considered_count=2,
		reason='Lowest displayed total.',
	)

	contract = listing_handoff_contract(
		_listing_url('2'),
		['Cheapest stay'],
		candidates,
		selection,
		_parameters(),
	)

	assert contract.passed is True
	assert contract.evidence['expected_listing_id'] == '2'


def test_handoff_verifies_the_highest_rated_policy_and_metric_evidence() -> None:
	candidates = [
		_candidate('1', 200, 1, rating=4.9, review_count=500),
		_candidate('2', 300, 2, rating=5.0, review_count=24),
	]
	selection = AgentListingSelection(
		listing_id='2',
		total_price=300,
		currency='$',
		rating=5.0,
		review_count=24,
		considered_count=2,
		reason='Highest rating.',
	)

	contract = listing_handoff_contract(
		_listing_url('2'),
		['Highest-rated stay'],
		candidates,
		selection,
		_parameters(),
		AirbnbWorkflow.HIGHEST_RATED_LOADED_STAY,
	)

	assert contract.passed is True
	assert contract.evidence['expected_rating'] == 5.0


@pytest.mark.parametrize(
	('url', 'selected_id', 'price', 'count', 'violation'),
	[
		(_listing_url('1'), '1', 300, 2, 'does not satisfy'),
		(_listing_url('2'), '1', 300, 2, 'does not match the opened'),
		(_listing_url('2'), '2', 250, 2, 'price does not match'),
		(_listing_url('2'), '2', 200, 1, 'candidate count does not match'),
	],
)
def test_handoff_refuses_wrong_or_unverifiable_agent_output(
	url: str,
	selected_id: str,
	price: float,
	count: int,
	violation: str,
) -> None:
	candidates = [_candidate('1', 300, 1), _candidate('2', 200, 2)]
	selection = AgentListingSelection(
		listing_id=selected_id,
		total_price=price,
		currency='$',
		considered_count=count,
		reason='Agent choice.',
	)

	contract = listing_handoff_contract(url, ['Listing heading'], candidates, selection, _parameters())

	assert contract.passed is False
	assert any(violation in item for item in contract.violations)


def test_agent_bridge_exposes_only_scoped_open_and_structured_done() -> None:
	tools = build_agent_bridge_tools([_candidate('1', 300, 1), _candidate('2', 200, 2)], [])

	assert set(tools.registry.registry.actions) == {'done', 'open_scoped_listing'}


def test_live_event_observer_is_optional_and_fail_open() -> None:
	event = HybridEvent(node='search_results', mode=ExecutionMode.DETERMINISTIC, status='verified')
	received: list[HybridEvent] = []

	publish_live_event(received.append, event)
	publish_live_event(lambda _: (_ for _ in ()).throw(RuntimeError('observer down')), event)
	publish_live_event(None, event)

	assert received == [event]


def test_candidate_scope_refuses_airbnb_links_with_substituted_dates_or_guests() -> None:
	parameters = _parameters()
	matching = '/rooms/1?adults=2&check_in=2026-10-16&check_out=2026-10-18'
	wrong_dates = '/rooms/1?adults=2&check_in=2026-10-15&check_out=2026-10-18'
	wrong_guests = '/rooms/1?adults=1&check_in=2026-10-16&check_out=2026-10-18'

	assert candidate_href_matches_parameters(matching, parameters) is True
	assert candidate_href_matches_parameters(wrong_dates, parameters) is False
	assert candidate_href_matches_parameters(wrong_guests, parameters) is False


def _matched_draft(
	*,
	city: str = 'Chicago',
	objective_quote: str = 'least expensive',
	city_quote: str = 'Chicago',
	check_in_quote: str = 'October 16',
	check_out_quote: str = 'October 18, 2026',
	adults: int | None = 2,
	guests_quote: str | None = 'two adults',
	workflow: AirbnbWorkflow = AirbnbWorkflow.CHEAPEST_LOADED_STAY,
) -> AirbnbTaskRouteDraft:
	return AirbnbTaskRouteDraft(
		disposition=RouteDisposition.MATCHED,
		workflow_id=workflow,
		city=city,
		check_in=date(2026, 10, 16),
		check_out=date(2026, 10, 18),
		adults=adults,
		confidence=0.97,
		evidence=TaskRoutingEvidence(
			objective_quote=objective_quote,
			city_quote=city_quote,
			check_in_quote=check_in_quote,
			check_out_quote=check_out_quote,
			guests_quote=guests_quote,
		),
		reason='The request asks for the lowest-price stay with complete search parameters.',
	)


@pytest.mark.parametrize(
	'task',
	[
		'Find me the least expensive Airbnb in Chicago for two adults from October 16 to October 18, 2026.',
		'I need the least expensive place to stay in Chicago, October 16 through October 18, 2026, for two adults.',
		'Show the least expensive rental in Chicago for two adults: October 16 to October 18, 2026.',
	],
)
def test_validated_router_accepts_reasonable_paraphrases(task: str) -> None:
	route = validate_task_route(task, _matched_draft())

	assert route.disposition == RouteDisposition.MATCHED
	assert route.workflow_id == AirbnbWorkflow.CHEAPEST_LOADED_STAY
	assert route.parameters == _parameters('Chicago')


@pytest.mark.parametrize(
	('objective', 'workflow'),
	[
		('highest rated', AirbnbWorkflow.HIGHEST_RATED_LOADED_STAY),
		('most reviewed', AirbnbWorkflow.MOST_REVIEWED_LOADED_STAY),
		('first visible', AirbnbWorkflow.FIRST_VISIBLE_STAY),
	],
)
def test_validated_router_accepts_each_selection_branch(objective: str, workflow: AirbnbWorkflow) -> None:
	task = f'Open the {objective} Airbnb in Chicago for two adults from October 16 to October 18, 2026.'
	draft = _matched_draft(objective_quote=objective, workflow=workflow)

	route = validate_task_route(task, draft)

	assert route.disposition == RouteDisposition.MATCHED
	assert route.workflow_id == workflow


def test_router_defaults_only_omitted_guest_count_and_records_it() -> None:
	task = 'Find the cheapest stay in Chicago from October 16 to October 18, 2026.'
	draft = _matched_draft(objective_quote='cheapest', adults=None, guests_quote=None)

	route = validate_task_route(task, draft)

	assert route.parameters is not None
	assert route.parameters.adults == 2
	assert route.assumptions == ['Guest count was omitted; defaulted to 2 adults.']


def test_router_refuses_fabricated_evidence_even_when_parameters_look_valid() -> None:
	task = 'Find the cheapest stay in Chicago from October 16 to October 18, 2026.'
	draft = _matched_draft(objective_quote='best bargain', adults=None, guests_quote=None)

	route = validate_task_route(task, draft)

	assert route.disposition == RouteDisposition.NEEDS_CLARIFICATION
	assert route.parameters is None
	assert 'objective' in route.reason


def test_router_refuses_an_invalid_date_order_before_execution() -> None:
	task = 'Find the cheapest stay in Chicago from October 18 to October 16, 2026.'
	draft = _matched_draft(
		objective_quote='cheapest',
		check_in_quote='October 18',
		check_out_quote='October 16, 2026',
		adults=None,
		guests_quote=None,
	)
	draft.check_in = date(2026, 10, 18)
	draft.check_out = date(2026, 10, 16)

	route = validate_task_route(task, draft)

	assert route.disposition == RouteDisposition.NEEDS_CLARIFICATION
	assert 'failed validation' in route.reason


class _RoutingLLM:
	def __init__(self, draft: AirbnbTaskRouteDraft):
		self.draft = draft
		self.calls = 0

	async def ainvoke(self, messages, output_format=None, **kwargs):
		self.calls += 1
		assert output_format is AirbnbTaskRouteDraft
		return SimpleNamespace(
			completion=self.draft,
			usage=SimpleNamespace(total_tokens=321),
		)


@pytest.mark.asyncio
async def test_model_router_uses_one_call_and_preserves_usage() -> None:
	task = 'Find the least expensive Airbnb in Chicago for two adults from October 16 to October 18, 2026.'
	llm = _RoutingLLM(_matched_draft())

	route = await route_airbnb_task(task, llm=llm, today=date(2026, 9, 1))

	assert llm.calls == 1
	assert route.router_model_calls == 1
	assert route.router_tokens == 321


@pytest.mark.asyncio
async def test_unsupported_task_stops_before_a_browser_is_created() -> None:
	draft = AirbnbTaskRouteDraft(
		disposition=RouteDisposition.UNSUPPORTED,
		confidence=0.99,
		reason='The catalog does not support highest-rated selection.',
	)
	llm = _RoutingLLM(draft)

	with pytest.raises(TaskRoutingError) as exc_info:
		await run_airbnb_task(
			'Find the highest-rated cabin in Chicago for October 16 to October 18, 2026.',
			llm=llm,
		)

	assert exc_info.value.route.disposition == RouteDisposition.UNSUPPORTED
	assert llm.calls == 1
