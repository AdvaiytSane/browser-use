from examples.integrations.memorable.mutation_benchmark import (
	BrowserAudit,
	MutationClass,
	MutationTrial,
	TrialOutcome,
	build_mutation_plan,
	classify_trial,
	summarize_trials,
)
from examples.integrations.memorable.replay import ReplayStatus, accessible_name_drift_fields


def test_mutation_plans_are_seeded_and_class_specific():
	first = build_mutation_plan(MutationClass.LAYOUT, 42)
	second = build_mutation_plan(MutationClass.LAYOUT, 42)
	ambiguity = build_mutation_plan(MutationClass.AMBIGUITY, 42)

	assert first == second
	assert first.randomize_ids and first.reorder_blocks and first.wrap_controls
	assert ambiguity.duplicate_submit and not ambiguity.wrap_controls


def test_accessible_name_drift_requires_stable_dom_identity():
	assert accessible_name_drift_fields(
		{
			'node_name': 'input',
			'ax_role': 'textbox',
			'ax_name': 'old label',
			'attributes.name': 'email',
			'attributes.type': 'email',
		}
	) == {
		'node_name': 'input',
		'ax_role': 'textbox',
		'attributes.name': 'email',
		'attributes.type': 'email',
	}
	assert accessible_name_drift_fields({'node_name': 'button', 'ax_role': 'button', 'ax_name': 'unanchored button'}) is None
	assert accessible_name_drift_fields(
		{'node_name': 'button', 'ax_role': 'button', 'ax_name': 'submit', 'attributes.type': 'submit'}
	) == {'node_name': 'button', 'ax_role': 'button', 'attributes.type': 'submit'}


def test_classification_separates_completion_refusal_and_wrong_interaction():
	complete = BrowserAudit(success_visible=True, submission_count=1, interaction_count=6)
	refused = BrowserAudit(success_visible=False, submission_count=0, interaction_count=4)
	wrong = BrowserAudit(
		success_visible=False,
		submission_count=0,
		interaction_count=1,
		unexpected_interactions=[{'type': 'click', 'tag': 'DIV'}],
	)

	assert classify_trial(ReplayStatus.COMPLETED, complete) == TrialOutcome.VERIFIED_COMPLETION
	assert classify_trial(ReplayStatus.NEEDS_RECOVERY, refused) == TrialOutcome.SAFE_REFUSAL
	assert classify_trial(ReplayStatus.NEEDS_RECOVERY, wrong) == TrialOutcome.INCORRECT_INTERACTION


def test_summary_keeps_unsafe_outcomes_visible():
	def trial(outcome: TrialOutcome, seed: int) -> MutationTrial:
		return MutationTrial(
			trial_id=str(seed),
			procedure_id='procedure',
			mutation=build_mutation_plan(MutationClass.LAYOUT, seed),
			outcome=outcome,
			replay_status=ReplayStatus.COMPLETED,
			model_calls=0,
			actions_attempted=6,
			duration_seconds=1,
			success_visible=outcome == TrialOutcome.VERIFIED_COMPLETION,
			submission_count=int(outcome == TrialOutcome.VERIFIED_COMPLETION),
			interaction_count=6,
			incorrect_interaction_count=int(outcome == TrialOutcome.INCORRECT_INTERACTION),
			created_at='2026-09-01T00:00:00+00:00',
		)

	summary = summarize_trials(
		[
			trial(TrialOutcome.VERIFIED_COMPLETION, 1),
			trial(TrialOutcome.SAFE_REFUSAL, 2),
			trial(TrialOutcome.INCORRECT_INTERACTION, 3),
		],
		['1.json', '2.json', '3.json'],
	)

	assert summary.trials == 3
	assert summary.verified_completion.count == 1
	assert summary.safe_refusal.count == 1
	assert summary.incorrect_interaction.count == 1
	assert summary.incorrect_interaction.wilson_high > summary.incorrect_interaction.rate
	assert summary.model_calls == 0
