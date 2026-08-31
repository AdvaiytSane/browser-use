import base64
import hashlib
import json

from cryptography.fernet import Fernet

from examples.integrations.memorable.bu_bench import (
	BENCHMARK_NAME,
	BenchmarkTask,
	ExecutionStatus,
	JudgeStatus,
	TrialMetrics,
	TrialMode,
	TrialRecord,
	load_task,
	start_url,
	summarize,
	task_fingerprint,
	write_checkpoint,
)


def _record(mode: TrialMode, *, score: int | None, judge: JudgeStatus = JudgeStatus.COMPLETED) -> TrialRecord:
	is_agent = mode != TrialMode.WARM_REPLAY
	return TrialRecord(
		trial_id=f'trial-{mode.value}-{score}',
		task_id='task-1',
		task_index=2,
		task_fingerprint='a' * 64,
		category='InteractionTests',
		mode=mode,
		execution_status=ExecutionStatus.COMPLETED,
		judge_status=judge,
		metrics=TrialMetrics(
			steps=5 if is_agent else 3,
			duration_seconds=10 if is_agent else 1,
			cost_usd=2 if is_agent else 0,
			model_calls=5 if is_agent else 0,
		),
		score=score,
		created_at='2026-08-31T00:00:00+00:00',
	)


def test_summary_keeps_cold_and_warm_scores_separate():
	summary = summarize(
		[
			_record(TrialMode.COLD_AGENT, score=0),
			_record(TrialMode.COLD_AGENT, score=None, judge=JudgeStatus.INFRA_ERROR),
			_record(TrialMode.WARM_REPLAY, score=1),
			_record(TrialMode.AGENT_FALLBACK, score=1),
		]
	)

	assert summary.cold_trials == 2
	assert summary.warm_trials == 1
	assert summary.fallback_trials == 1
	assert summary.judge_infra_errors == 1
	assert summary.cold_scored_success_rate == 0
	assert summary.warm_scored_success_rate == 1
	assert summary.cold_total_cost_usd == 4
	assert summary.warm_total_cost_usd == 0
	assert summary.fallback_total_cost_usd == 2
	assert summary.fallback_total_duration_seconds == 10


def test_checkpoint_survives_pending_judge_and_is_private(tmp_path):
	record = _record(TrialMode.COLD_AGENT, score=None, judge=JudgeStatus.PENDING)
	path = write_checkpoint(record, tmp_path / 'private')

	assert TrialRecord.model_validate_json(path.read_text()) == record
	assert path.stat().st_mode & 0o777 == 0o600
	assert path.parent.stat().st_mode & 0o777 == 0o700


def test_load_task_matches_official_interleave_without_writing_plaintext(tmp_path):
	tasks = [
		{
			'task_id': f'task-{index}',
			'confirmed_task': f'Open https://example.test/{index}',
			'category': 'InteractionTests',
			'answer': None,
		}
		for index in range(100)
	]
	key = base64.urlsafe_b64encode(hashlib.sha256(BENCHMARK_NAME.encode()).digest())
	encrypted = Fernet(key).encrypt(json.dumps(tasks).encode())
	(tmp_path / f'{BENCHMARK_NAME}.enc').write_text(base64.b64encode(encrypted).decode())

	assert load_task(tmp_path, 0).task_id == 'task-0'
	assert load_task(tmp_path, 1).task_id == 'task-20'
	assert load_task(tmp_path, 5).task_id == 'task-1'
	assert not list(tmp_path.glob('*.json'))


def test_task_identity_and_start_url_are_stable():
	task = BenchmarkTask(
		task_id='task-1',
		confirmed_task='  Open  https://example.test/form  and submit it. ',
		category='InteractionTests',
	)
	equivalent = task.model_copy(update={'confirmed_task': 'open https://example.test/form and submit it.'})

	assert task_fingerprint(task) == task_fingerprint(equivalent)
	assert start_url(task) == 'https://example.test/form'
