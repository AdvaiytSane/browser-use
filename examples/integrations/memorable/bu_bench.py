"""Run checkpointed BU Bench V1 cold trials and deterministic warm replays.

This adapter deliberately separates execution from judging.  A judge outage can
leave ``score`` unknown, but it can never erase the already-observed browser
steps, duration, cost, capture bundle, or replay report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from browser_use import Agent, BrowserProfile, BrowserSession, ChatAnthropic, ChatBrowserUse, ChatGoogle, ChatOpenAI
from browser_use.browser.profile import ViewportSize
from examples.integrations.memorable.offline_capture import OfflineCaptureOptions, OfflineRunCapture
from examples.integrations.memorable.procedure import BrowserProcedure
from examples.integrations.memorable.replay import DeterministicReplayer, ReplayOptions, ReplayStatus

BENCHMARK_NAME = 'BU_Bench_V1'
URL_PATTERN = re.compile(r'https?://[^\s)\]>]+')


class TrialMode(str, Enum):
	COLD_AGENT = 'cold_agent'
	WARM_REPLAY = 'warm_replay'
	AGENT_FALLBACK = 'agent_fallback'


class ExecutionStatus(str, Enum):
	COMPLETED = 'completed'
	NEEDS_RECOVERY = 'needs_recovery'
	FAILED = 'failed'


class JudgeStatus(str, Enum):
	PENDING = 'pending'
	COMPLETED = 'completed'
	INFRA_ERROR = 'infra_error'


class BenchmarkTask(BaseModel):
	model_config = ConfigDict(extra='allow')

	task_id: str
	confirmed_task: str
	category: str
	answer: Any = None


class TrialMetrics(BaseModel):
	steps: int = Field(ge=0)
	duration_seconds: float = Field(ge=0)
	cost_usd: float = Field(ge=0)
	model_calls: int = Field(ge=0)


class TrialRecord(BaseModel):
	model_config = ConfigDict(extra='forbid')

	schema_version: str = '0.1.0'
	trial_id: str
	task_id: str
	task_index: int = Field(ge=0)
	task_fingerprint: str
	category: str
	mode: TrialMode
	execution_status: ExecutionStatus
	judge_status: JudgeStatus
	metrics: TrialMetrics
	score: int | None = Field(default=None, ge=0, le=1)
	judge_verdict: bool | None = None
	judge_reason: str | None = None
	execution_error: str | None = None
	capture_dir: str | None = None
	replay_report: dict[str, Any] | None = None
	created_at: str


class BenchmarkSummary(BaseModel):
	cold_trials: int
	warm_trials: int
	fallback_trials: int
	judge_infra_errors: int
	cold_scored_success_rate: float | None
	warm_scored_success_rate: float | None
	cold_total_cost_usd: float
	warm_total_cost_usd: float
	fallback_total_cost_usd: float
	cold_total_duration_seconds: float
	warm_total_duration_seconds: float
	fallback_total_duration_seconds: float


def load_task(benchmark_root: Path, task_index: int) -> BenchmarkTask:
	"""Decrypt one official task in memory, matching the public interleave order."""

	path = benchmark_root / f'{BENCHMARK_NAME}.enc'
	key = base64.urlsafe_b64encode(hashlib.sha256(BENCHMARK_NAME.encode()).digest())
	payload = Fernet(key).decrypt(base64.b64decode(path.read_text()))
	tasks = json.loads(payload)
	if len(tasks) == 100:
		tasks = [tasks[domain * 20 + item] for item in range(20) for domain in range(5)]
	if task_index < 0 or task_index >= len(tasks):
		raise IndexError(f'Task index {task_index} is outside 0..{len(tasks) - 1}')
	return BenchmarkTask.model_validate(tasks[task_index])


def task_fingerprint(task: BenchmarkTask) -> str:
	normalized = ' '.join(task.confirmed_task.split()).casefold()
	return hashlib.sha256(normalized.encode()).hexdigest()


def start_url(task: BenchmarkTask) -> str:
	match = URL_PATTERN.search(task.confirmed_task)
	if not match:
		raise ValueError('The selected task has no explicit start URL; deterministic replay is ineligible')
	return match.group(0).rstrip('.,;')


def write_checkpoint(record: TrialRecord, output_dir: Path) -> Path:
	"""Atomically persist private metrics before any judge call."""

	output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
	os.chmod(output_dir, 0o700)
	path = output_dir / f'{record.trial_id}.json'
	temporary = path.with_name(f'.{path.name}.partial')
	temporary.write_text(record.model_dump_json(indent=2) + '\n')
	os.chmod(temporary, 0o600)
	temporary.replace(path)
	return path


def summarize(records: list[TrialRecord]) -> BenchmarkSummary:
	"""Aggregate cold and warm modes separately so warm scores cannot inflate cold accuracy."""

	cold = [record for record in records if record.mode == TrialMode.COLD_AGENT]
	warm = [record for record in records if record.mode == TrialMode.WARM_REPLAY]
	fallback = [record for record in records if record.mode == TrialMode.AGENT_FALLBACK]
	return BenchmarkSummary(
		cold_trials=len(cold),
		warm_trials=len(warm),
		fallback_trials=len(fallback),
		judge_infra_errors=sum(record.judge_status == JudgeStatus.INFRA_ERROR for record in records),
		cold_scored_success_rate=_success_rate(cold),
		warm_scored_success_rate=_success_rate(warm),
		cold_total_cost_usd=sum(record.metrics.cost_usd for record in cold),
		warm_total_cost_usd=sum(record.metrics.cost_usd for record in warm),
		fallback_total_cost_usd=sum(record.metrics.cost_usd for record in fallback),
		cold_total_duration_seconds=sum(record.metrics.duration_seconds for record in cold),
		warm_total_duration_seconds=sum(record.metrics.duration_seconds for record in warm),
		fallback_total_duration_seconds=sum(record.metrics.duration_seconds for record in fallback),
	)


def _success_rate(records: list[TrialRecord]) -> float | None:
	scored = [record.score for record in records if record.score is not None]
	return sum(scored) / len(scored) if scored else None


def _trial_id(task: BenchmarkTask, mode: TrialMode) -> str:
	stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	return f'{task.task_id}-{mode.value}-{stamp}'


def _llm(provider: str, model: str):
	if provider == 'openai':
		return ChatOpenAI(model=model, api_key=os.getenv('OPENAI_API_KEY'))
	if provider == 'google':
		return ChatGoogle(model=model, api_key=os.getenv('GOOGLE_API_KEY'))
	if provider == 'anthropic':
		return ChatAnthropic(model=model, api_key=os.getenv('ANTHROPIC_API_KEY'))
	if provider == 'browser-use':
		return ChatBrowserUse(model=model)
	raise ValueError(f'Unsupported provider: {provider}')


def _judge_module(benchmark_root: Path) -> ModuleType:
	path = benchmark_root / 'judge.py'
	spec = importlib.util.spec_from_file_location('memorable_bu_bench_official_judge', path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f'Cannot load official judge from {path}')
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


async def _judge(
	benchmark_root: Path,
	task: BenchmarkTask,
	final_result: str,
	steps: list[str],
	screenshots: list[str],
	judge_model: str,
) -> tuple[int, bool, str]:
	module = _judge_module(benchmark_root)
	messages = module.construct_judge_messages(
		task=task.confirmed_task,
		final_result=final_result,
		agent_steps=steps,
		ground_truth=task.answer,
		screenshots_b64=screenshots,
	)
	response = await ChatGoogle(model=judge_model, api_key=os.getenv('GOOGLE_API_KEY')).ainvoke(
		messages, output_format=module.JudgementResult
	)
	completion = response.completion
	if not isinstance(completion, BaseModel):
		raise TypeError('The official judge did not return its structured result model')
	judgement = completion.model_dump()
	verdict = bool(judgement.get('verdict'))
	return int(verdict), verdict, str(judgement.get('reasoning') or '')


async def run_cold_trial(
	args: argparse.Namespace,
	task: BenchmarkTask,
	repeat: int,
	*,
	mode: TrialMode = TrialMode.COLD_AGENT,
	replay_report: dict[str, Any] | None = None,
) -> TrialRecord:
	"""Run and capture one agent trial, checkpoint it, then ask the official judge."""

	trial_id = _trial_id(task, mode)
	trial_dir = args.output_dir / trial_id
	capture_root = args.output_dir / 'captures' / task_fingerprint(task)
	agent = Agent(
		task=task.confirmed_task,
		llm=_llm(args.provider, args.model),
		initial_actions=[{'navigate': {'url': start_url(task), 'new_tab': False}}],
		directly_open_url=False,
		max_actions_per_step=1,
		use_judge=False,
		calculate_cost=True,
		browser_profile=BrowserProfile(
			headless=args.headless,
			user_data_dir=None,
			window_size=ViewportSize(width=args.viewport_width, height=args.viewport_height),
		),
	)
	capture = OfflineRunCapture(
		OfflineCaptureOptions(
			output_dir=capture_root,
			include_screenshots=True,
			include_full_dom=True,
			include_candidates=True,
			include_rendered_page=True,
			run_label=f'bu-bench-{mode.value}-task-{args.task_index}-repeat-{repeat}',
		)
	)
	started = time.monotonic()
	try:
		history = await capture.run_agent(agent, max_steps=args.max_steps)
		record = TrialRecord(
			trial_id=trial_id,
			task_id=task.task_id,
			task_index=args.task_index,
			task_fingerprint=task_fingerprint(task),
			category=task.category,
			mode=mode,
			execution_status=ExecutionStatus.COMPLETED,
			judge_status=JudgeStatus.PENDING,
			metrics=TrialMetrics(
				steps=history.number_of_steps(),
				duration_seconds=history.total_duration_seconds(),
				cost_usd=history.usage.total_cost if history.usage else 0,
				model_calls=history.number_of_steps(),
			),
			capture_dir=str(capture.final_dir) if capture.final_dir.exists() else None,
			replay_report=replay_report,
			created_at=datetime.now(timezone.utc).isoformat(),
		)
		write_checkpoint(record, trial_dir)
		try:
			score, verdict, reason = await _judge(
				args.benchmark_root,
				task,
				history.final_result() or 'Agent did not return a result',
				history.agent_steps(),
				[screenshot for screenshot in history.screenshots() if screenshot],
				args.judge_model,
			)
			record = record.model_copy(
				update={'judge_status': JudgeStatus.COMPLETED, 'score': score, 'judge_verdict': verdict, 'judge_reason': reason}
			)
		except Exception as exc:
			record = record.model_copy(
				update={'judge_status': JudgeStatus.INFRA_ERROR, 'judge_reason': f'{type(exc).__name__}: {exc}'}
			)
		write_checkpoint(record, trial_dir)
		return record
	except Exception as exc:
		record = TrialRecord(
			trial_id=trial_id,
			task_id=task.task_id,
			task_index=args.task_index,
			task_fingerprint=task_fingerprint(task),
			category=task.category,
			mode=mode,
			execution_status=ExecutionStatus.FAILED,
			judge_status=JudgeStatus.PENDING,
			metrics=TrialMetrics(steps=0, duration_seconds=time.monotonic() - started, cost_usd=0, model_calls=0),
			execution_error=f'{type(exc).__name__}: {exc}',
			capture_dir=str(capture.final_dir) if capture.final_dir.exists() else None,
			replay_report=replay_report,
			created_at=datetime.now(timezone.utc).isoformat(),
		)
		write_checkpoint(record, trial_dir)
		return record


async def run_warm_trial(args: argparse.Namespace, task: BenchmarkTask) -> TrialRecord:
	"""Execute a compiled procedure and preserve a zero-model replay record."""

	if args.procedure is None:
		raise ValueError('--procedure is required for --mode warm')
	procedure = BrowserProcedure.read(args.procedure)
	if procedure.task_fingerprint != task_fingerprint(task):
		raise ValueError('Procedure fingerprint does not match the selected benchmark task')
	parameters = _parse_parameters(args.parameter)
	trial_id = _trial_id(task, TrialMode.WARM_REPLAY)
	trial_dir = args.output_dir / trial_id
	profile = BrowserProfile(
		headless=args.headless,
		user_data_dir=None,
		window_size=ViewportSize(width=args.viewport_width, height=args.viewport_height),
	)
	browser = BrowserSession(browser_profile=profile)
	started = time.monotonic()
	await browser.start()
	try:
		await browser.navigate_to(start_url(task))
		report = await DeterministicReplayer(procedure, ReplayOptions(output_dir=args.output_dir / 'replays')).run(
			browser, parameters
		)
		state = await browser.get_browser_state_summary(include_screenshot=True)
		steps = [event.model_dump_json() for event in report.events]
		status = ExecutionStatus.COMPLETED if report.status == ReplayStatus.COMPLETED else ExecutionStatus.NEEDS_RECOVERY
		record = TrialRecord(
			trial_id=trial_id,
			task_id=task.task_id,
			task_index=args.task_index,
			task_fingerprint=task_fingerprint(task),
			category=task.category,
			mode=TrialMode.WARM_REPLAY,
			execution_status=status,
			judge_status=JudgeStatus.PENDING,
			metrics=TrialMetrics(
				steps=len(report.events),
				duration_seconds=time.monotonic() - started,
				cost_usd=0,
				model_calls=report.model_calls,
			),
			replay_report=report.model_dump(mode='json'),
			created_at=datetime.now(timezone.utc).isoformat(),
		)
		write_checkpoint(record, trial_dir)
		if status == ExecutionStatus.COMPLETED:
			try:
				score, verdict, reason = await _judge(
					args.benchmark_root,
					task,
					procedure.expected_success_text or 'Deterministic replay completed with verified postconditions.',
					steps,
					[state.screenshot] if state.screenshot else [],
					args.judge_model,
				)
				record = record.model_copy(
					update={
						'judge_status': JudgeStatus.COMPLETED,
						'score': score,
						'judge_verdict': verdict,
						'judge_reason': reason,
					}
				)
			except Exception as exc:
				record = record.model_copy(
					update={'judge_status': JudgeStatus.INFRA_ERROR, 'judge_reason': f'{type(exc).__name__}: {exc}'}
				)
		write_checkpoint(record, trial_dir)
		return record
	finally:
		await browser.kill()


def _parse_parameters(items: list[str]) -> dict[str, str]:
	parameters: dict[str, str] = {}
	for item in items:
		if '=' not in item:
			raise ValueError(f'Parameter must use KEY=VALUE syntax: {item!r}')
		key, value = item.split('=', 1)
		if not key or key in parameters:
			raise ValueError(f'Invalid or duplicate parameter: {key!r}')
		parameters[key] = value
	return parameters


async def _main_async(args: argparse.Namespace) -> int:
	if args.env_file:
		load_dotenv(args.env_file, override=False)
	task = load_task(args.benchmark_root, args.task_index)
	if task.category != 'InteractionTests' and not args.allow_non_interaction:
		raise ValueError('Only InteractionTests are replay-eligible by default')
	records: list[TrialRecord] = []
	if args.mode == 'cold':
		for repeat in range(1, args.repeat + 1):
			records.append(await run_cold_trial(args, task, repeat))
	else:
		replay_record = await run_warm_trial(args, task)
		records.append(replay_record)
		if replay_record.execution_status != ExecutionStatus.COMPLETED and args.fallback_to_agent:
			# run_warm_trial has already killed its browser.  Recovery always starts
			# from a fresh session rather than inheriting partial replay state.
			records.append(
				await run_cold_trial(
					args,
					task,
					1,
					mode=TrialMode.AGENT_FALLBACK,
					replay_report=replay_record.replay_report,
				)
			)
	summary = summarize(records)
	print(summary.model_dump_json(indent=2))
	return 0 if all(record.execution_status == ExecutionStatus.COMPLETED for record in records) else 2


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--benchmark-root', type=Path, required=True)
	parser.add_argument('--task-index', type=int, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--mode', choices=['cold', 'warm'], default='cold')
	parser.add_argument('--provider', choices=['openai', 'google', 'anthropic', 'browser-use'], default='openai')
	parser.add_argument('--model', default='gpt-4.1')
	parser.add_argument('--judge-model', default='gemini-2.5-flash')
	parser.add_argument('--env-file', type=Path)
	parser.add_argument('--repeat', type=int, default=1)
	parser.add_argument('--max-steps', type=int, default=30)
	parser.add_argument('--procedure', type=Path)
	parser.add_argument('--fallback-to-agent', action='store_true')
	parser.add_argument('-p', '--parameter', action='append', default=[])
	parser.add_argument('--viewport-width', type=int, default=1280)
	parser.add_argument('--viewport-height', type=int, default=800)
	parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument('--allow-non-interaction', action='store_true')
	args = parser.parse_args()
	raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == '__main__':
	main()
