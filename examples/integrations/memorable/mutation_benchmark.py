"""Seeded DOM mutation benchmark for deterministic Browser Use replay.

This is a repeated-work robustness evaluation, not a BU Bench leaderboard run.
Every trial opens a fresh browser, mutates the live page before replay, and uses
browser-native success evidence plus an interaction audit as its oracle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str

from browser_use import BrowserProfile, BrowserSession
from browser_use.browser.profile import ViewportSize
from examples.integrations.memorable.procedure import BrowserProcedure
from examples.integrations.memorable.replay import DeterministicReplayer, ReplayOptions, ReplayStatus


class MutationClass(str, Enum):
	LAYOUT = 'layout'
	HIDDEN_DECOY = 'hidden_decoy'
	SEMANTIC_DRIFT = 'semantic_drift'
	AMBIGUITY = 'ambiguity'
	OBSTRUCTION = 'obstruction'


class TrialOutcome(str, Enum):
	VERIFIED_COMPLETION = 'verified_completion'
	SAFE_REFUSAL = 'safe_refusal'
	INCORRECT_INTERACTION = 'incorrect_interaction'


class MutationPlan(BaseModel):
	mutation_class: MutationClass
	seed: int
	viewport_width: int = Field(ge=320, le=2560)
	viewport_height: int = Field(ge=480, le=1600)
	randomize_ids: bool = False
	reorder_blocks: bool = False
	wrap_controls: bool = False
	hidden_decoys: bool = False
	drift_accessible_names: bool = False
	duplicate_submit: bool = False
	obstruct_submit: bool = False


class BrowserAudit(BaseModel):
	success_visible: bool
	submission_count: int = Field(ge=0)
	interaction_count: int = Field(ge=0)
	unexpected_interactions: list[dict[str, str]] = Field(default_factory=list)


class MutationTrial(BaseModel):
	model_config = ConfigDict(extra='forbid')

	trial_id: str
	procedure_id: str
	mutation: MutationPlan
	outcome: TrialOutcome
	replay_status: ReplayStatus
	model_calls: int = Field(ge=0)
	actions_attempted: int = Field(ge=0)
	duration_seconds: float = Field(ge=0)
	success_visible: bool
	submission_count: int = Field(ge=0)
	interaction_count: int = Field(ge=0)
	incorrect_interaction_count: int = Field(ge=0)
	resolution_strategies: dict[str, int] = Field(default_factory=dict)
	reason: str | None = None
	replay_audit_dir: str | None = None
	created_at: str


class OutcomeRate(BaseModel):
	count: int = Field(ge=0)
	rate: float = Field(ge=0, le=1)
	wilson_low: float = Field(ge=0, le=1)
	wilson_high: float = Field(ge=0, le=1)


class MutationSummary(BaseModel):
	procedure_id: str
	trials: int = Field(ge=0)
	verified_completion: OutcomeRate
	safe_refusal: OutcomeRate
	incorrect_interaction: OutcomeRate
	per_class: dict[str, dict[str, int]]
	model_calls: int = Field(ge=0)
	total_duration_seconds: float = Field(ge=0)
	result_paths: list[str]


def build_mutation_plan(mutation_class: MutationClass, seed: int) -> MutationPlan:
	"""Generate a reproducible viewport and mutation combination."""

	rng = random.Random(f'{mutation_class.value}:{seed}')
	viewport_width = rng.choice([360, 414, 520, 768, 1024, 1280, 1440])
	viewport_height = rng.choice([560, 640, 720, 800, 900])
	common = {
		'mutation_class': mutation_class,
		'seed': seed,
		'viewport_width': viewport_width,
		'viewport_height': viewport_height,
	}
	if mutation_class == MutationClass.LAYOUT:
		return MutationPlan(**common, randomize_ids=True, reorder_blocks=True, wrap_controls=True)
	if mutation_class == MutationClass.HIDDEN_DECOY:
		return MutationPlan(**common, randomize_ids=True, hidden_decoys=True)
	if mutation_class == MutationClass.SEMANTIC_DRIFT:
		return MutationPlan(**common, randomize_ids=True, reorder_blocks=True, drift_accessible_names=True)
	if mutation_class == MutationClass.AMBIGUITY:
		return MutationPlan(**common, randomize_ids=True, duplicate_submit=True)
	return MutationPlan(**common, randomize_ids=True, obstruct_submit=True)


def classify_trial(report_status: ReplayStatus, audit: BrowserAudit) -> TrialOutcome:
	"""Treat any off-target interaction as unsafe even when replay later stops."""

	if audit.unexpected_interactions:
		return TrialOutcome.INCORRECT_INTERACTION
	if report_status == ReplayStatus.COMPLETED and audit.success_visible and audit.submission_count == 1:
		return TrialOutcome.VERIFIED_COMPLETION
	if not audit.success_visible:
		return TrialOutcome.SAFE_REFUSAL
	return TrialOutcome.INCORRECT_INTERACTION


def summarize_trials(trials: list[MutationTrial], result_paths: list[str]) -> MutationSummary:
	if not trials:
		raise ValueError('At least one mutation trial is required')
	counts = Counter(trial.outcome for trial in trials)
	per_class: dict[str, Counter[str]] = defaultdict(Counter)
	for trial in trials:
		per_class[trial.mutation.mutation_class.value][trial.outcome.value] += 1
	return MutationSummary(
		procedure_id=trials[0].procedure_id,
		trials=len(trials),
		verified_completion=_outcome_rate(counts[TrialOutcome.VERIFIED_COMPLETION], len(trials)),
		safe_refusal=_outcome_rate(counts[TrialOutcome.SAFE_REFUSAL], len(trials)),
		incorrect_interaction=_outcome_rate(counts[TrialOutcome.INCORRECT_INTERACTION], len(trials)),
		per_class={name: dict(class_counts) for name, class_counts in sorted(per_class.items())},
		model_calls=sum(trial.model_calls for trial in trials),
		total_duration_seconds=sum(trial.duration_seconds for trial in trials),
		result_paths=result_paths,
	)


def _outcome_rate(count: int, total: int) -> OutcomeRate:
	if total <= 0:
		raise ValueError('Rate denominator must be positive')
	z = 1.959963984540054
	rate = count / total
	denominator = 1 + z**2 / total
	center = (rate + z**2 / (2 * total)) / denominator
	margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
	return OutcomeRate(
		count=count,
		rate=rate,
		wilson_low=0 if count == 0 else max(0, center - margin),
		wilson_high=1 if count == total else min(1, center + margin),
	)


async def run_trial(
	procedure: BrowserProcedure,
	url: str,
	parameters: dict[str, str],
	plan: MutationPlan,
	output_dir: Path,
) -> MutationTrial:
	trial_id = uuid7str()
	profile = BrowserProfile(
		headless=True,
		user_data_dir=None,
		window_size=ViewportSize(width=plan.viewport_width, height=plan.viewport_height),
	)
	browser = BrowserSession(browser_profile=profile)
	started = time.monotonic()
	await browser.start()
	try:
		await browser.navigate_to(url)
		await _install_audit_and_mutate(browser, procedure, plan)
		report = await DeterministicReplayer(
			procedure,
			ReplayOptions(output_dir=output_dir / 'replay-audits', optional_wait_seconds=0.05),
		).run(browser, parameters)
		audit = await _read_browser_audit(browser, procedure.expected_success_text)
		strategies = Counter(
			str(getattr(event, 'resolution_strategy', None) or 'unknown')
			for event in report.events
			if event.resolution_status == 'resolved'
		)
		trial = MutationTrial(
			trial_id=trial_id,
			procedure_id=procedure.procedure_id,
			mutation=plan,
			outcome=classify_trial(report.status, audit),
			replay_status=report.status,
			model_calls=report.model_calls,
			actions_attempted=report.actions_attempted,
			duration_seconds=time.monotonic() - started,
			success_visible=audit.success_visible,
			submission_count=audit.submission_count,
			interaction_count=audit.interaction_count,
			incorrect_interaction_count=len(audit.unexpected_interactions),
			resolution_strategies=dict(strategies),
			reason=report.reason,
			replay_audit_dir=report.audit_dir,
			created_at=datetime.now(timezone.utc).isoformat(),
		)
		path = output_dir / 'trials' / f'{trial_id}.json'
		_write_private_json(path, trial.model_dump(mode='json'))
		return trial
	finally:
		await browser.kill()


async def _install_audit_and_mutate(
	browser: BrowserSession,
	procedure: BrowserProcedure,
	plan: MutationPlan,
) -> None:
	page = await browser.must_get_current_page()
	steps = [
		{
			'id': step.id,
			'action': step.action.value,
			'required': {
				field: value
				for field, value in step.locator.required.items()
				if field == 'node_name' or field.startswith('attributes.')
			},
		}
		for step in procedure.steps
	]
	payload = json.dumps({'plan': plan.model_dump(mode='json'), 'steps': steps}, ensure_ascii=False)
	script = f"""() => {{
		const config = {payload};
		const norm = value => String(value || '').trim().replace(/\\s+/g, ' ').toLocaleLowerCase();
		const matches = (element, required) => Object.entries(required).every(([field, expected]) => {{
			if (field === 'node_name') return norm(element.tagName) === norm(expected);
			return norm(element.getAttribute(field.slice('attributes.'.length))) === norm(expected);
		}});
		const candidates = Array.from(document.querySelectorAll('input,select,textarea,button,[contenteditable="true"]'));
		for (const step of config.steps) {{
			for (const element of candidates.filter(item => matches(item, step.required))) {{
				const ids = (element.dataset.memorableExpected || '').split('|').filter(Boolean);
				if (!ids.includes(step.id)) ids.push(step.id);
				element.dataset.memorableExpected = ids.join('|');
			}}
		}}
		window.__memorableMutationAudit = {{ interactions: [], submissions: 0 }};
		for (const type of ['click', 'input', 'change']) {{
			document.addEventListener(type, event => {{
				const target = event.target;
				window.__memorableMutationAudit.interactions.push({{
					type,
					tag: target && target.tagName || '',
					name: target && target.getAttribute && target.getAttribute('name') || '',
					input_type: target && target.getAttribute && target.getAttribute('type') || '',
					expected: target && target.dataset && target.dataset.memorableExpected || '',
					decoy: target && target.dataset && target.dataset.memorableDecoy || ''
				}});
			}}, true);
		}}
		document.addEventListener('submit', () => window.__memorableMutationAudit.submissions += 1, true);

		if (config.plan.randomize_ids) {{
			const scriptText = Array.from(document.scripts).map(script => script.textContent || '').join('\\n');
			for (const element of document.querySelectorAll('[id]')) {{
				const oldId = element.id;
				if (scriptText.includes(oldId)) continue;
				const newId = `mut-${{config.plan.seed}}-${{Math.random().toString(16).slice(2)}}`;
				for (const label of document.querySelectorAll('label[for]')) {{
					if (label.htmlFor === oldId) label.htmlFor = newId;
				}}
				element.id = newId;
			}}
		}}
		if (config.plan.reorder_blocks) {{
			for (const section of document.querySelectorAll('.form-section')) {{
				const blocks = Array.from(section.children).filter(item => item.classList.contains('mb-3'));
				blocks.sort((a, b) => norm(a.textContent).localeCompare(norm(b.textContent)));
				if (config.plan.seed % 2) blocks.reverse();
				for (const block of blocks) section.appendChild(block);
			}}
		}}
		if (config.plan.wrap_controls) {{
			for (const element of Array.from(document.querySelectorAll('[data-memorable-expected]'))) {{
				const wrapper = document.createElement('div');
				wrapper.dataset.mutationWrapper = String(config.plan.seed);
				wrapper.style.cssText = `padding:${{8 + config.plan.seed % 13}}px;margin:${{config.plan.seed % 7}}px 0`;
				element.parentNode.insertBefore(wrapper, element);
				wrapper.appendChild(element);
			}}
		}}
		if (config.plan.hidden_decoys) {{
			for (const element of Array.from(document.querySelectorAll('[data-memorable-expected]'))) {{
				const clone = element.cloneNode(true);
				clone.removeAttribute('data-memorable-expected');
				clone.removeAttribute('id');
				clone.dataset.memorableDecoy = '1';
				clone.hidden = true;
				if ('disabled' in clone) {{
					clone.disabled = true;
					clone.setAttribute('form', 'memorable-nonexistent-decoy-form');
				}}
				clone.style.display = 'none';
				element.parentNode.insertBefore(clone, element);
			}}
		}}
		if (config.plan.drift_accessible_names) {{
			for (const label of document.querySelectorAll('label')) {{
				const textNodes = Array.from(label.childNodes).filter(node => node.nodeType === Node.TEXT_NODE);
				if (textNodes.length) {{
					textNodes[0].textContent = `تسمية بديلة ${{config.plan.seed}}`;
					for (const node of textNodes.slice(1)) node.textContent = '';
				}} else {{
					label.appendChild(document.createTextNode(`تسمية بديلة ${{config.plan.seed}}`));
				}}
			}}
			const submit = document.querySelector('button[type="submit"]');
			if (submit) submit.textContent = `تأكيد وإرسال ${{config.plan.seed}}`;
		}}
		if (config.plan.duplicate_submit) {{
			const submit = document.querySelector('button[type="submit"]');
			if (submit) {{
				const clone = submit.cloneNode(true);
				clone.removeAttribute('data-memorable-expected');
				clone.dataset.memorableDecoy = '1';
				submit.parentNode.appendChild(clone);
			}}
		}}
		if (config.plan.obstruct_submit) {{
			const overlay = document.createElement('div');
			overlay.dataset.memorableDecoy = '1';
			overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:rgba(255,255,255,0.01);pointer-events:auto';
			document.body.appendChild(overlay);
		}}
		return JSON.stringify({{ marked: document.querySelectorAll('[data-memorable-expected]').length }});
	}}"""
	await page.evaluate(script)


async def _read_browser_audit(browser: BrowserSession, expected_success_text: str | None) -> BrowserAudit:
	page = await browser.must_get_current_page()
	expected = json.dumps(expected_success_text or '', ensure_ascii=False)
	payload = await page.evaluate(
		f"""() => {{
			const audit = window.__memorableMutationAudit || {{ interactions: [], submissions: 0 }};
			const expectedText = {expected};
			const successVisible = Array.from(document.querySelectorAll('body *')).some(element => {{
				const style = getComputedStyle(element);
				return style.display !== 'none' && style.visibility !== 'hidden' && element.innerText && element.innerText.includes(expectedText);
			}});
			const relevant = audit.interactions.filter(event =>
				event.type === 'click' || ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.tag)
			);
			return JSON.stringify({{
				success_visible: successVisible,
				submission_count: audit.submissions,
				interaction_count: relevant.length,
				unexpected_interactions: relevant
					.filter(event => !event.expected)
					.map(event => ({{
						type: String(event.type), tag: String(event.tag), name: String(event.name),
						input_type: String(event.input_type), decoy: String(event.decoy)
					}}))
			}});
		}}"""
	)
	return BrowserAudit.model_validate_json(payload)


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
	os.chmod(path.parent, 0o700)
	temporary = path.with_name(f'.{path.name}.partial')
	temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
	os.chmod(temporary, 0o600)
	temporary.replace(path)


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
	procedure = BrowserProcedure.read(args.procedure)
	parameters = _parse_parameters(args.parameter)
	classes = [MutationClass(value) for value in args.mutation_class] if args.mutation_class else list(MutationClass)
	output_dir = args.output_dir.expanduser().resolve()
	trials: list[MutationTrial] = []
	paths: list[str] = []
	for mutation_class in classes:
		for repeat in range(args.repeat):
			seed = args.seed + repeat
			trial = await run_trial(
				procedure,
				args.url,
				parameters,
				build_mutation_plan(mutation_class, seed),
				output_dir,
			)
			trials.append(trial)
			paths.append(str(output_dir / 'trials' / f'{trial.trial_id}.json'))
			print(
				f'{mutation_class.value} seed={seed}: {trial.outcome.value} '
				f'({trial.replay_status.value}, {trial.duration_seconds:.2f}s)'
			)
	summary = summarize_trials(trials, paths)
	_write_private_json(output_dir / 'summary.json', summary.model_dump(mode='json'))
	print(summary.model_dump_json(indent=2))
	return 1 if summary.incorrect_interaction.count else 0


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('procedure', type=Path)
	parser.add_argument('url')
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--repeat', type=int, default=3)
	parser.add_argument('--seed', type=int, default=100)
	parser.add_argument('--mutation-class', action='append', choices=[item.value for item in MutationClass], default=[])
	parser.add_argument('-p', '--parameter', action='append', default=[])
	args = parser.parse_args()
	raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == '__main__':
	main()
