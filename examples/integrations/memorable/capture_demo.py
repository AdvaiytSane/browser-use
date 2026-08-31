"""Run one high-fidelity offline capture against a caller-selected website."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from browser_use import Agent, ChatAnthropic, ChatBrowserUse
from browser_use.browser import BrowserProfile
from browser_use.browser.profile import ViewportSize
from examples.integrations.memorable.offline_capture import (
	OfflineCaptureOptions,
	OfflineRunCapture,
	write_corpus_report,
)


def _arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--task', required=True)
	parser.add_argument('--start-url')
	parser.add_argument('--provider', choices=['browser-use', 'anthropic'], default='browser-use')
	parser.add_argument('--model')
	parser.add_argument('--env-file', type=Path)
	parser.add_argument('--max-steps', type=int, default=20)
	parser.add_argument('--viewport-width', type=int, default=1280)
	parser.add_argument('--viewport-height', type=int, default=800)
	parser.add_argument('--expected-success-text')
	parser.add_argument('--label')
	parser.add_argument('--chaos-seed')
	parser.add_argument('--conversations', action='store_true')
	parser.add_argument('--har', action='store_true')
	parser.add_argument('--video', action='store_true')
	parser.add_argument('--downloads', action='store_true')
	return parser.parse_args()


def _llm(args: argparse.Namespace):
	if args.provider == 'browser-use':
		if not os.environ.get('BROWSER_USE_API_KEY'):
			raise RuntimeError('BROWSER_USE_API_KEY is required for --provider browser-use')
		return ChatBrowserUse(model=args.model) if args.model else ChatBrowserUse()
	if not os.environ.get('ANTHROPIC_API_KEY'):
		raise RuntimeError('ANTHROPIC_API_KEY is required for --provider anthropic')
	return ChatAnthropic(model=args.model or 'claude-sonnet-4-6', temperature=0.0)


async def _run(args: argparse.Namespace) -> None:
	if args.env_file:
		load_dotenv(args.env_file, override=False)

	initial_actions = [{'navigate': {'url': args.start_url, 'new_tab': False}}] if args.start_url else None
	agent = Agent(
		task=args.task,
		llm=_llm(args),
		initial_actions=initial_actions,
		directly_open_url=not bool(initial_actions),
		max_actions_per_step=1,
		use_judge=False,
		calculate_cost=True,
		browser_profile=BrowserProfile(
			window_size=ViewportSize(width=args.viewport_width, height=args.viewport_height),
		),
	)
	capture = OfflineRunCapture(
		OfflineCaptureOptions(
			output_dir=args.output_dir,
			include_screenshots=True,
			include_dom_text=True,
			include_eval_dom=True,
			include_full_dom=True,
			include_candidates=True,
			include_rendered_page=True,
			include_conversations=args.conversations,
			include_har=args.har,
			include_video=args.video,
			include_downloads=args.downloads,
			expected_success_text=args.expected_success_text,
			run_label=args.label,
			chaos_seed=args.chaos_seed,
		)
	)

	history = await capture.run_agent(agent, max_steps=args.max_steps)
	report_path = write_corpus_report(args.output_dir)
	print(
		json.dumps(
			{
				'run_dir': str(capture.final_dir),
				'corpus_report': str(report_path),
				'is_done': history.is_done(),
				'is_successful': history.is_successful(),
				'has_errors': history.has_errors(),
				'steps': len(history.history),
				'actions': history.action_names(),
			},
			indent=2,
		)
	)


if __name__ == '__main__':
	asyncio.run(_run(_arguments()))
