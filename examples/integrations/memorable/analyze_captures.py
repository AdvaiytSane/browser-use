"""Analyze local Browser Use capture bundles without calling a model."""

import argparse
import json
from pathlib import Path

from examples.integrations.memorable.offline_capture import analyze_capture_corpus, write_corpus_report


def _arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--destination', type=Path)
	return parser.parse_args()


def main() -> None:
	args = _arguments()
	report_path = write_corpus_report(args.output_dir, args.destination)
	report = analyze_capture_corpus(args.output_dir)
	print(
		json.dumps(
			{
				'report': str(report_path),
				'run_count': report['run_count'],
				'task_groups': [
					{
						'task_fingerprint': group['task_fingerprint'],
						'runs': group['runs'],
						'evidence_verified_successful_runs': group['evidence_verified_successful_runs'],
						'route_variants': len(group['route_variants']),
						'semantic_target_slots': len(group['stable_targets']),
						'inference': group['inference'],
					}
					for group in report['task_groups']
				],
			},
			indent=2,
		)
	)


if __name__ == '__main__':
	main()
