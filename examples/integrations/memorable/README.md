# Offline Browser Use run capture

This prototype records a Browser Use agent run into a local, self-describing bundle for later procedure mining. The collector does not call Memorable, an embedding service, or any other network endpoint. It composes Browser Use's existing hooks and does not modify the agent loop.

## Open the Replay Lab

Start the loopback demo server from the repository root:

```bash
uv run python examples/integrations/memorable/spotify_demo_server.py --no-headless
```

Then open `http://127.0.0.1:8765/`. The dashboard routes Spotify tasks through one parameterized graph. A run can stop at the canonical artist page, extract a ranked Popular track, or play that track. Every run gets an evidence-backed graph card in the dashboard.

The server keeps one visible Chromium window open and reuses `./tmp/spotify-demo-profile`, so playback can retain an authenticated Spotify session. On the first guest playback attempt, sign in in that window and rerun the task. Guest playback is reported as `needs_recovery`; it is never misreported as success. Override the profile location with `--profile-dir`. Follows, likes, and playlist mutations remain disabled.

The dashboard POSTs its task to the loopback server, which launches visible Chromium and returns the verified graph result. A plain `python -m http.server` can render the page but cannot run Browser Use, so the Spotify button explicitly reports that the live runner is unavailable.

## Capture a run

```python
from pathlib import Path

from browser_use import Agent, ChatBrowserUse
from examples.integrations.memorable.offline_capture import (
	OfflineCaptureOptions,
	OfflineRunCapture,
)

agent = Agent(
	task='Complete the task',
	llm=ChatBrowserUse(),
	max_actions_per_step=1,
	use_judge=False,
)
capture = OfflineRunCapture(
	OfflineCaptureOptions(
		output_dir=Path('./tmp/offline-runs'),
		include_screenshots=True,
		include_full_dom=True,
		include_rendered_page=True,
		include_conversations=True,
		include_har=True,
		expected_success_text='Order confirmed',
	)
)

history = await capture.run_agent(agent, max_steps=20)
print(capture.final_dir)
```

Set `BROWSER_USE_OFFLINE_CAPTURE=0` to bypass every capture hook and file write. The output directory is mode `0700`; files are mode `0600`. A bundle is first written as `.<run-id>.partial` and renamed only after its manifest and hashes are complete.

For exact action-level DOM transitions, use `max_actions_per_step=1`. With larger batches, the bundle still retains every action/result but has only one pre/post DOM pair for the batch; `derived.json` records that limitation.

The runnable CLI also supports explicit viewport perturbation and page-evidence verification:

```bash
uv run python -m http.server 8765 --directory examples/integrations/memorable/sites

uv run python examples/integrations/memorable/capture_demo.py \
  --output-dir ./tmp/offline-runs \
  --task "Complete the registration and report the success code" \
  --start-url http://127.0.0.1:8765/variant-a.html \
  --viewport-width 1280 --viewport-height 800 \
  --expected-success-text "Registration complete. Success code: MEM-042" \
  --conversations --har
```

Run the same task against `variant-b.html` at `500x800` to exercise reordered fields, a responsive layout, random element IDs, an optional overlay, different button text and a randomized completion delay. `?notice=0&delay=350` can pin the two timing controls while leaving IDs volatile.

## Bundle contents

```text
<run-id>/
├── manifest.json
├── history.json
├── usage.json
├── agent_state.json
├── events.jsonl
├── derived.json
├── conversations/                   # opt-in
├── network.har                      # opt-in
├── video/                           # opt-in; requires browser-use[video]
├── downloads/                       # opt-in
└── steps/<n>/
    ├── pre.json
    ├── pre.dom.txt
    ├── pre.eval.dom.txt
    ├── pre.full.dom.json.gz
    ├── pre.candidates.json
    ├── pre.html.gz
    ├── pre.rendered.txt
    ├── pre.page.json
    ├── pre.model_output.json
    ├── pre.png
    ├── post.json
    ├── post.dom.txt
    ├── post.eval.dom.txt
    ├── post.full.dom.json.gz
    ├── post.candidates.json
    ├── post.html.gz
    ├── post.rendered.txt
    ├── post.page.json
    ├── post.results.json
    └── post.png
```

The standard Browser Use history is preserved, including actions, results, interacted targets, timing and reasoning. Token/cost usage is written separately because the current `AgentHistoryList.model_dump()` implementation does not serialize its `usage` field.

Candidate records retain current DOM identity, accessibility role/name/properties, attributes, exact and stable hashes, XPath, ancestor context, computed styles, paint/stacking information, document bounds, viewport client rectangles and meaningful text. The compressed full tree retains Browser Use's simplified DOM, accessibility, snapshot and layout data.

The browser-native page capture closes a gap in the simplified tree: it stores raw current HTML, visible `body.innerText`, document/viewport metadata, the active element, and live form-control values, checked states, disabled states and geometry. This is why a terminal confirmation can be verified offline even when it is non-interactive and absent from Browser Use's normal DOM representation.

## Recorded facts versus inference

`events.jsonl` is the loss-minimized join of Browser Use history and captured states. An ordinary action result is classified as `no_reported_error`, not “successful,” because Browser Use reserves `success=True` for terminal `done` actions.

`derived.json` contains deterministic inferences with provenance:

- selected target and same-name ambiguity;
- target identity quality and available semantic evidence;
- URL, title, DOM, screenshot and candidate-set changes;
- likely transition type such as navigation, progressive disclosure or no observed change;
- candidate postcondition hints such as URL change, target disappearance or semantic element appearance.
- optional exact success-text evidence from the browser page, compared with the agent's own terminal claim.

These are hypotheses for offline analysis, not replacements for raw evidence.

## Analyze repeated runs

```python
from examples.integrations.memorable.offline_capture import write_corpus_report

report_path = write_corpus_report('./tmp/offline-runs')
print(report_path)
```

Or run the local analyzer directly:

```bash
uv run python examples/integrations/memorable/analyze_captures.py --output-dir ./tmp/offline-runs
```

The report groups exact task fingerprints, distinguishes agent-reported success from page-evidence-verified success, counts route variants, and aligns targets by semantic action slot rather than absolute step number. It identifies stable locator fields, checks their combined uniqueness against every captured pre-action candidate set, and quantifies cross-viewport geometry shifts. It deliberately does not recommend historical XPath, ID or Browser Use's `stable_hash` as a semantic replay key because those incorporate topology or identifiers that can change.

Two evidence-backed runs are enough for an adaptive replay experiment, not a general robustness claim. The analyzer requires at least three before it can emit `ready_for_replay_prototype` and reports route divergence and small samples as risks.

## Compile and replay a semantic procedure

Compile repeated successful runs entirely offline. The compiler retains stable DOM/accessibility fields and runtime parameter names, but excludes historical selector indices, coordinates, XPath, generated IDs, stable hashes, model prose and captured form values:

```bash
uv run python examples/integrations/memorable/procedure.py \
  ./tmp/offline-runs ./tmp/registration.procedure.json \
  --minimum-successful-runs 3
```

Replay uses a fresh Browser Use DOM snapshot before every action, requires exactly one visible/action-compatible match, executes through `Tools.act()`, and verifies a browser-native postcondition after every action. It makes no model calls:

```bash
uv run python examples/integrations/memorable/replay.py \
  ./tmp/registration.procedure.json \
  'http://127.0.0.1:8765/variant-b.html?notice=0' \
  -p full_name='Ada Lovelace' \
  -p email='ada.lovelace@example.test' \
  -p country='Canada' \
  --viewport-width 520 --viewport-height 700
```

The optional `Continue` transition executes on variant A and is skipped only when it is absent and a later required state is already uniquely available on variant B. `variant-b.html?duplicate_submit=1` creates two semantically identical submit buttons; replay must return `needs_recovery` without clicking either. Set `BROWSER_USE_MEMORABLE_SEMANTIC_REPLAY=0` for a fail-closed kill switch. Local audit bundles hash parameter-derived values rather than retaining them.

## BU Bench repeated-work protocol

The public BU Bench leaderboard measures cold, heterogeneous tasks. Do not add warm replay results to that cold score. `bu_bench.py` instead runs a paired repeated-work protocol against BU Bench V1's `InteractionTests`: a cold Browser Use agent run produces a private capture, and a compiled procedure can later run as a separately labeled zero-model warm trial.

The runner checkpoints execution metrics before calling the official judge. A judge outage therefore leaves `score` unknown while preserving the real steps, duration, cost, and capture bundle; it never fabricates a `0 steps / $0` agent failure.

```bash
uv run python examples/integrations/memorable/bu_bench.py \
  --benchmark-root /path/to/browser-use-benchmark \
  --task-index 2 \
  --output-dir ./tmp/bu-bench-memory \
  --mode cold --provider openai --model gpt-4.1 --repeat 3

uv run python examples/integrations/memorable/procedure.py \
  ./tmp/bu-bench-memory/captures/<task-fingerprint> \
  ./tmp/bu-bench-memory/procedure.json \
  --minimum-successful-runs 3

uv run python examples/integrations/memorable/bu_bench.py \
  --benchmark-root /path/to/browser-use-benchmark \
  --task-index 2 \
  --output-dir ./tmp/bu-bench-memory \
  --mode warm --procedure ./tmp/bu-bench-memory/procedure.json \
  --fallback-to-agent -p field_name='runtime value'
```

The runner rejects non-interaction tasks by default, checks the procedure's task fingerprint, resolves every action against fresh DOM state, and starts an agent fallback in a fresh browser after replay refusal. Fallback cost and duration remain a distinct result mode. Benchmark prompts, answers, captures, screenshots, and judge reasoning remain private local artifacts and must not be committed.

## Privacy boundary

The entire bundle is a **private raw tier**. DOM text, raw HTML, live form values, screenshots, model conversations, action inputs, downloads and HAR data can contain credentials or personal information. Conversations, HAR, video and downloads are opt-in. `manifest.json` reports credential-shaped matches but does not redact or delete the raw evidence.

Do not upload a raw bundle. A later Memorable relay should create a separate derived tier with a closed action-field allowlist, identifier scrubbing and a fail-closed secret scan.
