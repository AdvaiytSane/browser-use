const normalize = (value) => String(value ?? "").replace(/\s+/g, " ").trim().toLowerCase();

const ui = {
  form: document.querySelector("#spotify-task-form"),
  task: document.querySelector("#spotify-task"),
  graph: document.querySelector("#spotify-graph"),
  graphId: document.querySelector("#spotify-graph-id"),
  binding: document.querySelector("#spotify-binding"),
  result: document.querySelector("#spotify-result"),
  corpus: document.querySelector("#spotify-corpus"),
  history: document.querySelector("#spotify-run-history"),
  runCount: document.querySelector("#spotify-run-count"),
};

let spotifyGraph;
let runCount = 0;

function spotifyIntent(task) {
  const quoted = task.match(/[“"]([^”"]+)[”"]/);
  const patterns = [
    /(?:search(?:\s+spotify)?\s+for)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)/i,
    /(?:artist\s+result\s+for|artist\s+page\s+for)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)/i,
    /(?:navigate|go|open)(?:\s+spotify)?\s+to\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)/i,
    /(?:song|track|music)\s+(?:by|from)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)/i,
    /(?:best|top|popular)\s+(?:song|track)\s+(?:for|by)\s+(.+?)(?:,|\s+and\s+|\s+then\s+|$)/i,
  ];
  const match = quoted || patterns.map((pattern) => task.match(pattern)).find(Boolean);
  const artist = (match?.[1] || "").trim().replace(/^[“”"'\s]+|[“”"'.,;:!?\s]+$/g, "");
  const lowered = task.toLowerCase();
  const rankWords = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5 };
	const playIntent = /\bplay\b/.test(lowered);
  const trackIntent = playIntent || /\b(track|song|popular|first|second|third|fourth|fifth)\b/.test(lowered);
  const rankEntry = Object.entries(rankWords).find(([word]) => new RegExp(`\\b${word}\\b`).test(lowered));
  const numericRank = lowered.match(/\b(\d{1,2})(?:[a-z]{2})?\s+(?:visible\s+)?(?:track|song)\b/);
  return {
    artist,
    terminal: playIntent ? "play_track" : trackIntent ? "popular_track" : "canonical_artist",
    rank: trackIntent ? (rankEntry?.[1] || Number(numericRank?.[1]) || 1) : null,
  };
}

function bindingCard(label, value) {
  const item = document.createElement("span");
  const key = document.createElement("small");
  const bound = document.createElement("strong");
  key.textContent = label;
  bound.textContent = String(value);
  item.append(key, bound);
  return item;
}

function graphNode(node, status, label) {
  const card = document.createElement("div");
  card.className = `graph-node ${status}`;
  const id = document.createElement("code");
  const detail = document.createElement("span");
  id.textContent = node.id;
  detail.textContent = label;
  card.append(id, detail);
  return card;
}

function renderSpotifyTask() {
  if (!spotifyGraph) return;
  const intent = spotifyIntent(ui.task.value);
	const activeNodes = intent.terminal === "play_track"
		? ["spotify_home", "search_results", "canonical_artist", "popular_track", "play_track"]
		: intent.terminal === "popular_track"
			? ["spotify_home", "search_results", "canonical_artist", "popular_track"]
			: ["spotify_home", "search_results", "canonical_artist"];
  ui.graph.replaceChildren();
  spotifyGraph.nodes.forEach((node, index) => {
    if (index > 0) {
      const connector = document.createElement("span");
      connector.className = `graph-arrow ${activeNodes.includes(node.id) ? "active" : ""}`;
      connector.textContent = "→";
      ui.graph.append(connector);
    }
    const status = activeNodes.includes(node.id) ? "active" : "inactive";
    const label = node.id === intent.terminal ? "task terminal" : node.state_guard;
    const card = graphNode(node, `${status} ${node.id === intent.terminal ? "terminal" : ""}`, label);
    ui.graph.append(card);
  });
  ui.binding.replaceChildren(
    bindingCard("artist", intent.artist || "unbound"),
    bindingCard("track_rank", intent.rank ?? "not needed"),
    bindingCard("model calls", "0"),
  );
  const sample = spotifyGraph.training.samples?.find((item) => normalize(item.artist) === normalize(intent.artist));
  if (!intent.artist) {
    ui.result.textContent = "Describe an artist using phrases such as “search for,” “open,” or “song by.”";
    ui.result.className = "graph-result";
  } else if (intent.terminal === "canonical_artist") {
    ui.result.textContent = sample
      ? `Recorded proof: stopped on ${sample.artist}'s canonical artist page.`
      : `Runtime plan: open ${intent.artist}'s canonical artist page, verify it, then stop.`;
    ui.result.className = "graph-result completed";
  } else {
    const track = sample?.popular_tracks?.[intent.rank - 1];
		const verb = intent.terminal === "play_track" ? "play" : "read";
		ui.result.textContent = track
			? `Recorded proof: Popular #${intent.rank} for ${sample.artist} was “${track.name}”; runtime will ${verb} it.`
			: `Runtime plan: bind ${intent.artist}, verify its artist page, then ${verb} Popular #${intent.rank}.`;
    ui.result.className = "graph-result completed";
  }
}

function evidenceLabel(event, nodeId, payload) {
	if (nodeId === "spotify_home") return event ? "Spotify scope verified" : "not reached";
  if (!event) return "not reached";
  if (event.status === "needs_recovery") return event.reason || "needs recovery";
	if (event.evidence?.playback_started) return "playback verified";
  if (event.evidence?.track_name) return `#${event.evidence.rank} · ${event.evidence.track_name}`;
  if (event.evidence?.artist_url) return "artist page verified";
  if (event.evidence?.url_path) return `query stabilized · attempt ${event.evidence.input_attempts || 1}`;
  return event.status;
}

function shortHash(value) {
  return value ? `${value.slice(0, 10)}…` : "unavailable";
}

function stateField(label, value, monospace = false) {
  const row = document.createElement("div");
  const key = document.createElement("small");
  const content = document.createElement(monospace ? "code" : "span");
  key.textContent = label;
  content.textContent = value ?? "unavailable";
  row.append(key, content);
  return row;
}

function stateInspector(node, event) {
  const details = document.createElement("details");
  details.className = `state-inspector ${event.status}`;
  const summary = document.createElement("summary");
  const identity = document.createElement("span");
  const nodeName = document.createElement("code");
  const status = document.createElement("small");
  nodeName.textContent = node.id;
  status.textContent = `${event.status.replace("_", " ")} · ${event.duration_ms} ms`;
  identity.append(nodeName, status);
  const signature = document.createElement("code");
  signature.className = "state-summary-signature";
  signature.textContent = `DOM ${shortHash(event.state?.dom_sha256)}`;
  summary.append(identity, signature);

  const body = document.createElement("div");
  body.className = "state-inspector-body";
  if (event.state?.screenshot_url) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    const caption = document.createElement("figcaption");
    image.src = event.state.screenshot_url;
    image.alt = `${node.id} browser state`;
    image.loading = "lazy";
    caption.textContent = `captured ${event.state.captured_at}`;
    figure.append(image, caption);
    body.append(figure);
  }

  const metadata = document.createElement("div");
  metadata.className = "state-metadata";
  const viewport = event.state?.viewport;
  const viewportText = viewport?.width && viewport?.height
    ? `${viewport.width} × ${viewport.height} @ ${viewport.device_pixel_ratio || 1}x`
    : "unavailable";
	const delta = event.state?.delta_from_previous || {};
	const deltaText = delta.previous_state
		? `from ${delta.previous_state}: URL ${delta.url_changed ? "changed" : "same"}, DOM ${delta.dom_changed ? "changed" : "same"}, semantics ${delta.semantic_dom_changed ? "changed" : "same"}, pixels ${delta.screenshot_changed ? "changed" : "same"}, selectors ${delta.selectors_added >= 0 ? "+" : ""}${delta.selectors_added}`
		: "initial captured state";
  metadata.append(
    stateField("URL", event.state?.url, true),
    stateField("title", event.state?.title),
    stateField("viewport", viewportText, true),
    stateField("selector index", event.selector_index ?? "none", true),
		stateField("action time", `${event.action_duration_ms} ms`, true),
		stateField("capture overhead", `${event.capture_duration_ms} ms`, true),
    stateField("interactive selectors", event.state?.selector_count ?? 0, true),
    stateField("semantic chars", event.state?.semantic_dom_chars ?? 0, true),
		stateField("change from previous", deltaText, true),
    stateField("DOM SHA-256", event.state?.dom_sha256, true),
    stateField("semantic SHA-256", event.state?.semantic_dom_sha256, true),
    stateField("screenshot SHA-256", event.state?.screenshot_sha256, true),
    stateField("browser errors", event.state?.browser_error_count ?? 0, true),
  );
  if (event.state?.capture_error || event.state?.state_error) {
    metadata.append(stateField("capture warning", event.state.capture_error || event.state.state_error));
  }
  body.append(metadata);

  const raw = document.createElement("details");
  raw.className = "raw-state";
  const rawSummary = document.createElement("summary");
  const pre = document.createElement("pre");
  rawSummary.textContent = "raw event JSON";
  pre.textContent = JSON.stringify(event, null, 2);
  raw.append(rawSummary, pre);
  body.append(raw);
  details.append(summary, body);
  return details;
}

function appendRunGraph(payload, task, intent) {
  ui.history.querySelector(".run-history-empty")?.remove();
  runCount += 1;
  ui.runCount.textContent = `${runCount} ${runCount === 1 ? "run" : "runs"}`;
  const item = document.createElement("li");
  item.className = `run-card ${payload?.status || "needs_recovery"}`;
  const header = document.createElement("div");
  header.className = "run-card-header";
  const title = document.createElement("div");
  const kicker = document.createElement("small");
  const heading = document.createElement("strong");
  kicker.textContent = payload?.run_id ? `run ${payload.run_id.slice(0, 8)}` : `run ${String(runCount).padStart(2, "0")}`;
  heading.textContent = payload?.intent?.artist || intent.artist || "Unbound task";
  title.append(kicker, heading);
  const status = document.createElement("span");
  status.className = "run-status";
  status.textContent = payload?.status || "needs recovery";
  header.append(title, status);

  const taskText = document.createElement("p");
  taskText.className = "run-task";
  taskText.textContent = task;
  const flow = document.createElement("div");
  flow.className = "run-flow";
	const reachedStates = [];
  const path = payload?.intent?.path || ["spotify_home"];
  path.forEach((nodeId, index) => {
    if (index > 0) {
      const arrow = document.createElement("span");
      arrow.className = "run-arrow";
      arrow.textContent = "→";
      flow.append(arrow);
    }
    const node = spotifyGraph.nodes.find((candidate) => candidate.id === nodeId) || { id: nodeId };
    const event = payload?.events?.find((candidate) => candidate.target === nodeId);
		const nodeStatus = event?.status || "pending";
    flow.append(graphNode(node, `run-node ${nodeStatus}`, evidenceLabel(event, nodeId, payload)));
		if (event) reachedStates.push([node, event]);
  });
	const inspectors = document.createElement("div");
	inspectors.className = "state-inspectors";
	reachedStates.forEach(([node, event]) => inspectors.append(stateInspector(node, event)));

  const outcome = document.createElement("p");
  outcome.className = "run-outcome";
	if (payload?.status && payload.status !== "completed") {
		outcome.textContent = `Recovery · ${payload.reason || "browser evidence did not satisfy the next node"}`;
	} else if (payload?.playback_started) {
		outcome.textContent = `Playing · Popular #${payload.track.rank} · ${payload.track.name}`;
	} else if (payload?.track) {
    outcome.textContent = `Result · Popular #${payload.track.rank} · ${payload.track.name}`;
  } else if (payload?.status === "completed") {
    outcome.textContent = `Result · canonical artist page verified · stopped as requested`;
	} else {
    outcome.textContent = `Recovery · ${payload?.reason || "artist binding failed before browser launch"}`;
  }
  item.append(header, taskText, flow, inspectors, outcome);
  ui.history.prepend(item);
}

async function runSpotifyLive(event) {
  event.preventDefault();
  renderSpotifyTask();
  const task = ui.task.value;
  const intent = spotifyIntent(task);
  if (!intent.artist) {
    ui.result.textContent = "Artist binding failed. No browser was opened.";
    ui.result.className = "graph-result refused";
    appendRunGraph(null, task, intent);
    return;
  }
  const runButton = ui.form.querySelector('button[type="submit"]');
  runButton.disabled = true;
  runButton.textContent = "Running in visible Chromium…";
  ui.result.textContent = "Opening Spotify and rebinding against its live DOM…";
  ui.result.className = "graph-result running";
  let payload;
  try {
    const response = await fetch("/api/spotify/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
    const body = await response.text();
    try { payload = JSON.parse(body); } catch { payload = null; }
    if (!response.ok || !payload) {
      if (response.status === 501 || !payload) {
        throw new Error("Start spotify_demo_server.py instead of python -m http.server.");
      }
      throw new Error(payload.detail || payload.reason || "The semantic executor stopped for recovery.");
    }
    if (payload.status !== "completed") throw new Error(payload.reason || "The executor stopped for recovery.");
		const result = payload.playback_started
			? `Playing ${payload.intent.artist} · Popular #${payload.track.rank} · “${payload.track.name}”`
			: payload.track
				? `${payload.intent.artist} · Popular #${payload.track.rank} · “${payload.track.name}”`
      : `Opened and verified ${payload.intent.artist}, then stopped as requested.`;
    ui.result.textContent = `Completed live: ${result} · ${payload.model_calls} model calls.`;
    ui.result.className = "graph-result completed";
  } catch (error) {
    ui.result.textContent = error instanceof Error ? error.message : "The live run failed.";
    ui.result.className = "graph-result refused";
  } finally {
    appendRunGraph(payload, task, intent);
    runButton.disabled = false;
    runButton.textContent = "Run live on Spotify";
  }
}

ui.form.addEventListener("submit", runSpotifyLive);
ui.task.addEventListener("input", renderSpotifyTask);
document.querySelectorAll("[data-task]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.task.value = button.dataset.task;
    renderSpotifyTask();
  });
});

fetch("spotify-graph.json")
  .then((response) => {
    if (!response.ok) throw new Error("The Spotify graph could not be loaded.");
    return response.json();
  })
  .then((graph) => {
    spotifyGraph = graph;
    ui.graphId.textContent = graph.graph_id;
    ui.corpus.textContent = `${graph.training.trace_count} live traces · ${graph.training.distinct_artists} artists · ${graph.training.model_calls} model calls`;
    renderSpotifyTask();
  })
  .catch((error) => {
    ui.result.textContent = error.message;
    ui.result.className = "graph-result refused";
  });
