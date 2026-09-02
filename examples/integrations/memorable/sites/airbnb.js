const form = document.querySelector('#task-form');
const taskInput = document.querySelector('#task');
const runButton = document.querySelector('#run');
const statusNode = document.querySelector('#status');
const resultNode = document.querySelector('#result');
const runStateNode = document.querySelector('#run-state');
const actionStream = document.querySelector('#action-stream');
const actionCount = document.querySelector('#action-count');
const browserShot = document.querySelector('#browser-shot');
const browserEmpty = document.querySelector('#browser-empty');
const browserURL = document.querySelector('#browser-url');
const domSignature = document.querySelector('#dom-signature');
const browserFrame = document.querySelector('#browser-frame');
const screenMode = document.querySelector('#screen-mode');

const MODE_LABELS = {
	idle: 'Waiting',
	deterministic: 'Deterministic',
	agent: 'Agent decision',
	repair: 'Popup repair',
	refuse: 'Refusal',
};

let activeRunId = null;
let pollTimer = null;
let latestEvents = [];
let shownEventIndex = -1;

function setStatus(message, kind = '') {
	statusNode.textContent = message;
	statusNode.className = `status ${kind}`.trim();
}

function escapeHTML(value) {
	return String(value ?? '')
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#039;');
}

function renderResult(report) {
	const selected = report.selected;
	if (!selected) {
		resultNode.hidden = true;
		return;
	}
	const params = report.parameters;
	const details = [
		report.workflow_id.replaceAll('_', ' '),
		params.city,
		`${params.check_in} → ${params.check_out}`,
		`${params.adults} adult${params.adults === 1 ? '' : 's'}`,
		`${report.candidate_count} listings checked`,
	].join(' · ');
	let primary = `${selected.currency}${selected.total_price}`;
	if (report.workflow_id === 'highest_rated_loaded_stay') {
		primary = `${selected.rating} ★`;
	} else if (report.workflow_id === 'most_reviewed_loaded_stay') {
		primary = `${selected.review_count} reviews`;
	} else if (report.workflow_id === 'first_visible_stay') {
		primary = '#1 result';
	}
	resultNode.innerHTML = `
		<div>
			<h2>${escapeHTML(selected.title || report.listing_heading || 'Selected stay')}</h2>
			<p>${escapeHTML(details)}</p>
		</div>
		<div class="price">${escapeHTML(primary)}</div>
	`;
	resultNode.hidden = false;
}

function setScreenMode(mode = 'idle') {
	const safeMode = Object.hasOwn(MODE_LABELS, mode) ? mode : 'idle';
	browserFrame.dataset.mode = safeMode;
	screenMode.textContent = MODE_LABELS[safeMode];
}

function resetObserver() {
	latestEvents = [];
	shownEventIndex = -1;
	actionStream.innerHTML = '<li class="stream-empty">Connecting to the browser run…</li>';
	actionCount.textContent = '0 actions';
	runStateNode.textContent = 'Running';
	runStateNode.className = 'run-state running';
	browserShot.hidden = true;
	browserShot.removeAttribute('src');
	browserEmpty.hidden = false;
	browserURL.textContent = 'Waiting for Chromium';
	domSignature.textContent = 'DOM —';
	setScreenMode();
}

function showEventState(event) {
	if (!event?.state) return;
	shownEventIndex = event.index;
	browserURL.textContent = event.state.url || event.state.title || 'Browser state captured';
	browserURL.title = event.state.url || '';
	domSignature.textContent = event.state.dom_signature
		? `DOM ${event.state.dom_signature} · ${event.state.selectors} nodes`
		: 'DOM —';
	setScreenMode(event.mode);
	if (event.state.screenshot_url) {
		browserShot.src = `${event.state.screenshot_url}&v=${event.index}`;
		browserShot.hidden = false;
		browserEmpty.hidden = true;
	}
	document.querySelectorAll('.action.selected').forEach((node) => node.classList.remove('selected'));
	document.querySelector(`.action[data-event-index="${event.index}"]`)?.classList.add('selected');
}

function renderLive(snapshot) {
	const status = snapshot.status || 'running';
	runStateNode.textContent = status.replaceAll('_', ' ');
	runStateNode.className = `run-state ${status}`;
	latestEvents = snapshot.events || [];
	actionCount.textContent = `${latestEvents.length} action${latestEvents.length === 1 ? '' : 's'}`;
	actionStream.innerHTML = latestEvents
		.map(
			(event) => `
			<li class="action ${event.status === 'running' ? 'running' : ''}" data-mode="${escapeHTML(event.mode)}" data-event-index="${event.index}" ${event.state ? 'tabindex="0"' : ''}>
				<div class="action-main">
					<p class="action-title">
						<span>${escapeHTML(event.label)}</span>
						<span class="action-mode">${escapeHTML(event.mode)}</span>
					</p>
					<p class="action-detail">${escapeHTML(event.detail)}</p>
				</div>
				<span class="action-time">${event.duration_ms ? `${event.duration_ms}ms` : event.status}</span>
			</li>
		`,
		)
		.join('');

	const newestState = [...latestEvents].reverse().find((event) => event.state?.screenshot_url);
	if (newestState && newestState.index !== shownEventIndex) {
		showEventState(newestState);
	}
	setScreenMode(latestEvents.at(-1)?.mode);
	actionStream.scrollTop = actionStream.scrollHeight;
}

async function fetchLive(runId) {
	const response = await fetch(`/api/airbnb/live?run_id=${encodeURIComponent(runId)}`, { cache: 'no-store' });
	if (!response.ok) return null;
	const snapshot = await response.json();
	renderLive(snapshot);
	return snapshot;
}

async function pollLive() {
	const runId = activeRunId;
	if (!runId) return;
	try {
		await fetchLive(runId);
	} catch {
		// The POST owns user-facing failure reporting; observer polling is best-effort.
	}
	if (activeRunId === runId) {
		pollTimer = window.setTimeout(pollLive, 350);
	}
}

function stopPolling() {
	if (pollTimer) window.clearTimeout(pollTimer);
	pollTimer = null;
}

function chooseEvent(eventIndex) {
	const event = latestEvents.find((item) => item.index === eventIndex);
	if (event?.state) showEventState(event);
}

actionStream.addEventListener('click', (event) => {
	const action = event.target.closest('.action');
	if (action) chooseEvent(Number(action.dataset.eventIndex));
});

actionStream.addEventListener('keydown', (event) => {
	if (event.key !== 'Enter' && event.key !== ' ') return;
	const action = event.target.closest('.action');
	if (!action) return;
	event.preventDefault();
	chooseEvent(Number(action.dataset.eventIndex));
});

browserShot.addEventListener('error', () => {
	browserShot.hidden = true;
	browserEmpty.hidden = false;
});

form.addEventListener('submit', async (event) => {
	event.preventDefault();
	const task = taskInput.value.trim();
	if (!task) {
		setStatus('Describe where, when, and what you want to find.', 'error');
		taskInput.focus();
		return;
	}

	stopPolling();
	activeRunId = crypto.randomUUID();
	const runId = activeRunId;
	runButton.disabled = true;
	resultNode.hidden = true;
	resetObserver();
	setStatus('Understanding the task… live execution will appear on the right.', 'busy');
	pollLive();

	try {
		const response = await fetch('/api/airbnb/run', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ task, client_run_id: runId }),
		});
		const payload = await response.json();
		await fetchLive(runId);
		if (!response.ok) {
			throw new Error(payload.detail || payload.reason || 'The task could not be completed.');
		}
		renderResult(payload);
		setStatus(`Done · ${payload.model_calls} model calls · verified listing opened`);
	} catch (error) {
		runStateNode.textContent = 'Failed';
		runStateNode.className = 'run-state failed';
		setScreenMode('refuse');
		setStatus(error instanceof Error ? error.message : 'The task could not be completed.', 'error');
	} finally {
		if (activeRunId === runId) activeRunId = null;
		stopPolling();
		runButton.disabled = false;
	}
});
