"""Loopback-only natural-language Airbnb demo with visible Chromium execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from examples.integrations.memorable.airbnb_hybrid import (
	AirbnbHybridReport,
	AirbnbTaskRoute,
	HybridEvent,
	TaskRoutingError,
	build_airbnb_llm,
	run_airbnb_task,
)

MAX_REQUEST_BYTES = 8 * 1024
STATIC_ROOT = Path(__file__).with_name('sites').resolve()


class AirbnbDemoRequest(BaseModel):
	model_config = ConfigDict(extra='forbid')

	task: str = Field(min_length=1, max_length=1000)
	client_run_id: str | None = Field(default=None, pattern=r'^[a-f0-9-]{36}$')


EVENT_LABELS = {
	'task_received': 'Start task router',
	'task_routing': 'Bind workflow',
	'search_navigation': 'Open parameterized search',
	'transient_overlay_repair': 'Dismiss known popup',
	'search_results': 'Verify result state',
	'selection_policy': 'Choose execution policy',
	'agent_open_listing': 'Open scoped listing',
	'agent_listing_selection': 'Return structured selection',
	'listing_details': 'Verify listing state',
}


def _live_event_detail(event: HybridEvent) -> str:
	evidence = event.evidence
	if event.node == 'task_routing':
		workflow = str(evidence.get('workflow_id') or 'no workflow').replace('_', ' ')
		return f'{workflow} · confidence {float(evidence.get("confidence") or 0):.2f}'
	if event.node == 'search_navigation':
		return 'Opened an exact city, date, and guest-count search URL.'
	if event.node == 'transient_overlay_repair':
		return f'Closed “{evidence.get("clicked") or "known overlay"}” before continuing.'
	if event.node == 'search_results':
		return f'Frozen {int(evidence.get("candidate_count") or 0)} eligible cards from the initial DOM.'
	if event.node == 'selection_policy':
		workflow = str(evidence.get('workflow_id') or '').replace('_', ' ')
		return event.reason or f'Applied {workflow}.'
	if event.node == 'agent_open_listing':
		return f'Opened candidate #{evidence.get("position")} from the frozen scope.'
	if event.node == 'agent_listing_selection':
		selection = evidence.get('selection') or {}
		return str(selection.get('reason') or 'Returned a schema-validated candidate selection.')
	if event.node == 'listing_details':
		return 'Matched the opened listing ID and workflow predicate against frozen DOM evidence.'
	return event.reason or event.status.replace('_', ' ')


def airbnb_live_event_payload(event: HybridEvent, *, index: int, live_run_id: str) -> dict[str, Any]:
	"""Project one trace event onto an allowlisted, browser-safe UI surface."""

	state = event.state
	payload: dict[str, Any] = {
		'index': index,
		'node': event.node,
		'label': EVENT_LABELS.get(event.node, event.node.replace('_', ' ').title()),
		'mode': event.mode.value,
		'status': event.status,
		'duration_ms': event.duration_ms,
		'detail': _live_event_detail(event)[:400],
		'state': None,
	}
	if state:
		payload['state'] = {
			'url': state.url,
			'title': state.title,
			'dom_signature': state.semantic_dom_sha256[:12] if state.semantic_dom_sha256 else None,
			'selectors': state.selector_count,
			'screenshot_url': (
				f'/api/airbnb/live/screenshot?run_id={live_run_id}&event={index}' if state.screenshot_path else None
			),
		}
	return payload


class AirbnbLiveRunStore:
	"""Thread-safe, single-run observer state for the loopback demo."""

	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._run: dict[str, Any] | None = None

	def start(self, live_run_id: str, task: str) -> None:
		with self._lock:
			self._run = {
				'run_id': live_run_id,
				'task': task,
				'status': 'running',
				'events': [
					{
						'index': 0,
						'node': 'task_received',
						'label': EVENT_LABELS['task_received'],
						'mode': 'agent',
						'status': 'executed',
						'duration_ms': 0,
						'detail': 'Sent the request to the closed workflow router.',
						'state': None,
					}
				],
				'screenshots': {},
				'result': None,
				'error': None,
			}

	def add_event(self, live_run_id: str, event: HybridEvent) -> None:
		with self._lock:
			if not self._run or self._run['run_id'] != live_run_id:
				return
			index = len(self._run['events'])
			self._run['events'].append(airbnb_live_event_payload(event, index=index, live_run_id=live_run_id))
			if event.state and event.state.screenshot_path:
				self._run['screenshots'][index] = Path(event.state.screenshot_path)

	def finish(
		self,
		live_run_id: str,
		*,
		status: str,
		result: dict[str, Any] | None = None,
		error: str | None = None,
	) -> None:
		with self._lock:
			if not self._run or self._run['run_id'] != live_run_id:
				return
			self._run['status'] = status
			self._run['result'] = result
			self._run['error'] = error

	def snapshot(self, live_run_id: str) -> dict[str, Any] | None:
		with self._lock:
			if not self._run or self._run['run_id'] != live_run_id:
				return None
			return {key: deepcopy(value) for key, value in self._run.items() if key != 'screenshots'}

	def screenshot(self, live_run_id: str, event_index: int) -> Path | None:
		with self._lock:
			if not self._run or self._run['run_id'] != live_run_id:
				return None
			path = self._run['screenshots'].get(event_index)
			return path if path and path.is_file() else None


def airbnb_demo_report_payload(report: AirbnbHybridReport) -> dict[str, Any]:
	"""Expose the small result surface needed by the UI, not the private raw trace."""

	selected_candidate = next(
		(candidate for candidate in report.candidates if report.selected and candidate.listing_id == report.selected.listing_id),
		None,
	)
	return {
		'status': report.status.value,
		'run_id': report.run_id,
		'task': report.task,
		'workflow_id': report.workflow_id.value if report.workflow_id else None,
		'parameters': report.parameters.model_dump(mode='json'),
		'selected': {
			'listing_id': report.selected.listing_id,
			'title': selected_candidate.title if selected_candidate else report.listing_heading,
			'total_price': report.selected.total_price,
			'currency': report.selected.currency,
			'rating': report.selected.rating,
			'review_count': report.selected.review_count,
		}
		if report.selected
		else None,
		'listing_url': report.listing_url,
		'listing_heading': report.listing_heading,
		'candidate_count': len(report.candidates),
		'model_calls': report.model_calls,
		'model_cost': report.model_cost,
		'route_confidence': report.routing.confidence if report.routing else None,
		'assumptions': report.routing.assumptions if report.routing else [],
		'reason': report.reason,
	}


def airbnb_routing_error_payload(route: AirbnbTaskRoute) -> dict[str, Any]:
	return {
		'status': 'needs_input',
		'detail': route.reason,
		'routing': {
			'disposition': route.disposition.value,
			'workflow_id': route.workflow_id.value if route.workflow_id else None,
			'confidence': route.confidence,
			'assumptions': route.assumptions,
		},
	}


class AirbnbBrowserWorker:
	"""Run browser work on one event loop, serialized by the HTTP server."""

	def __init__(
		self,
		*,
		llm: Any,
		headless: bool,
		trace_dir: Path,
		linger_seconds: float,
	):
		self.llm = llm
		self.headless = headless
		self.trace_dir = trace_dir
		self.linger_seconds = linger_seconds
		self.loop = asyncio.new_event_loop()
		self.thread = threading.Thread(target=self._serve_loop, name='airbnb-browser-worker', daemon=True)
		self.thread.start()

	def _serve_loop(self) -> None:
		asyncio.set_event_loop(self.loop)
		self.loop.run_forever()

	async def _run(self, task: str, event_callback) -> AirbnbHybridReport:
		# run_airbnb_task performs routing first and creates Chromium only for a matched route.
		return await run_airbnb_task(
			task,
			llm=self.llm,
			trace_dir=self.trace_dir,
			headless=self.headless,
			linger_seconds=self.linger_seconds,
			event_callback=event_callback,
		)

	def run(self, task: str, event_callback) -> AirbnbHybridReport:
		return asyncio.run_coroutine_threadsafe(self._run(task, event_callback), self.loop).result(timeout=180)

	def close(self) -> None:
		if self.loop.is_running():
			self.loop.call_soon_threadsafe(self.loop.stop)
			self.thread.join(timeout=5)


class AirbnbDemoServer(ThreadingHTTPServer):
	def __init__(
		self,
		server_address: tuple[str, int],
		*,
		llm: Any,
		headless: bool,
		trace_dir: Path,
		linger_seconds: float,
	):
		super().__init__(server_address, AirbnbDemoHandler)
		self.run_lock = threading.Lock()
		self.live_runs = AirbnbLiveRunStore()
		self.browser_worker = AirbnbBrowserWorker(
			llm=llm,
			headless=headless,
			trace_dir=trace_dir,
			linger_seconds=linger_seconds,
		)

	def server_close(self) -> None:
		self.browser_worker.close()
		super().server_close()


class AirbnbDemoHandler(SimpleHTTPRequestHandler):
	server: AirbnbDemoServer

	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

	def end_headers(self) -> None:
		self.send_header('Cache-Control', 'no-store')
		self.send_header('X-Content-Type-Options', 'nosniff')
		super().end_headers()

	def do_GET(self) -> None:
		parsed = urlsplit(self.path)
		if parsed.path == '/api/airbnb/live':
			query = parse_qs(parsed.query)
			run_id = query.get('run_id', [''])[0]
			snapshot = self.server.live_runs.snapshot(run_id)
			self._json(HTTPStatus.OK, snapshot) if snapshot else self._json(HTTPStatus.NOT_FOUND, {'detail': 'run not found'})
			return
		if parsed.path == '/api/airbnb/live/screenshot':
			query = parse_qs(parsed.query)
			run_id = query.get('run_id', [''])[0]
			try:
				event_index = int(query.get('event', ['-1'])[0])
			except ValueError:
				event_index = -1
			path = self.server.live_runs.screenshot(run_id, event_index)
			if not path:
				self._json(HTTPStatus.NOT_FOUND, {'detail': 'screenshot not found'})
				return
			body = path.read_bytes()
			self.send_response(HTTPStatus.OK.value)
			self.send_header('Content-Type', 'image/png')
			self.send_header('Content-Length', str(len(body)))
			self.end_headers()
			self.wfile.write(body)
			return
		if parsed.path == '/':
			self.path = '/airbnb.html'
		super().do_GET()

	def do_POST(self) -> None:
		if urlsplit(self.path).path != '/api/airbnb/run':
			self._json(HTTPStatus.NOT_FOUND, {'detail': 'unknown endpoint'})
			return
		if not self._same_origin_request():
			self._json(HTTPStatus.FORBIDDEN, {'detail': 'loopback same-origin requests only'})
			return
		try:
			content_length = int(self.headers.get('Content-Length', '0'))
		except ValueError:
			content_length = 0
		if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
			self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {'detail': 'invalid request size'})
			return
		try:
			request = AirbnbDemoRequest.model_validate_json(self.rfile.read(content_length))
		except ValidationError:
			self._json(HTTPStatus.BAD_REQUEST, {'detail': 'task must be 1–1000 characters'})
			return
		if not self.server.run_lock.acquire(blocking=False):
			self._json(HTTPStatus.CONFLICT, {'detail': 'another Airbnb run is active'})
			return
		live_run_id = request.client_run_id or str(uuid4())
		self.server.live_runs.start(live_run_id, request.task)
		try:
			report = self.server.browser_worker.run(
				request.task,
				lambda event: self.server.live_runs.add_event(live_run_id, event),
			)
			status = HTTPStatus.OK if report.status.value == 'completed' else HTTPStatus.UNPROCESSABLE_ENTITY
			print(f'[airbnb-demo] run={report.run_id} status={report.status.value} reason={report.reason or "none"}')
			payload = airbnb_demo_report_payload(report)
			payload['live_run_id'] = live_run_id
			self.server.live_runs.finish(live_run_id, status=report.status.value, result=payload, error=report.reason)
			self._json(status, payload)
		except TaskRoutingError as exc:
			print(f'[airbnb-demo] route={exc.route.disposition.value} reason={exc.route.reason}')
			payload = airbnb_routing_error_payload(exc.route)
			payload['live_run_id'] = live_run_id
			self.server.live_runs.finish(live_run_id, status='needs_input', result=payload, error=exc.route.reason)
			self._json(HTTPStatus.UNPROCESSABLE_ENTITY, payload)
		except Exception as exc:
			print(f'[airbnb-demo] failed: {type(exc).__name__}: {exc}')
			self.server.live_runs.finish(live_run_id, status='failed', error='the local browser runner failed')
			self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {'detail': 'the local browser runner failed'})
		finally:
			self.server.run_lock.release()

	def _same_origin_request(self) -> bool:
		origin = self.headers.get('Origin')
		if not origin:
			return True
		parsed = urlsplit(origin)
		return parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'}

	def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
		body = json.dumps(payload).encode()
		self.send_response(status.value)
		self.send_header('Content-Type', 'application/json; charset=utf-8')
		self.send_header('Content-Length', str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def log_message(self, format: str, *args: Any) -> None:
		print(f'[airbnb-demo] {self.address_string()} {format % args}')


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost'])
	parser.add_argument('--port', type=int, default=8766)
	parser.add_argument('--provider', choices=['browser-use', 'anthropic'], default='anthropic')
	parser.add_argument('--model')
	parser.add_argument('--env-file', type=Path)
	parser.add_argument('--trace-dir', type=Path, default=Path('./tmp/airbnb-demo-traces'))
	parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument('--linger-seconds', type=float, default=4)
	args = parser.parse_args()
	if args.env_file:
		load_dotenv(args.env_file, override=False)
	server = AirbnbDemoServer(
		(args.host, args.port),
		llm=build_airbnb_llm(args.provider, args.model),
		headless=args.headless,
		trace_dir=args.trace_dir.expanduser().resolve(),
		linger_seconds=max(0, args.linger_seconds),
	)
	print(f'Airbnb task demo: http://{args.host}:{args.port}/')
	print('A visible Chromium window opens only after the task maps to a supported workflow.')
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


if __name__ == '__main__':
	main()
