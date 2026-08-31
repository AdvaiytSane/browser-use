"""Loopback-only Replay Lab server with a live Spotify Browser Use endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from browser_use import BrowserProfile, BrowserSession
from browser_use.browser.profile import ViewportSize
from examples.integrations.memorable.spotify_graph import (
	SPOTIFY_ORIGIN,
	SpotifyGraphExecutor,
	SpotifyGraphReport,
	spotify_graph_template,
)

MAX_REQUEST_BYTES = 16 * 1024
STATIC_ROOT = Path(__file__).with_name('sites').resolve()
ARTIFACT_PATH = re.compile(r'^/api/spotify/artifacts/(?P<run_id>[0-9a-f-]{30,40})/(?P<filename>state-[0-9]{2}-[a-z0-9_-]+\.png)$')


def resolve_spotify_artifact(trace_dir: Path, request_path: str) -> Path | None:
	"""Resolve only generated state screenshots underneath the configured trace root."""
	match = ARTIFACT_PATH.fullmatch(urlsplit(request_path).path)
	if match is None:
		return None
	root = trace_dir.expanduser().resolve()
	target = (root / match.group('run_id') / match.group('filename')).resolve()
	try:
		target.relative_to(root)
	except ValueError:
		return None
	return target if target.is_file() else None


class SpotifyDemoRequest(BaseModel):
	model_config = ConfigDict(extra='forbid')

	task: str = Field(min_length=1, max_length=500)


class SpotifyBrowserWorker:
	"""Own the browser loop and retain only login-required sessions."""

	def __init__(self, *, headless: bool, profile_dir: Path, trace_dir: Path, linger_seconds: float):
		self.headless = headless
		self.profile_dir = profile_dir
		self.trace_dir = trace_dir
		self.linger_seconds = linger_seconds
		self.loop = asyncio.new_event_loop()
		self.session: BrowserSession | None = None
		self.thread = threading.Thread(target=self._serve_loop, name='spotify-browser-worker', daemon=True)
		self.thread.start()

	def _serve_loop(self) -> None:
		asyncio.set_event_loop(self.loop)
		self.loop.run_forever()

	async def _new_session(self) -> BrowserSession:
		profile = BrowserProfile(
			headless=self.headless,
			user_data_dir=self.profile_dir,
			window_size=ViewportSize(width=1280, height=800),
		)
		self.session = BrowserSession(browser_profile=profile)
		await self.session.start()
		return self.session

	async def _discard_session(self) -> None:
		if self.session is not None:
			try:
				await self.session.kill()
			finally:
				self.session = None

	async def _run(self, task: str) -> SpotifyGraphReport:
		# A retained window exists only to let the user finish Spotify login.
		# Closing it before the next run flushes that profile state to disk.
		await self._discard_session()
		session = await self._new_session()
		await session.navigate_to(SPOTIFY_ORIGIN)
		report = await SpotifyGraphExecutor(spotify_graph_template(), trace_dir=self.trace_dir).run(session, task)
		if self.linger_seconds > 0:
			await asyncio.sleep(self.linger_seconds)
		if not (report.status == 'needs_recovery' and report.reason and 'signed-in session' in report.reason):
			await self._discard_session()
		return report

	def run(self, task: str) -> SpotifyGraphReport:
		return asyncio.run_coroutine_threadsafe(self._run(task), self.loop).result(timeout=120)

	async def _close(self) -> None:
		await self._discard_session()

	def close(self) -> None:
		if self.loop.is_running():
			try:
				asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=15)
			except Exception:
				pass
			self.loop.call_soon_threadsafe(self.loop.stop)
			self.thread.join(timeout=5)


class SpotifyDemoServer(ThreadingHTTPServer):
	def __init__(
		self,
		server_address: tuple[str, int],
		*,
		headless: bool,
		trace_dir: Path,
		linger_seconds: float,
		profile_dir: Path,
	):
		super().__init__(server_address, SpotifyDemoHandler)
		self.headless = headless
		self.trace_dir = trace_dir
		self.linger_seconds = linger_seconds
		self.profile_dir = profile_dir
		self.run_lock = threading.Lock()
		self.browser_worker = SpotifyBrowserWorker(
			headless=headless,
			profile_dir=profile_dir,
			trace_dir=trace_dir,
			linger_seconds=linger_seconds,
		)

	def server_close(self) -> None:
		self.browser_worker.close()
		super().server_close()


class SpotifyDemoHandler(SimpleHTTPRequestHandler):
	server: SpotifyDemoServer

	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

	def end_headers(self) -> None:
		self.send_header('Cache-Control', 'no-store')
		self.send_header('X-Content-Type-Options', 'nosniff')
		super().end_headers()

	def do_GET(self) -> None:
		if urlsplit(self.path).path.startswith('/api/spotify/artifacts/'):
			artifact = resolve_spotify_artifact(self.server.trace_dir, self.path)
			if artifact is None:
				self.send_error(HTTPStatus.NOT_FOUND.value)
				return
			body = artifact.read_bytes()
			self.send_response(HTTPStatus.OK.value)
			self.send_header('Content-Type', 'image/png')
			self.send_header('Content-Length', str(len(body)))
			self.end_headers()
			self.wfile.write(body)
			return
		super().do_GET()

	def do_POST(self) -> None:
		if urlsplit(self.path).path != '/api/spotify/run':
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
			request = SpotifyDemoRequest.model_validate_json(self.rfile.read(content_length))
		except ValidationError:
			self._json(HTTPStatus.BAD_REQUEST, {'detail': 'task must be 1–500 characters'})
			return
		if not self.server.run_lock.acquire(blocking=False):
			self._json(HTTPStatus.CONFLICT, {'detail': 'another Spotify demo run is active'})
			return
		try:
			report = self.server.browser_worker.run(request.task)
			payload = report.model_dump(mode='json', exclude={'trace_path'})
			status = HTTPStatus.OK if report.status == 'completed' else HTTPStatus.UNPROCESSABLE_ENTITY
			print(f'[spotify-demo] run={report.run_id} status={report.status} reason={report.reason or "none"}')
			self._json(status, payload)
		except Exception:
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
		print(f'[spotify-demo] {self.address_string()} {format % args}')


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost'])
	parser.add_argument('--port', type=int, default=8765)
	parser.add_argument('--trace-dir', type=Path, default=Path('./tmp/spotify-demo-traces'))
	parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument('--linger-seconds', type=float, default=2.5)
	parser.add_argument('--profile-dir', type=Path, default=Path('./tmp/spotify-demo-profile'))
	args = parser.parse_args()
	server = SpotifyDemoServer(
		(args.host, args.port),
		headless=args.headless,
		trace_dir=args.trace_dir.expanduser().resolve(),
		linger_seconds=max(0, args.linger_seconds),
		profile_dir=args.profile_dir.expanduser().resolve(),
	)
	print(f'Replay Lab: http://{args.host}:{args.port}/')
	print('The green button reuses one visible Chromium window for Spotify.')
	print(f'Playback profile: {server.profile_dir} (sign in to Spotify once; later runs reuse it)')
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


if __name__ == '__main__':
	main()
