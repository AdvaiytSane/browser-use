"""Loopback-only Replay Lab server with a live Spotify Browser Use endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from examples.integrations.memorable.spotify_graph import run_spotify_task, spotify_graph_template

MAX_REQUEST_BYTES = 16 * 1024
STATIC_ROOT = Path(__file__).with_name('sites').resolve()


class SpotifyDemoRequest(BaseModel):
	model_config = ConfigDict(extra='forbid')

	task: str = Field(min_length=1, max_length=500)


class SpotifyDemoServer(ThreadingHTTPServer):
	def __init__(
		self,
		server_address: tuple[str, int],
		*,
		headless: bool,
		trace_dir: Path,
		linger_seconds: float,
	):
		super().__init__(server_address, SpotifyDemoHandler)
		self.headless = headless
		self.trace_dir = trace_dir
		self.linger_seconds = linger_seconds
		self.run_lock = threading.Lock()


class SpotifyDemoHandler(SimpleHTTPRequestHandler):
	server: SpotifyDemoServer

	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

	def end_headers(self) -> None:
		self.send_header('Cache-Control', 'no-store')
		self.send_header('X-Content-Type-Options', 'nosniff')
		super().end_headers()

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
			report = asyncio.run(
				run_spotify_task(
					request.task,
					graph=spotify_graph_template(),
					trace_dir=self.server.trace_dir,
					headless=self.server.headless,
					linger_seconds=self.server.linger_seconds,
				)
			)
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
	args = parser.parse_args()
	server = SpotifyDemoServer(
		(args.host, args.port),
		headless=args.headless,
		trace_dir=args.trace_dir.expanduser().resolve(),
		linger_seconds=max(0, args.linger_seconds),
	)
	print(f'Replay Lab: http://{args.host}:{args.port}/')
	print('The green button now launches a visible, fresh Chromium window for Spotify.')
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


if __name__ == '__main__':
	main()
