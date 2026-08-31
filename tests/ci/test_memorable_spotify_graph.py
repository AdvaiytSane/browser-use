import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.integrations.memorable.spotify_demo_server import SpotifyDemoRequest
from examples.integrations.memorable.spotify_graph import (
	SpotifyGoal,
	SpotifyTaskRouter,
	compile_spotify_graph,
	search_url_matches_artist,
	spotify_graph_template,
)


@pytest.mark.parametrize(
	('task', 'artist', 'goal', 'rank', 'path'),
	[
		(
			'Search for “Daft Punk,” open the verified artist result, and identify the first visible track under Popular.',
			'Daft Punk',
			SpotifyGoal.POPULAR_TRACK,
			1,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track'],
		),
		(
			'Open the verified artist result for Radiohead and stop there',
			'Radiohead',
			SpotifyGoal.CANONICAL_ARTIST,
			None,
			['spotify_home', 'search_results', 'canonical_artist'],
		),
		(
			'Search Spotify for Björk and identify the third popular track',
			'Björk',
			SpotifyGoal.POPULAR_TRACK,
			3,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track'],
		),
		(
			'navigate to Portraits of Tracy and play the best song',
			'Portraits of Tracy',
			SpotifyGoal.PLAY_TRACK,
			1,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track', 'play_track'],
		),
		(
			'Search Spotify for Daft Punk and play the third track',
			'Daft Punk',
			SpotifyGoal.PLAY_TRACK,
			3,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track', 'play_track'],
		),
		(
			'Can you find me the best song by Portraits of Tracy',
			'Portraits of Tracy',
			SpotifyGoal.POPULAR_TRACK,
			1,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track'],
		),
		(
			'What is the third track by Björk?',
			'Björk',
			SpotifyGoal.POPULAR_TRACK,
			3,
			['spotify_home', 'search_results', 'canonical_artist', 'popular_track'],
		),
	],
)
def test_routes_task_to_parameterized_graph_path(
	task: str,
	artist: str,
	goal: SpotifyGoal,
	rank: int | None,
	path: list[str],
) -> None:
	intent = SpotifyTaskRouter().route(task, spotify_graph_template())

	assert intent.artist == artist
	assert intent.goal_node == goal
	assert intent.track_rank == rank
	assert intent.path == path


def _write_trace(root: Path, artist: str, artist_id: str, track_id: str) -> None:
	payload = {
		'trace_id': artist_id,
		'created_at': '2026-08-31T00:00:00+00:00',
		'artist': artist,
		'terminal_node': 'popular_track',
		'search_locator': {'ax_role': 'combobox'},
		'artist_result_locator': {'ax_role': 'link'},
		'artist_url': f'https://open.spotify.com/artist/{artist_id}',
		'popular_tracks': [{'rank': 1, 'name': f'{artist} track', 'url': f'https://open.spotify.com/track/{track_id}'}],
		'model_calls': 0,
		'verified': True,
	}
	(root / f'{artist_id}.json').write_text(json.dumps(payload))


def test_compiles_distinct_traces_without_memorizing_artist_values(tmp_path: Path) -> None:
	_write_trace(tmp_path, 'Daft Punk', 'daft', 'track1')
	_write_trace(tmp_path, 'Radiohead', 'radiohead', 'track2')
	_write_trace(tmp_path, 'Björk', 'bjork', 'track3')

	graph = compile_spotify_graph(tmp_path)

	assert graph.training['trace_count'] == 3
	assert graph.training['distinct_artists'] == 3
	assert graph.training['captured_values_used_as_runtime_locators'] is False
	serialized_locators = json.dumps([edge.locator.model_dump() for edge in graph.edges if edge.locator])
	assert 'Daft Punk' not in serialized_locators
	assert graph.edges[1].locator is not None
	assert graph.edges[1].locator.dynamic == {'ax_name': 'artist'}


def test_compiler_requires_three_distinct_verified_artists(tmp_path: Path) -> None:
	_write_trace(tmp_path, 'Daft Punk', 'daft', 'track1')
	_write_trace(tmp_path, 'Radiohead', 'radiohead', 'track2')

	with pytest.raises(ValueError, match='need 3 distinct verified artists'):
		compile_spotify_graph(tmp_path)


def test_demo_server_request_is_narrow_and_bounded() -> None:
	assert SpotifyDemoRequest(task='Open the artist result for Radiohead').task.endswith('Radiohead')

	with pytest.raises(ValidationError):
		SpotifyDemoRequest.model_validate({'task': 'x', 'headless': False})
	with pytest.raises(ValidationError):
		SpotifyDemoRequest(task='x' * 501)


def test_search_postcondition_rejects_partial_spotify_query() -> None:
	assert search_url_matches_artist('https://open.spotify.com/search/Portraits%20of%20Tracy', 'Portraits of Tracy')
	assert not search_url_matches_artist('https://open.spotify.com/search/Portraits%20of%20T', 'Portraits of Tracy')


def test_playback_is_an_optional_terminal_after_extraction() -> None:
	graph = spotify_graph_template()

	assert graph.path_to('popular_track')[-1].id == 'read_ranked_popular_track'
	assert graph.path_to('play_track')[-1].id == 'play_ranked_popular_track'
	assert graph.node('play_track').terminal is True
