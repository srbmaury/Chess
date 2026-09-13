import json
from pathlib import Path

import httpx

from chess_ml_coach.chesscom import ChessComClient, canonical_game_id, merge_games, sync_games
from chess_ml_coach.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def test_game_url_is_primary_stable_id():
    game = {"url": "https://www.chess.com/game/live/123", "pgn": "PGN"}
    assert canonical_game_id(game) == game["url"]


def test_fallback_id_is_stable_for_same_game_without_url():
    game = {"pgn": "[White \"srbmaury\"]\n\n1. e4 e5 1-0"}
    assert canonical_game_id(game) == canonical_game_id(dict(game))


def test_merge_games_is_idempotent():
    old = [{"url": "https://www.chess.com/game/live/1", "pgn": "A"}]
    incoming = old + [{"url": "https://www.chess.com/game/live/2", "pgn": "B"}]
    merged = merge_games(old, incoming)
    assert [g["url"] for g in merged] == [old[0]["url"], incoming[1]["url"]]
    assert merge_games(merged, incoming) == merged


def test_client_parses_archives_and_month_fixture():
    archives = json.loads((FIXTURES / "archives.json").read_text())
    month = json.loads((FIXTURES / "month.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json=archives)
        return httpx.Response(200, json=month)

    client = ChessComClient(httpx.Client(transport=httpx.MockTransport(handler)))
    urls = client.archive_urls("srbmaury")
    assert len(urls) == 2
    assert len(client.games_for_archive(urls[0])) == 2


def test_missing_month_archive_is_non_fatal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Data not found"})

    client = ChessComClient(httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.games_for_archive("https://api.chess.com/pub/player/srbmaury/games/2025/05") is None


def test_sync_skips_missing_archive_and_continues(tmp_path: Path):
    month = json.loads((FIXTURES / "month.json").read_text())
    missing_url = "https://api.chess.com/pub/player/srbmaury/games/2025/05"
    good_url = "https://api.chess.com/pub/player/srbmaury/games/2025/06"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json={"archives": [missing_url, good_url]})
        if request.url.path.endswith("/2025/05"):
            return httpx.Response(404, json={"message": "Data not found"})
        return httpx.Response(200, json=month)

    client = ChessComClient(httpx.Client(transport=httpx.MockTransport(handler)))
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")

    result = sync_games(client, settings)

    assert result.downloaded == 2
    assert result.total == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["archives_completed"] == [good_url]


def test_sync_is_idempotent(tmp_path: Path):
    archives = json.loads((FIXTURES / "archives.json").read_text())
    month = json.loads((FIXTURES / "month.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json=archives)
        return httpx.Response(200, json=month)

    client = ChessComClient(httpx.Client(transport=httpx.MockTransport(handler)))
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    first = sync_games(client, settings)
    second = sync_games(client, settings)
    assert first.downloaded == 2
    assert second.downloaded == 0
    assert first.total == second.total == 2
    text = first.pgn_path.read_text()
    assert text.count("https://www.chess.com/game/live/1") == 1
    assert text.count("https://www.chess.com/game/live/2") == 1
