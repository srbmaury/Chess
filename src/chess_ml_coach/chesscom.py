from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time

import httpx

from .config import Settings

API = "https://api.chess.com/pub"


class ChessComError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncResult:
    downloaded: int
    existing: int
    total: int
    pgn_path: Path
    manifest_path: Path


class ChessComClient:
    def __init__(self, http: httpx.Client | None = None, retries: int = 3):
        self.http = http or httpx.Client(
            timeout=30,
            headers={"User-Agent": "chess-ml-coach/0.1 (username: srbmaury)"},
        )
        self.retries = retries

    def _json(self, url: str) -> dict:
        for attempt in range(self.retries + 1):
            response = self.http.get(url)
            if response.status_code < 400:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                time.sleep(2**attempt)
                continue
            raise ChessComError(f"Chess.com request failed: {response.status_code} {url}")
        raise AssertionError("unreachable")

    def archive_urls(self, username: str) -> list[str]:
        return self._json(f"{API}/player/{username}/games/archives").get("archives", [])

    def games_for_archive(self, archive_url: str) -> list[dict]:
        return self._json(archive_url).get("games", [])


def canonical_game_id(game: dict) -> str:
    url = game.get("url")
    if url:
        return str(url)
    return sha256(str(game.get("pgn", "")).encode("utf-8")).hexdigest()


def merge_games(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_id = {canonical_game_id(game): game for game in existing if game.get("pgn")}
    for game in incoming:
        if game.get("pgn"):
            by_id.setdefault(canonical_game_id(game), game)
    return [by_id[key] for key in sorted(by_id)]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _load_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ChessComError(f"Invalid game cache format: {path}")
    return data


def _persist_sync(raw_dir: Path, username: str, games: list[dict], archives_done: list[str]) -> tuple[Path, Path]:
    games_path = raw_dir / "games.json"
    pgn_path = raw_dir / f"{username}_all_games.pgn"
    manifest_path = raw_dir / "sync_manifest.json"

    _atomic_write_text(games_path, json.dumps(games, indent=2, sort_keys=True))
    pgn_text = "\n\n".join(str(game["pgn"]).rstrip() for game in games if game.get("pgn"))
    if pgn_text:
        pgn_text += "\n"
    _atomic_write_text(pgn_path, pgn_text)
    manifest = {
        "username": username,
        "archives_completed": archives_done,
        "game_count": len(games),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return pgn_path, manifest_path


def sync_games(client: ChessComClient, settings: Settings) -> SyncResult:
    raw_dir = settings.data_dir / "raw"
    games_path = raw_dir / "games.json"
    existing_games = _load_games(games_path)
    existing_count = len(existing_games)
    games = list(existing_games)
    archives_done: list[str] = []

    for archive_url in client.archive_urls(settings.username):
        games = merge_games(games, client.games_for_archive(archive_url))
        archives_done.append(archive_url)
        pgn_path, manifest_path = _persist_sync(raw_dir, settings.username, games, archives_done)

    if not archives_done:
        pgn_path, manifest_path = _persist_sync(raw_dir, settings.username, games, archives_done)

    return SyncResult(
        downloaded=max(0, len(games) - existing_count),
        existing=existing_count,
        total=len(games),
        pgn_path=pgn_path,
        manifest_path=manifest_path,
    )
