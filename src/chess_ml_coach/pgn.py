from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote, urlparse

import chess
import chess.pgn
import pandas as pd

LOGGER = logging.getLogger(__name__)

GAME_COLUMNS = [
    "game_id",
    "game_date",
    "white",
    "black",
    "white_rating",
    "black_rating",
    "result",
    "time_control",
    "rated",
    "eco",
    "opening",
    "source_url",
]
MOVE_COLUMNS = [
    "game_id",
    "ply",
    "fullmove_number",
    "color",
    "fen_before",
    "fen_after",
    "san",
    "uci",
    "is_user_move",
    "clock_seconds",
]


class ArtifactSchemaError(RuntimeError):
    pass


def _parse_int(value: str | None) -> int | None:
    if value in (None, "", "?"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "rated"}:
        return True
    if lowered in {"false", "0", "no", "unrated"}:
        return False
    return None


def _game_date(headers: chess.pgn.Headers) -> pd.Timestamp:
    raw = headers.get("UTCDate") or headers.get("Date")
    if not raw or raw == "????.??.??":
        return pd.NaT
    return pd.to_datetime(raw.replace(".", "-"), errors="coerce")


def _opening_name(headers: chess.pgn.Headers) -> str | None:
    explicit = headers.get("Opening")
    if explicit:
        return str(explicit)
    eco_url = headers.get("ECOUrl")
    if not eco_url:
        return None
    slug = Path(urlparse(str(eco_url)).path.rstrip("/")).name
    if not slug:
        return None
    return unquote(slug).replace("-", " ")


def _fallback_game_id(game: chess.pgn.Game) -> str:
    board = game.board()
    moves: list[str] = []
    for move in game.mainline_moves():
        moves.append(move.uci())
        board.push(move)
    header_bits = "|".join(
        str(game.headers.get(key, ""))
        for key in ("Date", "White", "Black", "Result", "Round")
    )
    canonical = header_bits + "|" + " ".join(moves)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _game_id(game: chess.pgn.Game) -> str:
    site = str(game.headers.get("Site", ""))
    if "chess.com/game/" in site:
        return site
    return _fallback_game_id(game)


def _iter_games(handle: TextIO):
    while True:
        position = handle.tell()
        try:
            game = chess.pgn.read_game(handle)
        except Exception as exc:  # noqa: BLE001 - malformed third-party PGNs should not kill the corpus
            LOGGER.warning("Skipping unreadable PGN game: %s", exc)
            if handle.tell() == position:
                handle.readline()
            continue
        if game is None:
            break
        yield game


def parse_pgn_file(path: Path, username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_rows: list[dict] = []
    move_rows: list[dict] = []
    username_lower = username.casefold()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for game in _iter_games(handle):
            headers = game.headers
            game_id = _game_id(game)
            white = str(headers.get("White", ""))
            black = str(headers.get("Black", ""))
            game_rows.append(
                {
                    "game_id": game_id,
                    "game_date": _game_date(headers),
                    "white": white,
                    "black": black,
                    "white_rating": _parse_int(headers.get("WhiteElo")),
                    "black_rating": _parse_int(headers.get("BlackElo")),
                    "result": headers.get("Result"),
                    "time_control": headers.get("TimeControl"),
                    "rated": _parse_bool(headers.get("Rated")),
                    "eco": headers.get("ECO"),
                    "opening": _opening_name(headers),
                    "source_url": headers.get("Link") or headers.get("Site"),
                }
            )

            board = game.board()
            for ply, node in enumerate(game.mainline(), start=1):
                move = node.move
                mover = board.turn
                fen_before = board.fen()
                san = board.san(move)
                uci = move.uci()
                fullmove_number = board.fullmove_number
                board.push(move)
                mover_name = white if mover == chess.WHITE else black
                clock = node.clock()
                move_rows.append(
                    {
                        "game_id": game_id,
                        "ply": ply,
                        "fullmove_number": fullmove_number,
                        "color": "white" if mover == chess.WHITE else "black",
                        "fen_before": fen_before,
                        "fen_after": board.fen(),
                        "san": san,
                        "uci": uci,
                        "is_user_move": mover_name.casefold() == username_lower,
                        "clock_seconds": float(clock) if clock is not None else None,
                    }
                )

    return pd.DataFrame(game_rows, columns=GAME_COLUMNS), pd.DataFrame(move_rows, columns=MOVE_COLUMNS)


def _validate_columns(frame: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ArtifactSchemaError(f"Invalid artifact schema at {path}: missing {', '.join(missing)}")


def write_normalized(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    processed_dir: Path,
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    games_path = processed_dir / "games.parquet"
    moves_path = processed_dir / "moves.parquet"
    _validate_columns(games, GAME_COLUMNS, games_path)
    _validate_columns(moves, MOVE_COLUMNS, moves_path)
    game_sort = games.sort_values(["game_date", "game_id"], na_position="last").reset_index(drop=True)
    move_sort = moves.sort_values(["game_id", "ply"]).reset_index(drop=True)
    game_sort.to_parquet(games_path, index=False)
    move_sort.to_parquet(moves_path, index=False)
    return games_path, moves_path


def read_normalized(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_path = processed_dir / "games.parquet"
    moves_path = processed_dir / "moves.parquet"
    try:
        games = pd.read_parquet(games_path)
        moves = pd.read_parquet(moves_path)
    except Exception as exc:
        raise ArtifactSchemaError(f"Could not read normalized artifacts in {processed_dir}: {exc}") from exc
    _validate_columns(games, GAME_COLUMNS, games_path)
    _validate_columns(moves, MOVE_COLUMNS, moves_path)
    return games, moves
