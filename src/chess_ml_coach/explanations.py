from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import chess

from .config import Settings
from .engine import StockfishAdapter, _resolve_stockfish
from .training import StoredPuzzle

AnalyseFn = Callable[[chess.Board, int], dict[str, Any]]


def _iso(value: datetime | None = None) -> str:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=UTC)
    return resolved.astimezone(UTC).isoformat()


def puzzle_fingerprint(puzzle: StoredPuzzle) -> str:
    raw = f"{puzzle.fen_before}|{puzzle.best_move_uci}|{puzzle.your_move_uci}"
    return sha256(raw.encode()).hexdigest()[:24]


class ExplanationStore:
    """Caches engine-grounded puzzle explanations in the training database."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS puzzle_explanations (
                    puzzle_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (puzzle_id) REFERENCES puzzles(puzzle_id) ON DELETE CASCADE
                )
                """
            )

    def get(
        self,
        puzzle_id: str,
        *,
        fingerprint: str,
        depth: int,
    ) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM puzzle_explanations
                WHERE puzzle_id = ? AND fingerprint = ? AND depth = ?
                """,
                (puzzle_id, fingerprint, int(depth)),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return payload if isinstance(payload, dict) else None

    def save(
        self,
        puzzle_id: str,
        *,
        fingerprint: str,
        depth: int,
        payload: dict[str, object],
        now: datetime | None = None,
    ) -> None:
        timestamp = _iso(now)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO puzzle_explanations (
                    puzzle_id, fingerprint, depth, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(puzzle_id) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    depth = excluded.depth,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (puzzle_id, fingerprint, int(depth), encoded, timestamp, timestamp),
            )


def _san_line(board: chess.Board, moves: list[chess.Move], *, limit: int = 6) -> list[str]:
    current = board.copy(stack=False)
    line: list[str] = []
    for move in moves[:limit]:
        if move not in current.legal_moves:
            break
        line.append(current.san(move))
        current.push(move)
    return line


def _best_move_facts(board: chess.Board, puzzle: StoredPuzzle) -> tuple[str, str]:
    try:
        move = chess.Move.from_uci(puzzle.best_move_uci)
    except ValueError:
        return (
            "Best continuation",
            "Stockfish evaluates this move as the strongest continuation in the position.",
        )

    if move not in board.legal_moves:
        return (
            "Best continuation",
            "Stockfish evaluates this move as the strongest continuation in the position.",
        )

    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)
    after = board.copy(stack=False)
    after.push(move)

    if after.is_checkmate():
        return "Checkmate", "This move ends the game immediately with checkmate."
    if gives_check and is_capture:
        return (
            "Forcing check",
            "This move captures with check, forcing the opponent to answer the king threat before doing anything else.",
        )
    if gives_check:
        return (
            "Forcing check",
            "This move gives check, forcing an immediate king response and keeping the initiative.",
        )
    if is_capture:
        return (
            "Tactical capture",
            "This move makes the strongest capture in the position and leads to the best concrete continuation.",
        )
    if move.promotion:
        return (
            "Promotion",
            "This move promotes a pawn and creates the strongest immediate material threat.",
        )
    if puzzle.motif and puzzle.motif != "positional / calculation":
        return (
            puzzle.motif.replace("_", " ").title(),
            "Stockfish confirms this tactical idea as the strongest continuation in the position.",
        )
    return (
        "Best continuation",
        "Stockfish evaluates this move as the strongest continuation in the position.",
    )


def _why_original_move_was_worse(puzzle: StoredPuzzle) -> str:
    if puzzle.eval_loss_pawns >= 50:
        return (
            f"Your game move {puzzle.your_move_san} missed this continuation and caused "
            "a decisive, mate-level evaluation swing."
        )
    return (
        f"Your game move {puzzle.your_move_san} missed this continuation and lost about "
        f"{puzzle.eval_loss_pawns:.2f} pawns of evaluation compared with the best move."
    )


def _engine_disagreement_fallback(
    board: chess.Board,
    puzzle: StoredPuzzle,
    *,
    depth: int,
    preferred_move: chess.Move,
    board_reason: str,
    why_worse: str,
) -> dict[str, object]:
    if preferred_move in board.legal_moves:
        preferred = board.san(preferred_move)
    else:
        preferred = preferred_move.uci()
    return {
        "idea": "Analysis changed",
        "why": (
            f"Stockfish at depth {depth} now prefers {preferred} rather than the stored "
            f"best move {puzzle.best_move_san}. Refresh Analyze → Features → Puzzles "
            f"before relying on an engine-grounded explanation. Board-only note: {board_reason}"
        ),
        "best_line": [puzzle.best_move_san],
        "why_your_move_was_worse": why_worse,
        "engine_grounded": False,
        "depth": depth,
        "cached": False,
    }


class PuzzleExplanationService:
    def __init__(
        self,
        settings: Settings,
        cache: ExplanationStore,
        *,
        analyse: AnalyseFn | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._injected_analyse = analyse

    def _analyse(self, board: chess.Board, depth: int) -> dict[str, Any]:
        if self._injected_analyse is not None:
            return self._injected_analyse(board, depth)
        path = _resolve_stockfish(self.settings.stockfish_path)
        adapter = StockfishAdapter(path)
        try:
            return adapter.analyse(board, depth)
        finally:
            adapter.close()

    def explain(self, puzzle: StoredPuzzle) -> dict[str, object]:
        depth = int(self.settings.stockfish_depth)
        fingerprint = puzzle_fingerprint(puzzle)
        cached = self.cache.get(puzzle.puzzle_id, fingerprint=fingerprint, depth=depth)
        if cached is not None:
            return {**cached, "cached": True}

        board = chess.Board(puzzle.fen_before)
        idea, board_reason = _best_move_facts(board, puzzle)
        why_worse = _why_original_move_was_worse(puzzle)
        try:
            info = self._analyse(board.copy(stack=False), depth)
            pv = list(info.get("pv") or [])
            if pv and pv[0].uci() != puzzle.best_move_uci:
                return _engine_disagreement_fallback(
                    board,
                    puzzle,
                    depth=depth,
                    preferred_move=pv[0],
                    board_reason=board_reason,
                    why_worse=why_worse,
                )
            best_line = _san_line(board, pv)
            if not best_line:
                best_line = [puzzle.best_move_san]
            payload: dict[str, object] = {
                "idea": idea,
                "why": board_reason,
                "best_line": best_line,
                "why_your_move_was_worse": why_worse,
                "engine_grounded": True,
                "depth": depth,
            }
            self.cache.save(
                puzzle.puzzle_id,
                fingerprint=fingerprint,
                depth=depth,
                payload=payload,
            )
            return {**payload, "cached": False}
        except Exception:  # noqa: BLE001 - explanation failure must not break practice.
            return {
                "idea": idea,
                "why": (
                    "Stockfish is unavailable, so this explanation uses only the board "
                    f"position. {board_reason}"
                ),
                "best_line": [puzzle.best_move_san],
                "why_your_move_was_worse": why_worse,
                "engine_grounded": False,
                "depth": depth,
                "cached": False,
            }
