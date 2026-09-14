from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..move_quality import display_loss_pawns, stored_quality_reason


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _puzzle_payload(row: sqlite3.Row) -> dict[str, object]:
    attempts = int(row["attempts"])
    correct_attempts = int(row["correct_attempts"])
    cpl = int(row["cpl"])
    quality = str(row["quality"])
    quality_reason = stored_quality_reason(quality, cpl)
    return {
        "puzzle_id": str(row["puzzle_id"]),
        "fen": str(row["fen_before"]),
        "orientation": str(row["color"]),
        "game": str(row["game_label"]),
        "move": str(row["move_label"]),
        "your_move_san": str(row["your_move_san"]),
        "your_move_uci": str(row["your_move_uci"]),
        "best_move_san": str(row["best_move_san"]),
        "best_move_uci": str(row["best_move_uci"]),
        "evaluation_loss_pawns": display_loss_pawns(cpl, quality_reason),
        "quality": quality,
        "quality_reason": quality_reason,
        "opening": str(row["opening"]),
        "eco": str(row["eco"]),
        "phase": str(row["game_phase"]),
        "source_url": str(row["source_url"]) or None,
        "motif": str(row["motif"]),
        "difficulty": int(row["difficulty"]),
        "attempts": attempts,
        "correct_attempts": correct_attempts,
        "accuracy": (correct_attempts / attempts) if attempts else None,
        "consecutive_correct": int(row["consecutive_correct"]),
        "next_review_at": str(row["next_review_at"]),
        "mastered": bool(row["mastered"]),
    }


def list_puzzles(
    path: Path,
    *,
    quality: str | None = None,
    motif: str | None = None,
    opening: str | None = None,
    reviewed: bool | None = None,
    mastered: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, object]], int]:
    clauses = ["active = 1"]
    params: list[object] = []
    if quality:
        clauses.append("quality = ?")
        params.append(quality)
    if motif:
        clauses.append("motif = ?")
        params.append(motif)
    if opening:
        clauses.append("opening = ?")
        params.append(opening)
    if reviewed is not None:
        clauses.append("attempts > 0" if reviewed else "attempts = 0")
    if mastered is not None:
        clauses.append("mastered = ?")
        params.append(int(mastered))
    where = " AND ".join(clauses)

    with _connect(path) as connection:
        total = int(
            connection.execute(
                f"SELECT COUNT(*) FROM puzzles WHERE {where}",
                params,
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            SELECT *
            FROM puzzles
            WHERE {where}
            ORDER BY attempts ASC, difficulty DESC, next_review_at ASC, puzzle_id ASC
            LIMIT ? OFFSET ?
            """,
            [*params, int(limit), int(offset)],
        ).fetchall()
    return [_puzzle_payload(row) for row in rows], total


def get_puzzle(path: Path, puzzle_id: str) -> dict[str, object] | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM puzzles WHERE puzzle_id = ? AND active = 1",
            (puzzle_id,),
        ).fetchone()
    return _puzzle_payload(row) if row is not None else None


def daily_reviews(path: Path, *, days: int = 90) -> list[dict[str, object]]:
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT substr(r.reviewed_at, 1, 10) AS day,
                   COUNT(*) AS reviews,
                   SUM(r.correct) AS correct
            FROM reviews AS r
            JOIN puzzles AS p ON p.puzzle_id = r.puzzle_id
            WHERE p.active = 1 AND r.reviewed_at >= ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (since,),
        ).fetchall()
    result = []
    for row in rows:
        reviews = int(row["reviews"] or 0)
        correct = int(row["correct"] or 0)
        result.append(
            {
                "date": str(row["day"]),
                "reviews": reviews,
                "correct": correct,
                "accuracy": (correct / reviews) if reviews else 0.0,
            }
        )
    return result
