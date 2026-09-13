from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _iso(value: datetime | None = None) -> str:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=UTC)
    return resolved.astimezone(UTC).isoformat()


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
