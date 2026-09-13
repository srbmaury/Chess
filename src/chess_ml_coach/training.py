from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .puzzles import PuzzleSeed


@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int
    total: int


@dataclass(frozen=True)
class StoredPuzzle:
    puzzle_id: str
    game_id: str
    ply: int
    fen_before: str
    color: str
    game_label: str
    move_label: str
    your_move_san: str
    your_move_uci: str
    best_move_san: str
    best_move_uci: str
    cpl: int
    quality: str
    opening: str
    eco: str
    game_phase: str
    source_url: str
    motif: str
    difficulty: int
    attempts: int
    correct_attempts: int
    consecutive_correct: int
    last_reviewed_at: datetime | None
    next_review_at: datetime
    mastered: bool

    @property
    def eval_loss_pawns(self) -> float:
        return round(self.cpl / 100.0, 2)


@dataclass(frozen=True)
class ReviewResult:
    puzzle_id: str
    correct: bool
    next_interval_days: int
    consecutive_correct: int
    next_review_at: datetime
    mastered: bool


@dataclass(frozen=True)
class ProgressRow:
    label: str
    puzzles: int
    attempts: int
    correct: int
    accuracy: float | None


@dataclass(frozen=True)
class ProgressSummary:
    total_puzzles: int
    due_puzzles: int
    reviewed_puzzles: int
    mastered_puzzles: int
    total_reviews: int
    accuracy: float | None
    by_motif: list[ProgressRow]
    by_opening: list[ProgressRow]


def _ensure_utc(value: datetime | None = None) -> datetime:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=UTC)
    return resolved.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _ensure_utc(value).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return _ensure_utc(parsed)


def _interval_for(correct: bool, consecutive_correct: int) -> int:
    if not correct:
        return 1
    if consecutive_correct <= 1:
        return 3
    if consecutive_correct == 2:
        return 7
    if consecutive_correct == 3:
        return 14
    return 30


class TrainingStore:
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS puzzles (
                    puzzle_id TEXT PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    ply INTEGER NOT NULL,
                    fen_before TEXT NOT NULL,
                    color TEXT NOT NULL,
                    game_label TEXT NOT NULL,
                    move_label TEXT NOT NULL,
                    your_move_san TEXT NOT NULL,
                    your_move_uci TEXT NOT NULL,
                    best_move_san TEXT NOT NULL,
                    best_move_uci TEXT NOT NULL,
                    cpl INTEGER NOT NULL,
                    quality TEXT NOT NULL,
                    opening TEXT NOT NULL,
                    eco TEXT NOT NULL,
                    game_phase TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    motif TEXT NOT NULL,
                    difficulty INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    correct_attempts INTEGER NOT NULL DEFAULT 0,
                    consecutive_correct INTEGER NOT NULL DEFAULT 0,
                    last_reviewed_at TEXT,
                    next_review_at TEXT NOT NULL,
                    mastered INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    puzzle_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    previous_interval_days INTEGER NOT NULL,
                    next_interval_days INTEGER NOT NULL,
                    FOREIGN KEY (puzzle_id) REFERENCES puzzles(puzzle_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_puzzles_due
                    ON puzzles(next_review_at);
                CREATE INDEX IF NOT EXISTS idx_reviews_puzzle
                    ON reviews(puzzle_id, id);
                """
            )

    def upsert_puzzles(
        self,
        puzzles: list[PuzzleSeed],
        *,
        now: datetime | None = None,
    ) -> UpsertResult:
        resolved_now = _ensure_utc(now)
        timestamp = _iso(resolved_now)
        inserted = 0
        updated = 0
        with self._connect() as connection:
            existing_ids = {
                str(row["puzzle_id"])
                for row in connection.execute("SELECT puzzle_id FROM puzzles")
            }
            for puzzle in puzzles:
                if puzzle.puzzle_id in existing_ids:
                    updated += 1
                else:
                    inserted += 1
                    existing_ids.add(puzzle.puzzle_id)
                connection.execute(
                    """
                    INSERT INTO puzzles (
                        puzzle_id, game_id, ply, fen_before, color, game_label,
                        move_label, your_move_san, your_move_uci, best_move_san,
                        best_move_uci, cpl, quality, opening, eco, game_phase,
                        source_url, motif, difficulty, created_at, updated_at,
                        next_review_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(puzzle_id) DO UPDATE SET
                        game_id = excluded.game_id,
                        ply = excluded.ply,
                        fen_before = excluded.fen_before,
                        color = excluded.color,
                        game_label = excluded.game_label,
                        move_label = excluded.move_label,
                        your_move_san = excluded.your_move_san,
                        your_move_uci = excluded.your_move_uci,
                        best_move_san = excluded.best_move_san,
                        best_move_uci = excluded.best_move_uci,
                        cpl = excluded.cpl,
                        quality = excluded.quality,
                        opening = excluded.opening,
                        eco = excluded.eco,
                        game_phase = excluded.game_phase,
                        source_url = excluded.source_url,
                        motif = excluded.motif,
                        difficulty = excluded.difficulty,
                        updated_at = excluded.updated_at
                    """,
                    (
                        puzzle.puzzle_id,
                        puzzle.game_id,
                        puzzle.ply,
                        puzzle.fen_before,
                        puzzle.color,
                        puzzle.game_label,
                        puzzle.move_label,
                        puzzle.your_move_san,
                        puzzle.your_move_uci,
                        puzzle.best_move_san,
                        puzzle.best_move_uci,
                        puzzle.cpl,
                        puzzle.quality,
                        puzzle.opening,
                        puzzle.eco,
                        puzzle.game_phase,
                        puzzle.source_url,
                        puzzle.motif,
                        puzzle.difficulty,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            total = int(connection.execute("SELECT COUNT(*) FROM puzzles").fetchone()[0])
        return UpsertResult(inserted=inserted, updated=updated, total=total)

    @staticmethod
    def _row_to_puzzle(row: sqlite3.Row) -> StoredPuzzle:
        next_review = _parse_datetime(str(row["next_review_at"]))
        if next_review is None:
            raise ValueError("Puzzle has no next review time")
        return StoredPuzzle(
            puzzle_id=str(row["puzzle_id"]),
            game_id=str(row["game_id"]),
            ply=int(row["ply"]),
            fen_before=str(row["fen_before"]),
            color=str(row["color"]),
            game_label=str(row["game_label"]),
            move_label=str(row["move_label"]),
            your_move_san=str(row["your_move_san"]),
            your_move_uci=str(row["your_move_uci"]),
            best_move_san=str(row["best_move_san"]),
            best_move_uci=str(row["best_move_uci"]),
            cpl=int(row["cpl"]),
            quality=str(row["quality"]),
            opening=str(row["opening"]),
            eco=str(row["eco"]),
            game_phase=str(row["game_phase"]),
            source_url=str(row["source_url"]),
            motif=str(row["motif"]),
            difficulty=int(row["difficulty"]),
            attempts=int(row["attempts"]),
            correct_attempts=int(row["correct_attempts"]),
            consecutive_correct=int(row["consecutive_correct"]),
            last_reviewed_at=_parse_datetime(row["last_reviewed_at"]),
            next_review_at=next_review,
            mastered=bool(row["mastered"]),
        )

    def get_puzzle(self, puzzle_id: str) -> StoredPuzzle | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM puzzles WHERE puzzle_id = ?",
                (puzzle_id,),
            ).fetchone()
        return self._row_to_puzzle(row) if row is not None else None

    def review_count(self, puzzle_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE puzzle_id = ?",
                    (puzzle_id,),
                ).fetchone()[0]
            )

    def record_review(
        self,
        puzzle_id: str,
        *,
        answer: str,
        correct: bool,
        now: datetime | None = None,
    ) -> ReviewResult:
        resolved_now = _ensure_utc(now)
        reviewed_at = _iso(resolved_now)
        with self._connect() as connection:
            puzzle = connection.execute(
                "SELECT * FROM puzzles WHERE puzzle_id = ?",
                (puzzle_id,),
            ).fetchone()
            if puzzle is None:
                raise KeyError(f"Unknown puzzle: {puzzle_id}")

            old_streak = int(puzzle["consecutive_correct"])
            new_streak = old_streak + 1 if correct else 0
            next_interval = _interval_for(correct, new_streak)
            next_review = resolved_now + timedelta(days=next_interval)
            mastered = bool(correct and new_streak >= 4)
            previous_review = connection.execute(
                """
                SELECT next_interval_days
                FROM reviews
                WHERE puzzle_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (puzzle_id,),
            ).fetchone()
            previous_interval = int(previous_review[0]) if previous_review is not None else 0

            connection.execute(
                """
                UPDATE puzzles
                SET attempts = attempts + 1,
                    correct_attempts = correct_attempts + ?,
                    consecutive_correct = ?,
                    last_reviewed_at = ?,
                    next_review_at = ?,
                    mastered = ?
                WHERE puzzle_id = ?
                """,
                (
                    int(correct),
                    new_streak,
                    reviewed_at,
                    _iso(next_review),
                    int(mastered),
                    puzzle_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO reviews (
                    puzzle_id, reviewed_at, answer, correct,
                    previous_interval_days, next_interval_days
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    puzzle_id,
                    reviewed_at,
                    answer,
                    int(correct),
                    previous_interval,
                    next_interval,
                ),
            )

        return ReviewResult(
            puzzle_id=puzzle_id,
            correct=correct,
            next_interval_days=next_interval,
            consecutive_correct=new_streak,
            next_review_at=next_review,
            mastered=mastered,
        )

    def due_puzzles(
        self,
        *,
        limit: int = 10,
        now: datetime | None = None,
    ) -> list[StoredPuzzle]:
        if limit <= 0:
            return []
        timestamp = _iso(_ensure_utc(now))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM puzzles
                WHERE next_review_at <= ?
                ORDER BY
                    next_review_at ASC,
                    (attempts - correct_attempts) DESC,
                    difficulty DESC,
                    puzzle_id ASC
                LIMIT ?
                """,
                (timestamp, int(limit)),
            ).fetchall()
        return [self._row_to_puzzle(row) for row in rows]

    def _progress_rows(self, column: str) -> list[ProgressRow]:
        if column not in {"motif", "opening"}:
            raise ValueError(f"Unsupported progress dimension: {column}")
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {column} AS label,
                       COUNT(*) AS puzzles,
                       SUM(attempts) AS attempts,
                       SUM(correct_attempts) AS correct
                FROM puzzles
                GROUP BY {column}
                ORDER BY attempts DESC, puzzles DESC, label ASC
                """
            ).fetchall()
        result = []
        for row in rows:
            attempts = int(row["attempts"] or 0)
            correct = int(row["correct"] or 0)
            result.append(
                ProgressRow(
                    label=str(row["label"]),
                    puzzles=int(row["puzzles"]),
                    attempts=attempts,
                    correct=correct,
                    accuracy=(correct / attempts) if attempts else None,
                )
            )
        return result

    def progress(self, *, now: datetime | None = None) -> ProgressSummary:
        timestamp = _iso(_ensure_utc(now))
        with self._connect() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN next_review_at <= ? THEN 1 ELSE 0 END) AS due,
                       SUM(CASE WHEN attempts > 0 THEN 1 ELSE 0 END) AS reviewed,
                       SUM(CASE WHEN mastered = 1 THEN 1 ELSE 0 END) AS mastered
                FROM puzzles
                """,
                (timestamp,),
            ).fetchone()
            review_stats = connection.execute(
                """
                SELECT COUNT(*) AS total_reviews,
                       SUM(correct) AS correct_reviews
                FROM reviews
                """
            ).fetchone()

        total_reviews = int(review_stats["total_reviews"] or 0)
        correct_reviews = int(review_stats["correct_reviews"] or 0)
        return ProgressSummary(
            total_puzzles=int(counts["total"] or 0),
            due_puzzles=int(counts["due"] or 0),
            reviewed_puzzles=int(counts["reviewed"] or 0),
            mastered_puzzles=int(counts["mastered"] or 0),
            total_reviews=total_reviews,
            accuracy=(correct_reviews / total_reviews) if total_reviews else None,
            by_motif=self._progress_rows("motif"),
            by_opening=self._progress_rows("opening"),
        )
