from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .training import ReviewResult, StoredPuzzle, _ensure_utc, _interval_for, _iso


@dataclass(frozen=True)
class NewAdaptiveStep:
    side: str
    fen_before: str
    move_uci: str
    move_san: str
    accepted: bool
    source: str
    eval_cp: int | None = None
    best_eval_cp: int | None = None
    eval_loss_cp: int | None = None


@dataclass(frozen=True)
class AdaptiveStep:
    id: int
    session_id: str
    step_index: int
    side: str
    fen_before: str
    move_uci: str
    move_san: str
    accepted: bool
    source: str
    eval_cp: int | None
    best_eval_cp: int | None
    eval_loss_cp: int | None
    created_at: datetime


@dataclass(frozen=True)
class AdaptiveSession:
    session_id: str
    puzzle_id: str
    mode: str
    starting_fen: str
    current_fen: str
    user_color: str
    status: str
    user_moves_attempted: int
    user_moves_accepted: int
    current_ply: int
    engine_depth: int
    max_eval_loss_cp: int
    previous_best_eval_cp: int | None
    review_recorded: bool
    review_id: int | None
    review_next_interval_days: int | None
    review_next_review_at: datetime | None
    review_consecutive_correct: int | None
    review_mastered: bool | None
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class AdaptiveMetrics:
    sessions_completed: int
    success_rate: float | None
    continuation_accuracy: float | None
    average_accepted_decisions: float | None
    average_calculation_depth_plies: float | None


def _parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


class AdaptiveSessionStore:
    """Persistence for resumable adaptive puzzle sessions in training.db."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS practice_sessions (
                    session_id TEXT PRIMARY KEY,
                    puzzle_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    starting_fen TEXT NOT NULL,
                    current_fen TEXT NOT NULL,
                    user_color TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_moves_attempted INTEGER NOT NULL DEFAULT 0,
                    user_moves_accepted INTEGER NOT NULL DEFAULT 0,
                    current_ply INTEGER NOT NULL DEFAULT 0,
                    engine_depth INTEGER NOT NULL,
                    max_eval_loss_cp INTEGER NOT NULL DEFAULT 0,
                    previous_best_eval_cp INTEGER,
                    review_recorded INTEGER NOT NULL DEFAULT 0,
                    review_id INTEGER,
                    review_next_interval_days INTEGER,
                    review_next_review_at TEXT,
                    review_consecutive_correct INTEGER,
                    review_mastered INTEGER,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY (puzzle_id) REFERENCES puzzles(puzzle_id) ON DELETE CASCADE,
                    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS practice_session_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    fen_before TEXT NOT NULL,
                    move_uci TEXT NOT NULL,
                    move_san TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    eval_cp INTEGER,
                    best_eval_cp INTEGER,
                    eval_loss_cp INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES practice_sessions(session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, step_index)
                );

                CREATE INDEX IF NOT EXISTS idx_practice_sessions_puzzle
                    ON practice_sessions(puzzle_id, status);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_practice_sessions_active_puzzle
                    ON practice_sessions(puzzle_id)
                    WHERE status = 'active';
                CREATE INDEX IF NOT EXISTS idx_practice_steps_session
                    ON practice_session_steps(session_id, step_index);
                """
            )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> AdaptiveSession:
        review_mastered = row["review_mastered"]
        return AdaptiveSession(
            session_id=str(row["session_id"]),
            puzzle_id=str(row["puzzle_id"]),
            mode=str(row["mode"]),
            starting_fen=str(row["starting_fen"]),
            current_fen=str(row["current_fen"]),
            user_color=str(row["user_color"]),
            status=str(row["status"]),
            user_moves_attempted=int(row["user_moves_attempted"]),
            user_moves_accepted=int(row["user_moves_accepted"]),
            current_ply=int(row["current_ply"]),
            engine_depth=int(row["engine_depth"]),
            max_eval_loss_cp=int(row["max_eval_loss_cp"]),
            previous_best_eval_cp=_optional_int(row["previous_best_eval_cp"]),
            review_recorded=bool(row["review_recorded"]),
            review_id=_optional_int(row["review_id"]),
            review_next_interval_days=_optional_int(row["review_next_interval_days"]),
            review_next_review_at=_parse_datetime(row["review_next_review_at"]),
            review_consecutive_correct=_optional_int(row["review_consecutive_correct"]),
            review_mastered=(None if review_mastered is None else bool(review_mastered)),
            started_at=_parse_datetime(row["started_at"]) or datetime.now(UTC),
            updated_at=_parse_datetime(row["updated_at"]) or datetime.now(UTC),
            finished_at=_parse_datetime(row["finished_at"]),
        )

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> AdaptiveStep:
        created_at = _parse_datetime(row["created_at"])
        if created_at is None:
            raise ValueError("Adaptive step has no creation time")
        return AdaptiveStep(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            step_index=int(row["step_index"]),
            side=str(row["side"]),
            fen_before=str(row["fen_before"]),
            move_uci=str(row["move_uci"]),
            move_san=str(row["move_san"]),
            accepted=bool(row["accepted"]),
            source=str(row["source"]),
            eval_cp=_optional_int(row["eval_cp"]),
            best_eval_cp=_optional_int(row["best_eval_cp"]),
            eval_loss_cp=_optional_int(row["eval_loss_cp"]),
            created_at=created_at,
        )

    def get(self, session_id: str) -> AdaptiveSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def steps(self, session_id: str) -> list[AdaptiveStep]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM practice_session_steps
                WHERE session_id = ?
                ORDER BY step_index ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_step(row) for row in rows]

    def start_or_resume(
        self,
        puzzle: StoredPuzzle,
        *,
        depth: int,
        now: datetime | None = None,
    ) -> AdaptiveSession:
        if not puzzle.active:
            raise ValueError(f"Puzzle is inactive: {puzzle.puzzle_id}")
        resolved_now = _ensure_utc(now)
        timestamp = _iso(resolved_now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM practice_sessions
                WHERE puzzle_id = ? AND status = 'active'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (puzzle.puzzle_id,),
            ).fetchone()
            if existing is not None:
                return self._row_to_session(existing)
            session_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO practice_sessions (
                    session_id, puzzle_id, mode, starting_fen, current_fen,
                    user_color, status, engine_depth, started_at, updated_at
                ) VALUES (?, ?, 'adaptive', ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    session_id,
                    puzzle.puzzle_id,
                    puzzle.fen_before,
                    puzzle.fen_before,
                    puzzle.color,
                    int(depth),
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Failed to create adaptive session")
        return self._row_to_session(row)

    def advance(
        self,
        session_id: str,
        *,
        expected_current_fen: str,
        new_current_fen: str,
        steps: list[NewAdaptiveStep],
        user_attempted_delta: int,
        user_accepted_delta: int,
        current_ply: int,
        max_eval_loss_cp: int,
        previous_best_eval_cp: int | None,
        now: datetime | None = None,
    ) -> AdaptiveSession:
        resolved_now = _ensure_utc(now)
        timestamp = _iso(resolved_now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Unknown adaptive session: {session_id}")
            if str(session["status"]) != "active":
                raise ValueError(f"Adaptive session is not active: {session_id}")
            if str(session["current_fen"]) != expected_current_fen:
                raise ValueError("Adaptive session position changed; reload the current session")

            last_step = connection.execute(
                """
                SELECT COALESCE(MAX(step_index), -1)
                FROM practice_session_steps
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            next_index = int(last_step[0]) + 1
            for offset, step in enumerate(steps):
                if step.side not in {"user", "engine"}:
                    raise ValueError(f"Unsupported adaptive step side: {step.side}")
                if step.source not in {"pv", "engine", "user"}:
                    raise ValueError(f"Unsupported adaptive step source: {step.source}")
                connection.execute(
                    """
                    INSERT INTO practice_session_steps (
                        session_id, step_index, side, fen_before, move_uci, move_san,
                        accepted, source, eval_cp, best_eval_cp, eval_loss_cp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        next_index + offset,
                        step.side,
                        step.fen_before,
                        step.move_uci,
                        step.move_san,
                        int(step.accepted),
                        step.source,
                        step.eval_cp,
                        step.best_eval_cp,
                        step.eval_loss_cp,
                        timestamp,
                    ),
                )

            connection.execute(
                """
                UPDATE practice_sessions
                SET current_fen = ?,
                    user_moves_attempted = user_moves_attempted + ?,
                    user_moves_accepted = user_moves_accepted + ?,
                    current_ply = ?,
                    max_eval_loss_cp = MAX(max_eval_loss_cp, ?),
                    previous_best_eval_cp = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    new_current_fen,
                    int(user_attempted_delta),
                    int(user_accepted_delta),
                    int(current_ply),
                    int(max_eval_loss_cp),
                    previous_best_eval_cp,
                    timestamp,
                    session_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Adaptive session disappeared during update")
        return self._row_to_session(row)

    @staticmethod
    def _record_review_on_connection(
        connection: sqlite3.Connection,
        puzzle_id: str,
        *,
        answer: str,
        correct: bool,
        now: datetime,
    ) -> tuple[int, ReviewResult]:
        reviewed_at = _iso(now)
        puzzle = connection.execute(
            "SELECT * FROM puzzles WHERE puzzle_id = ?",
            (puzzle_id,),
        ).fetchone()
        if puzzle is None:
            raise KeyError(f"Unknown puzzle: {puzzle_id}")
        if not bool(puzzle["active"]):
            raise ValueError(f"Puzzle is inactive: {puzzle_id}")

        old_streak = int(puzzle["consecutive_correct"])
        new_streak = old_streak + 1 if correct else 0
        next_interval = _interval_for(correct, new_streak)
        next_review = now + __import__("datetime").timedelta(days=next_interval)
        mastered = bool(correct and new_streak >= 4)
        previous_review = connection.execute(
            """
            SELECT next_interval_days FROM reviews
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
        cursor = connection.execute(
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
        review_id = int(cursor.lastrowid)
        return review_id, ReviewResult(
            puzzle_id=puzzle_id,
            correct=correct,
            next_interval_days=next_interval,
            consecutive_correct=new_streak,
            next_review_at=next_review,
            mastered=mastered,
        )

    @staticmethod
    def _review_from_session(session: AdaptiveSession) -> ReviewResult:
        if (
            session.review_next_interval_days is None
            or session.review_next_review_at is None
            or session.review_consecutive_correct is None
            or session.review_mastered is None
        ):
            raise RuntimeError("Adaptive terminal session is missing its review snapshot")
        return ReviewResult(
            puzzle_id=session.puzzle_id,
            correct=session.status == "succeeded",
            next_interval_days=session.review_next_interval_days,
            consecutive_correct=session.review_consecutive_correct,
            next_review_at=session.review_next_review_at,
            mastered=session.review_mastered,
        )

    def finalize(
        self,
        session_id: str,
        *,
        succeeded: bool,
        answer: str,
        now: datetime | None = None,
    ) -> tuple[AdaptiveSession, ReviewResult]:
        resolved_now = _ensure_utc(now)
        timestamp = _iso(resolved_now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown adaptive session: {session_id}")
            session = self._row_to_session(row)
            if session.review_recorded:
                return session, self._review_from_session(session)
            if session.status != "active":
                raise ValueError(f"Adaptive session cannot be finalized from {session.status}")

            review_id, review = self._record_review_on_connection(
                connection,
                session.puzzle_id,
                answer=answer,
                correct=succeeded,
                now=resolved_now,
            )
            status = "succeeded" if succeeded else "failed"
            connection.execute(
                """
                UPDATE practice_sessions
                SET status = ?, review_recorded = 1, review_id = ?,
                    review_next_interval_days = ?, review_next_review_at = ?,
                    review_consecutive_correct = ?, review_mastered = ?,
                    updated_at = ?, finished_at = ?
                WHERE session_id = ?
                """,
                (
                    status,
                    review_id,
                    review.next_interval_days,
                    _iso(review.next_review_at),
                    review.consecutive_correct,
                    int(review.mastered),
                    timestamp,
                    timestamp,
                    session_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Adaptive session disappeared during finalization")
        return self._row_to_session(updated), review

    def abandon(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> AdaptiveSession:
        timestamp = _iso(_ensure_utc(now))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown adaptive session: {session_id}")
            session = self._row_to_session(row)
            if session.status != "active":
                return session
            connection.execute(
                """
                UPDATE practice_sessions
                SET status = 'abandoned', updated_at = ?, finished_at = ?
                WHERE session_id = ?
                """,
                (timestamp, timestamp, session_id),
            )
            updated = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Adaptive session disappeared during abandon")
        return self._row_to_session(updated)

    def metrics(self) -> AdaptiveMetrics:
        with self._connect() as connection:
            completed = connection.execute(
                """
                SELECT session_id, status, user_moves_accepted, current_ply
                FROM practice_sessions
                WHERE status IN ('succeeded', 'failed') AND review_recorded = 1
                """
            ).fetchall()
            user_steps = connection.execute(
                """
                SELECT steps.session_id, steps.step_index, steps.accepted,
                       steps.fen_before, sessions.starting_fen
                FROM practice_session_steps AS steps
                JOIN practice_sessions AS sessions
                  ON sessions.session_id = steps.session_id
                WHERE steps.side = 'user'
                ORDER BY steps.session_id, steps.step_index
                """
            ).fetchall()

        count = len(completed)
        if count == 0:
            return AdaptiveMetrics(0, None, None, None, None)

        success_count = sum(1 for row in completed if str(row["status"]) == "succeeded")
        continuation_attempts = 0
        continuation_correct = 0
        completed_ids = {str(row["session_id"]) for row in completed}
        for row in user_steps:
            session_id = str(row["session_id"])
            if session_id not in completed_ids:
                continue
            if str(row["fen_before"]) == str(row["starting_fen"]):
                continue
            continuation_attempts += 1
            continuation_correct += int(bool(row["accepted"]))

        return AdaptiveMetrics(
            sessions_completed=count,
            success_rate=success_count / count,
            continuation_accuracy=(
                continuation_correct / continuation_attempts
                if continuation_attempts
                else None
            ),
            average_accepted_decisions=(
                sum(int(row["user_moves_accepted"]) for row in completed) / count
            ),
            average_calculation_depth_plies=(
                sum(int(row["current_ply"]) for row in completed) / count
            ),
        )
