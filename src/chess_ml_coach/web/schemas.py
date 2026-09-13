from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    username: str
    data_dir: str
    model_dir: str


class PracticePuzzle(BaseModel):
    puzzle_id: str
    fen: str
    orientation: str
    game: str
    move: str
    opening: str
    eco: str
    phase: str
    motif: str
    difficulty: int
    source_url: str | None = None


class PracticeNextResponse(BaseModel):
    puzzle: PracticePuzzle | None


class AttemptRequest(BaseModel):
    move_uci: str = Field(min_length=4, max_length=5)


class AttemptResponse(BaseModel):
    correct: bool
    best_move_san: str
    best_move_uci: str
    your_game_move: str
    evaluation_loss_pawns: float
    next_interval_days: int
    next_review_at: datetime
    source_url: str | None = None


class PuzzleItem(BaseModel):
    puzzle_id: str
    fen: str
    orientation: str
    game: str
    move: str
    your_move_san: str
    your_move_uci: str
    best_move_san: str
    best_move_uci: str
    evaluation_loss_pawns: float
    quality: str
    opening: str
    eco: str
    phase: str
    source_url: str | None = None
    motif: str
    difficulty: int
    attempts: int
    correct_attempts: int
    accuracy: float | None
    consecutive_correct: int
    next_review_at: datetime
    mastered: bool


class PuzzleListResponse(BaseModel):
    items: list[PuzzleItem]
    total: int
    limit: int
    offset: int


class DailyReviewRow(BaseModel):
    date: str
    reviews: int
    correct: int
    accuracy: float


class ProgressGroupRow(BaseModel):
    label: str
    puzzles: int
    attempts: int
    correct: int
    accuracy: float | None


class ProgressResponse(BaseModel):
    total_puzzles: int
    due_puzzles: int
    reviewed_puzzles: int
    mastered_puzzles: int
    total_reviews: int
    accuracy: float | None
    by_motif: list[ProgressGroupRow]
    by_opening: list[ProgressGroupRow]
    daily_reviews: list[DailyReviewRow]


class TrainingSummary(BaseModel):
    total_puzzles: int
    due_puzzles: int
    reviewed_puzzles: int
    mastered_puzzles: int
    total_reviews: int
    accuracy: float | None


class ArtifactState(BaseModel):
    exists: bool
    updated_at: datetime | None = None
    rows: int | None = None


class DashboardResponse(BaseModel):
    analyzed_moves: int
    training: TrainingSummary
    artifacts: dict[str, ArtifactState]
