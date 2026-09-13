import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MoveQualityThresholds:
    inaccuracy: int = 50
    mistake: int = 100
    blunder: int = 200

    def __post_init__(self) -> None:
        if not 0 <= self.inaccuracy < self.mistake < self.blunder:
            raise ValueError("CPL thresholds must satisfy 0 <= inaccuracy < mistake < blunder")


@dataclass(frozen=True)
class Settings:
    username: str = "srbmaury"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    stockfish_path: str | None = None
    stockfish_depth: int = 14
    min_group_size: int = 10
    thresholds: MoveQualityThresholds = field(default_factory=MoveQualityThresholds)


def get_settings(
    username: str | None = None,
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    stockfish_path: str | None = None,
    stockfish_depth: int | None = None,
    min_group_size: int | None = None,
    inaccuracy_cpl: int | None = None,
    mistake_cpl: int | None = None,
    blunder_cpl: int | None = None,
) -> Settings:
    thresholds = MoveQualityThresholds(
        inaccuracy=(
            inaccuracy_cpl
            if inaccuracy_cpl is not None
            else int(os.getenv("CHESS_COACH_INACCURACY_CPL", "50"))
        ),
        mistake=(
            mistake_cpl
            if mistake_cpl is not None
            else int(os.getenv("CHESS_COACH_MISTAKE_CPL", "100"))
        ),
        blunder=(
            blunder_cpl
            if blunder_cpl is not None
            else int(os.getenv("CHESS_COACH_BLUNDER_CPL", "200"))
        ),
    )
    return Settings(
        username=username or os.getenv("CHESS_COACH_USERNAME", "srbmaury"),
        data_dir=data_dir or Path(os.getenv("CHESS_COACH_DATA_DIR", "data")),
        model_dir=model_dir or Path(os.getenv("CHESS_COACH_MODEL_DIR", "models")),
        stockfish_path=(
            stockfish_path if stockfish_path is not None else os.getenv("STOCKFISH_PATH")
        ),
        stockfish_depth=(
            stockfish_depth
            if stockfish_depth is not None
            else int(os.getenv("CHESS_COACH_STOCKFISH_DEPTH", "14"))
        ),
        min_group_size=(
            min_group_size
            if min_group_size is not None
            else int(os.getenv("CHESS_COACH_MIN_GROUP_SIZE", "10"))
        ),
        thresholds=thresholds,
    )
