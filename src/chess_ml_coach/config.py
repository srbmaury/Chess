from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class MoveQualityThresholds:
    inaccuracy: int = 50
    mistake: int = 100
    blunder: int = 200


@dataclass(frozen=True)
class Settings:
    username: str = "srbmaury"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    stockfish_path: str | None = None
    stockfish_depth: int = 14
    min_group_size: int = 10
    thresholds: MoveQualityThresholds = field(default_factory=MoveQualityThresholds)


def get_settings(username: str | None = None) -> Settings:
    return Settings(
        username=username or os.getenv("CHESS_COACH_USERNAME", "srbmaury"),
        data_dir=Path(os.getenv("CHESS_COACH_DATA_DIR", "data")),
        model_dir=Path(os.getenv("CHESS_COACH_MODEL_DIR", "models")),
        stockfish_path=os.getenv("STOCKFISH_PATH"),
        stockfish_depth=int(os.getenv("CHESS_COACH_STOCKFISH_DEPTH", "14")),
    )
