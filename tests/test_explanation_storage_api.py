from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.app import create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="g1",
        ply=1,
        fen_before=START_FEN,
        color="white",
        game_label="srbmaury vs opponent",
        move_label="1.e4",
        your_move_san="e4",
        your_move_uci="e2e4",
        best_move_san="d4",
        best_move_uci="d2d4",
        cpl=250,
        eval_loss_pawns=2.5,
        quality="blunder",
        opening="Queen's Pawn Opening",
        eco="D00",
        game_phase="opening",
        source_url="https://www.chess.com/game/live/1",
        motif="positional / calculation",
        difficulty=3,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")


def _store(tmp_path: Path) -> TrainingStore:
    settings = _settings(tmp_path)
    store = TrainingStore(settings.data_dir / "training" / "training.db")
    store.upsert_puzzles([_seed()], now=datetime(2026, 1, 1, tzinfo=UTC))
    return store


def test_explanation_cache_round_trips_by_fingerprint_and_depth(tmp_path: Path):
    store = _store(tmp_path)

    assert store.get_explanation("p1", fingerprint="abc", depth=14) is None

    store.save_explanation(
        "p1",
        fingerprint="abc",
        depth=14,
        payload={
            "idea": "Forcing line",
            "why": "The move creates a concrete threat.",
            "best_line": ["d4", "d5"],
            "why_your_move_was_worse": "e4 missed the stronger continuation.",
            "engine_grounded": True,
        },
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )

    cached = store.get_explanation("p1", fingerprint="abc", depth=14)
    assert cached is not None
    assert cached["idea"] == "Forcing line"
    assert cached["best_line"] == ["d4", "d5"]
    assert store.get_explanation("p1", fingerprint="changed", depth=14) is None
    assert store.get_explanation("p1", fingerprint="abc", depth=16) is None


def test_explanation_is_hidden_until_puzzle_has_been_attempted(tmp_path: Path):
    _store(tmp_path)
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.get("/api/practice/p1/explanation")

    assert response.status_code == 409
    assert "attempt" in response.json()["detail"].lower()
