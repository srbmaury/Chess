from __future__ import annotations

from pathlib import Path

import chess
import chess.engine
from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.services import training_db_path
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.adaptive_routes import AdaptiveServiceRegistry
from chess_ml_coach.web.app import create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="game-p1",
        ply=1,
        fen_before=START_FEN,
        color="white",
        game_label="player vs opponent",
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
        source_url="",
        motif="positional / calculation",
        difficulty=3,
    )


def _info(cp: int, *pv: str) -> dict:
    return {
        "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
        "pv": [chess.Move.from_uci(move) for move in pv],
    }


class FakeEngine:
    def __init__(self, script: dict[str, dict]):
        self.script = script

    def analyse(self, board: chess.Board, depth: int) -> dict:
        assert depth == 14
        result = self.script.get(board.fen())
        if result is None:
            raise AssertionError(f"No scripted analysis for {board.fen()}")
        return result

    def close(self) -> None:
        pass


def _client(tmp_path: Path, engine: FakeEngine):
    settings = Settings(
        username="player",
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        stockfish_path="unused-in-tests",
        stockfish_depth=14,
    )
    store = TrainingStore(training_db_path(settings))
    store.upsert_puzzles([_seed()])
    app = create_app(settings, adaptive_services=AdaptiveServiceRegistry(engine_factory=lambda: engine))
    return TestClient(app), store


def test_hint_reveals_only_current_best_move_and_keeps_session_active(tmp_path: Path):
    start = chess.Board(START_FEN)
    engine = FakeEngine({start.fen(): _info(100, "d2d4", "d7d5", "g1f3")})
    client, store = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    response = client.post(f"/api/practice/adaptive/{session_id}/hint")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "active"
    assert payload["move_uci"] == "d2d4"
    assert payload["move_san"] == "d4"
    assert payload["current_fen"] == START_FEN
    assert payload["current_ply"] == 0
    assert payload["user_moves_attempted"] == 1
    encoded = str(payload).lower()
    assert "d7d5" not in encoded
    assert "g1f3" not in encoded
    assert "expected" not in encoded
    assert "principal_variation" not in encoded
    assert store.review_count("p1") == 0


def test_resumed_session_hides_rejected_guesses_after_hint_from_public_line(tmp_path: Path):
    start = chess.Board(START_FEN)
    after_c4 = start.copy(stack=False)
    after_c4.push_uci("c2c4")
    engine = FakeEngine(
        {
            start.fen(): _info(100, "d2d4", "d7d5"),
            after_c4.fen(): _info(50, "e7e5"),
        }
    )
    client, _ = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    rejected = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "c2c4"},
    )
    hint = client.post(f"/api/practice/adaptive/{session_id}/hint")
    resumed = client.post("/api/practice/p1/adaptive/start")

    assert rejected.status_code == 200
    assert hint.status_code == 200
    assert resumed.status_code == 200
    payload = resumed.json()
    assert payload["steps"] == []
    assert payload["user_moves_attempted"] == 1
    assert payload["user_moves_accepted"] == 0
    assert payload["current_fen"] == START_FEN
