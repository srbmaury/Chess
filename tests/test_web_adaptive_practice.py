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


def _seed(puzzle_id: str = "p1") -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
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


def _settings(tmp_path: Path, username: str = "srbmaury") -> Settings:
    return Settings(
        username=username,
        data_dir=tmp_path / username / "data",
        model_dir=tmp_path / username / "models",
        stockfish_path="unused-in-tests",
        stockfish_depth=14,
    )


def _info(cp: int, *pv: str):
    return {
        "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
        "pv": [chess.Move.from_uci(move) for move in pv],
    }


class FakeEngine:
    def __init__(self, script: dict[str, dict | Exception]):
        self.script = script
        self.closed = False

    def analyse(self, board: chess.Board, depth: int):
        result = self.script.get(board.fen())
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AssertionError(f"No scripted analysis for {board.fen()}")
        return result

    def close(self):
        self.closed = True


def _board(*moves: str) -> chess.Board:
    board = chess.Board(START_FEN)
    for move in moves:
        board.push_uci(move)
    return board


def _client(tmp_path: Path, engine: FakeEngine):
    settings = _settings(tmp_path)
    store = TrainingStore(training_db_path(settings))
    store.upsert_puzzles([_seed()])
    services = AdaptiveServiceRegistry(engine_factory=lambda: engine)
    app = create_app(settings, adaptive_services=services)
    return TestClient(app), store, app


def test_adaptive_start_is_default_safe_state_and_resumes_same_session(tmp_path: Path):
    client, _, _ = _client(tmp_path, FakeEngine({}))

    first = client.post("/api/practice/p1/adaptive/start")
    second = client.post("/api/practice/p1/adaptive/start")

    assert first.status_code == 200
    assert second.status_code == 200
    payload = first.json()
    assert payload["session_id"] == second.json()["session_id"]
    assert payload["status"] == "active"
    assert payload["current_fen"] == START_FEN
    assert payload["orientation"] == "white"
    assert payload["max_user_decisions"] == 4
    encoded = str(payload).lower()
    assert "best_move" not in encoded
    assert "expected_user" not in encoded
    assert "principal_variation" not in encoded


def test_adaptive_move_returns_played_engine_reply_but_not_future_solution(tmp_path: Path):
    start = _board()
    after_d4 = _board("d2d4")
    after_reply = _board("d2d4", "d7d5")
    engine = FakeEngine(
        {
            start.fen(): _info(100, "d2d4", "d7d5", "g1f3"),
            after_d4.fen(): _info(100, "d7d5", "g1f3"),
        }
    )
    client, _, _ = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    response = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "d2d4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["engine_reply_uci"] == "d7d5"
    assert payload["current_fen"] == after_reply.fen()
    encoded = str(payload).lower()
    assert "g1f3" not in encoded
    assert "expected" not in encoded


def test_rejected_move_retry_stays_active_and_does_not_reveal_solution(tmp_path: Path):
    start = _board()
    after_c4 = _board("c2c4")
    engine = FakeEngine(
        {
            start.fen(): _info(100, "d2d4"),
            after_c4.fen(): _info(69, "e7e5"),
        }
    )
    client, store, _ = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    first = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "c2c4"},
    )
    second = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "c2c4"},
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["accepted"] is False
    assert first_payload["status"] == "active"
    assert first_payload["current_fen"] == START_FEN
    assert first_payload["current_ply"] == 0
    assert first_payload["review"] is None
    assert "d2d4" not in str(first_payload)
    assert "expected" not in str(first_payload).lower()

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "active"
    assert second_payload["current_fen"] == START_FEN
    assert second_payload["user_moves_attempted"] == 2
    assert store.review_count("p1") == 0


def test_illegal_move_is_422_without_session_progress(tmp_path: Path):
    client, _, _ = _client(tmp_path, FakeEngine({}))
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    response = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "e2e5"},
    )

    assert response.status_code == 422
    state = client.post("/api/practice/p1/adaptive/start").json()
    assert state["user_moves_attempted"] == 0
    assert state["current_fen"] == START_FEN


def test_inactive_puzzle_cannot_start_or_continue(tmp_path: Path):
    start = _board()
    engine = FakeEngine({start.fen(): _info(100, "d2d4")})
    client, store, _ = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]
    store.upsert_puzzles([])

    fresh = client.post("/api/practice/p1/adaptive/start")
    continued = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "d2d4"},
    )

    assert fresh.status_code == 409
    assert continued.status_code == 409


def test_engine_failure_is_503_and_does_not_penalize_review(tmp_path: Path):
    start = _board()
    engine = FakeEngine({start.fen(): RuntimeError("engine unavailable")})
    client, store, _ = _client(tmp_path, engine)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    response = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "d2d4"},
    )

    assert response.status_code == 503
    assert store.review_count("p1") == 0


def test_session_id_is_scoped_to_current_player_database(tmp_path: Path):
    engine = FakeEngine({})
    settings_a = _settings(tmp_path, "alice")
    store_a = TrainingStore(training_db_path(settings_a))
    store_a.upsert_puzzles([_seed()])
    settings_b = _settings(tmp_path, "bob")
    store_b = TrainingStore(training_db_path(settings_b))
    store_b.upsert_puzzles([_seed()])
    services = AdaptiveServiceRegistry(engine_factory=lambda: engine)
    app = create_app(settings_a, adaptive_services=services)
    client = TestClient(app)
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    app.state.settings = settings_b
    response = client.post(
        f"/api/practice/adaptive/{session_id}/move",
        json={"move_uci": "d2d4"},
    )

    assert response.status_code == 404


def test_abandon_is_idempotent_and_records_no_review(tmp_path: Path):
    client, store, _ = _client(tmp_path, FakeEngine({}))
    session_id = client.post("/api/practice/p1/adaptive/start").json()["session_id"]

    first = client.post(f"/api/practice/adaptive/{session_id}/abandon")
    second = client.post(f"/api/practice/adaptive/{session_id}/abandon")

    assert first.status_code == 200
    assert first.json()["status"] == "abandoned"
    assert second.status_code == 200
    assert second.json()["status"] == "abandoned"
    assert store.review_count("p1") == 0
