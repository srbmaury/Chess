from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.app import create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed(
    puzzle_id: str = "p1",
    *,
    best_move_san: str = "d4",
    best_move_uci: str = "d2d4",
    motif: str = "positional / calculation",
    opening: str = "Queen's Pawn Opening",
    quality: str = "blunder",
) -> PuzzleSeed:
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
        best_move_san=best_move_san,
        best_move_uci=best_move_uci,
        cpl=250,
        eval_loss_pawns=2.5,
        quality=quality,
        opening=opening,
        eco="D00",
        game_phase="opening",
        source_url="https://www.chess.com/game/live/1",
        motif=motif,
        difficulty=3,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")


def _seed_store(tmp_path: Path, *puzzles: PuzzleSeed) -> TrainingStore:
    settings = _settings(tmp_path)
    store = TrainingStore(settings.data_dir / "training" / "training.db")
    store.upsert_puzzles(list(puzzles), now=datetime(2026, 1, 1, tzinfo=UTC))
    return store


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_settings(tmp_path)))


def test_health_is_local_and_does_not_expose_arbitrary_files(tmp_path: Path):
    response = _client(tmp_path).get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["username"] == "srbmaury"
    assert payload["data_dir"] == str(tmp_path / "data")
    assert "raw_pgn" not in payload


def test_practice_next_hides_answer_until_attempt(tmp_path: Path):
    _seed_store(tmp_path, _seed())

    response = _client(tmp_path).get("/api/practice/next")

    assert response.status_code == 200
    puzzle = response.json()["puzzle"]
    assert puzzle["puzzle_id"] == "p1"
    assert puzzle["fen"] == START_FEN
    assert puzzle["orientation"] == "white"
    assert puzzle["game"] == "srbmaury vs opponent"
    assert puzzle["opening"] == "Queen's Pawn Opening"
    assert puzzle["motif"] == "positional / calculation"
    assert puzzle["difficulty"] == 3
    assert "best_move_uci" not in puzzle
    assert "best_move_san" not in puzzle
    assert "your_move_san" not in puzzle
    assert "cpl" not in puzzle
    assert "game_id" not in puzzle


def test_correct_practice_attempt_records_review_and_reveals_feedback(tmp_path: Path):
    store = _seed_store(tmp_path, _seed())

    response = _client(tmp_path).post(
        "/api/practice/p1/attempt",
        json={"move_uci": "d2d4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["correct"] is True
    assert payload["best_move_san"] == "d4"
    assert payload["best_move_uci"] == "d2d4"
    assert payload["your_game_move"] == "e4"
    assert payload["evaluation_loss_pawns"] == 2.5
    assert payload["next_interval_days"] == 3
    assert store.review_count("p1") == 1


def test_illegal_practice_attempt_does_not_record_review(tmp_path: Path):
    store = _seed_store(tmp_path, _seed())

    response = _client(tmp_path).post(
        "/api/practice/p1/attempt",
        json={"move_uci": "e2e5"},
    )

    assert response.status_code == 422
    assert "legal" in response.json()["detail"].lower()
    assert store.review_count("p1") == 0


def test_skip_does_not_record_review(tmp_path: Path):
    store = _seed_store(tmp_path, _seed())

    response = _client(tmp_path).post("/api/practice/p1/skip")

    assert response.status_code == 200
    assert response.json() == {"skipped": True}
    assert store.review_count("p1") == 0


def test_puzzle_explorer_filters_and_paginates_without_game_id(tmp_path: Path):
    store = _seed_store(
        tmp_path,
        _seed("p1", motif="fork", opening="Opening A", quality="blunder"),
        _seed("p2", motif="pin", opening="Opening B", quality="mistake"),
        _seed("p3", motif="fork", opening="Opening A", quality="mistake"),
    )
    store.record_review(
        "p1",
        answer="d4",
        correct=True,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    response = _client(tmp_path).get(
        "/api/puzzles",
        params={
            "motif": "fork",
            "opening": "Opening A",
            "reviewed": "true",
            "limit": 1,
            "offset": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["puzzle_id"] == "p1"
    assert "game_id" not in payload["items"][0]


def test_puzzle_detail_intentionally_includes_solution(tmp_path: Path):
    _seed_store(tmp_path, _seed())

    response = _client(tmp_path).get("/api/puzzles/p1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_move_san"] == "d4"
    assert payload["best_move_uci"] == "d2d4"
    assert payload["your_move_san"] == "e4"
    assert "game_id" not in payload


def test_progress_includes_daily_review_history(tmp_path: Path):
    store = _seed_store(tmp_path, _seed())
    reviewed_at = datetime.now(UTC)
    store.record_review("p1", answer="d4", correct=True, now=reviewed_at)

    response = _client(tmp_path).get("/api/progress")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_puzzles"] == 1
    assert payload["total_reviews"] == 1
    assert payload["accuracy"] == 1.0
    assert payload["daily_reviews"][-1]["reviews"] == 1
    assert payload["daily_reviews"][-1]["correct"] == 1


def test_missing_training_database_returns_actionable_conflict(tmp_path: Path):
    client = _client(tmp_path)

    response = client.get("/api/practice/next")

    assert response.status_code == 409
    assert "chess-coach puzzles" in response.json()["detail"]


def test_dashboard_works_before_puzzle_bank_exists(tmp_path: Path):
    response = _client(tmp_path).get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["training"]["total_puzzles"] == 0
    assert payload["training"]["due_puzzles"] == 0
    assert payload["artifacts"]["analysis"]["exists"] is False
