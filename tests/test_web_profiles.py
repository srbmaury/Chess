import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.profiles import ProfileManager
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.app import create_app
from chess_ml_coach.web.pipeline import PipelineManager
from chess_ml_coach.web.profile_routes import enable_profiles


def _root_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")


def _seed(puzzle_id: str) -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
        ply=1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
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


def test_profile_api_creates_and_switches_active_player(tmp_path: Path):
    root = _root_settings(tmp_path)
    profiles = ProfileManager(root)
    profiles.create_or_activate("srbmaury")
    app = create_app(root)
    enable_profiles(app, root, manager=profiles)
    client = TestClient(app)

    listed = client.get("/api/profiles")
    assert listed.status_code == 200
    assert listed.json()["active_username"] == "srbmaury"

    created = client.post("/api/profiles", json={"username": "Other_Player", "activate": False})
    assert created.status_code == 201
    assert created.json()["username"] == "other_player"

    activated = client.post("/api/profiles/other_player/activate")
    assert activated.status_code == 200
    assert activated.json()["active_username"] == "other_player"

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["username"] == "other_player"
    assert health.json()["data_dir"].endswith("data/users/other_player")


def test_profile_switch_cannot_read_previous_players_puzzles(tmp_path: Path):
    root = _root_settings(tmp_path)
    profiles = ProfileManager(root)
    profiles.create_or_activate("srbmaury")
    profiles.create_or_activate("other_player", activate=False)
    first_db = profiles.settings_for("srbmaury").data_dir / "training" / "training.db"
    second_db = profiles.settings_for("other_player").data_dir / "training" / "training.db"
    TrainingStore(first_db).upsert_puzzles([_seed("only-first")], now=datetime(2026, 1, 1, tzinfo=UTC))
    TrainingStore(second_db).upsert_puzzles([_seed("only-second")], now=datetime(2026, 1, 1, tzinfo=UTC))

    app = create_app(root)
    enable_profiles(app, root, manager=profiles)
    client = TestClient(app)

    assert client.get("/api/puzzles/only-first").status_code == 200
    assert client.post("/api/profiles/other_player/activate").status_code == 200
    assert client.get("/api/puzzles/only-first").status_code == 404
    assert client.get("/api/puzzles/only-second").status_code == 200


def test_profile_activation_is_blocked_while_pipeline_is_running(tmp_path: Path):
    root = _root_settings(tmp_path)
    profiles = ProfileManager(root)
    profiles.create_or_activate("srbmaury")
    profiles.create_or_activate("other_player", activate=False)
    release = threading.Event()

    def blocking_runner(settings, progress):
        progress({"message": "working"})
        release.wait(timeout=2)
        return {"ok": True}

    pipeline = PipelineManager(root, runners={"sync": blocking_runner})
    app = create_app(root, pipeline_manager=pipeline)
    enable_profiles(app, root, manager=profiles)
    client = TestClient(app)

    started = client.post("/api/pipeline/sync")
    assert started.status_code == 202
    assert started.json()["username"] == "srbmaury"

    switched = client.post("/api/profiles/other_player/activate")
    assert switched.status_code == 409

    release.set()
