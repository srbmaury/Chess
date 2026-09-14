import threading
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.profiles import ProfileManager
from chess_ml_coach.web.app import create_app
from chess_ml_coach.web.pipeline import PipelineManager
from chess_ml_coach.web.profile_routes import enable_profiles


def _root_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")


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
