# ruff: noqa: I001

import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.web.app import create_app
from chess_ml_coach.web.pipeline import PipelineBusyError, PipelineManager


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        stockfish_depth=14,
    )


def _wait_for_terminal(manager: PipelineManager, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if snapshot.status in {"succeeded", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("pipeline job did not reach a terminal state")


def test_manager_records_monotonic_progress_and_success(tmp_path: Path):
    def fake_runner(settings, progress):
        assert settings.stockfish_depth == 14
        progress({"completed": 2, "total": 10, "reused": 1, "analyzed": 1})
        progress({"completed": 10, "total": 10, "reused": 3, "analyzed": 7})
        return {"rows": 10}

    manager = PipelineManager(_settings(tmp_path), runners={"analyze": fake_runner})

    started = manager.start("analyze")
    assert started.status in {"running", "succeeded"}
    finished = _wait_for_terminal(manager)

    assert finished.status == "succeeded"
    assert finished.stage == "analyze"
    assert finished.result == {"rows": 10}
    events = manager.events(after_sequence=0)
    sequences = [event.sequence for event in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    assert any(event.payload.get("completed") == 10 for event in events)


def test_manager_rejects_second_job_while_one_is_running(tmp_path: Path):
    release = threading.Event()

    def blocking_runner(settings, progress):
        progress({"message": "working"})
        release.wait(timeout=2)
        return {"ok": True}

    manager = PipelineManager(_settings(tmp_path), runners={"sync": blocking_runner})
    manager.start("sync")

    try:
        manager.start("sync")
    except PipelineBusyError as exc:
        assert "already running" in str(exc).lower()
    else:
        raise AssertionError("starting a second job should fail")
    release.set()
    assert _wait_for_terminal(manager).status == "succeeded"


def test_manager_exposes_failed_state_without_raising_from_worker(tmp_path: Path):
    def failing_runner(settings, progress):
        raise RuntimeError("Stockfish exploded")

    manager = PipelineManager(_settings(tmp_path), runners={"analyze": failing_runner})
    manager.start("analyze")
    finished = _wait_for_terminal(manager)

    assert finished.status == "failed"
    assert finished.error == "Stockfish exploded"
    assert manager.events(after_sequence=0)[-1].payload["status"] == "failed"


def test_analyze_depth_override_does_not_mutate_base_settings(tmp_path: Path):
    seen_depths = []

    def fake_runner(settings, progress):
        seen_depths.append(settings.stockfish_depth)
        return {"rows": 1}

    settings = _settings(tmp_path)
    manager = PipelineManager(settings, runners={"analyze": fake_runner})
    manager.start("analyze", {"depth": 7})
    assert _wait_for_terminal(manager).status == "succeeded"

    assert seen_depths == [7]
    assert settings.stockfish_depth == 14


def test_pipeline_api_rejects_invalid_and_concurrent_starts(tmp_path: Path):
    release = threading.Event()

    def blocking_runner(settings, progress):
        progress({"message": "working"})
        release.wait(timeout=2)
        return {"ok": True}

    manager = PipelineManager(_settings(tmp_path), runners={"sync": blocking_runner})
    client = TestClient(create_app(_settings(tmp_path), pipeline_manager=manager))

    invalid = client.post("/api/pipeline/not-a-stage")
    assert invalid.status_code == 404

    first = client.post("/api/pipeline/sync")
    assert first.status_code == 202
    second = client.post("/api/pipeline/sync")
    assert second.status_code == 409
    assert "already running" in second.json()["detail"].lower()

    release.set()
    assert _wait_for_terminal(manager).status == "succeeded"


def test_pipeline_sse_serializes_retained_events_and_closes_after_terminal(tmp_path: Path):
    def fake_runner(settings, progress):
        progress({"completed": 1, "total": 1, "reused": 0, "analyzed": 1})
        return {"rows": 1}

    manager = PipelineManager(_settings(tmp_path), runners={"analyze": fake_runner})
    client = TestClient(create_app(_settings(tmp_path), pipeline_manager=manager))
    response = client.post("/api/pipeline/analyze", json={"depth": 9})
    assert response.status_code == 202
    assert _wait_for_terminal(manager).status == "succeeded"

    events_response = client.get("/api/pipeline/events", params={"after_sequence": 0})
    assert events_response.status_code == 200
    assert events_response.headers["content-type"].startswith("text/event-stream")
    payloads = []
    for line in events_response.text.splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line.removeprefix("data: ")))

    assert payloads
    assert any(payload.get("completed") == 1 for payload in payloads)
    assert payloads[-1]["status"] == "succeeded"
