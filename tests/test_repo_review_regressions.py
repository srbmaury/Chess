from pathlib import Path

import pytest

from chess_ml_coach.chesscom import ChessComClient
from chess_ml_coach.config import Settings
from chess_ml_coach.web.pipeline import PipelineManager
from chess_ml_coach.web.serve import _frontend_source_fingerprint, create_served_app


class _TrackingLock:
    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_pipeline_record_event_owns_its_lock(tmp_path: Path):
    manager = PipelineManager(
        Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    )
    lock = _TrackingLock()
    manager._lock = lock  # type: ignore[assignment]

    manager._record_event({"status": "test"})

    assert lock.entries == 1


def test_default_chesscom_user_agent_is_not_tied_to_project_author():
    client = ChessComClient()
    try:
        user_agent = client.http.headers["User-Agent"].lower()
        assert "chess-ml-coach" in user_agent
        assert "srbmaury" not in user_agent
    finally:
        client.http.close()


def test_frontend_fingerprint_invalidates_when_build_config_changes(tmp_path: Path):
    web_root = tmp_path / "web"
    source = web_root / "src"
    dist = web_root / "dist"
    source.mkdir(parents=True)
    dist.mkdir()
    (source / "App.tsx").write_text("export default function App(){return null}", encoding="utf-8")
    (web_root / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
    (web_root / "package.json").write_text('{"scripts":{"build":"vite build"}}', encoding="utf-8")
    vite_config = web_root / "vite.config.ts"
    vite_config.write_text("export default {}", encoding="utf-8")
    (dist / "index.html").write_text("<html><body>fresh build</body></html>", encoding="utf-8")
    (dist / ".source-fingerprint").write_text(
        _frontend_source_fingerprint(web_root),
        encoding="utf-8",
    )

    vite_config.write_text("export default { base: '/changed/' }", encoding="utf-8")

    with pytest.raises(RuntimeError, match="frontend build is stale"):
        create_served_app(
            Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models"),
            static_dir=dist,
            source_dir=web_root,
        )
