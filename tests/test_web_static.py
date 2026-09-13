from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from chess_ml_coach.cli import app
from chess_ml_coach.config import Settings
from chess_ml_coach.web.serve import create_served_app

runner = CliRunner()


def test_served_app_returns_spa_and_keeps_unknown_api_as_404(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>Chess UI</body></html>", encoding="utf-8")

    web_app = create_served_app(
        Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models"),
        static_dir=dist,
    )
    client = TestClient(web_app)

    assert client.get("/").status_code == 200
    assert "Chess UI" in client.get("/practice").text
    assert client.get("/api/does-not-exist").status_code == 404


def test_served_app_requires_built_frontend(tmp_path: Path):
    try:
        create_served_app(static_dir=tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "npm run build" in str(exc)
    else:
        raise AssertionError("expected missing frontend build to fail")


def test_cli_exposes_ui_and_defaults_to_localhost():
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "ui" in help_result.stdout

    ui_help = runner.invoke(app, ["ui", "--help"])
    assert ui_help.exit_code == 0
    assert "127.0.0.1" in ui_help.stdout
    assert "--no-open" in ui_help.stdout
