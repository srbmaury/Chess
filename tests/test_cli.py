from pathlib import Path

from typer.testing import CliRunner

from chess_ml_coach.cli import app
from chess_ml_coach.config import MoveQualityThresholds, Settings

runner = CliRunner()


def test_default_settings_are_personalized_but_local(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    assert settings.username == "srbmaury"
    assert settings.thresholds == MoveQualityThresholds(50, 100, 200)
    assert settings.stockfish_depth == 14


def test_cli_exposes_all_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["sync", "analyze", "features", "train", "report"]:
        assert command in result.stdout


def test_sync_command_uses_default_username_and_data_dir(monkeypatch, tmp_path: Path):
    import chess_ml_coach.cli as cli

    seen = {}

    def fake_run(settings):
        seen["settings"] = settings
        return {"downloaded": 3, "total": 10, "pgn_path": settings.data_dir / "raw" / "x.pgn"}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert seen["settings"].username == "srbmaury"
    assert seen["settings"].data_dir == tmp_path
    assert "3 new games" in result.stdout


def test_sync_command_accepts_username_override(monkeypatch):
    import chess_ml_coach.cli as cli

    seen = {}

    def fake_run(settings):
        seen["username"] = settings.username
        return {"downloaded": 0, "total": 0, "pgn_path": Path("games.pgn")}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--username", "another-user"])
    assert result.exit_code == 0
    assert seen["username"] == "another-user"


def test_analyze_surfaces_actionable_stockfish_error(monkeypatch):
    import chess_ml_coach.cli as cli

    def fake_run(settings):
        raise RuntimeError("Stockfish is not configured. Set STOCKFISH_PATH.")

    monkeypatch.setattr(cli, "_run_analyze", fake_run)
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 1
    assert "STOCKFISH_PATH" in result.output
