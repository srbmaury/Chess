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
    from chess_ml_coach import cli

    seen = {}

    def fake_run(settings, progress=None):
        seen["settings"] = settings
        if progress:
            progress(
                {
                    "stage": "sync",
                    "current": 1,
                    "total": 2,
                    "archive": "https://api.chess.com/pub/player/srbmaury/games/2026/08",
                    "game_count": 10,
                    "skipped": False,
                }
            )
        return {"downloaded": 3, "total": 10, "pgn_path": settings.data_dir / "raw" / "x.pgn"}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert seen["settings"].username == "srbmaury"
    assert seen["settings"].data_dir == tmp_path
    assert "[sync] 1/2" in result.stdout
    assert "10 games" in result.stdout
    assert "3 new games" in result.stdout


def test_sync_command_accepts_username_override(monkeypatch):
    from chess_ml_coach import cli

    seen = {}

    def fake_run(settings, progress=None):
        seen["username"] = settings.username
        return {"downloaded": 0, "total": 0, "pgn_path": Path("games.pgn")}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--username", "another-user"])
    assert result.exit_code == 0
    assert seen["username"] == "another-user"


def test_analyze_surfaces_actionable_stockfish_error(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        raise RuntimeError("Stockfish is not configured. Set STOCKFISH_PATH.")

    monkeypatch.setattr(cli, "_run_analyze", fake_run)
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 1
    assert "STOCKFISH_PATH" in result.output


def test_analyze_command_prints_move_progress(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        assert progress is not None
        progress(
            {
                "stage": "analyze",
                "completed": 43,
                "total": 130,
                "reused": 10,
                "analyzed": 33,
                "game_id": "g1",
                "ply": 46,
            }
        )
        return {"rows": 130, "output": Path("data/engine/analysis.parquet")}

    monkeypatch.setattr(cli, "_run_analyze", fake_run)
    result = runner.invoke(app, ["analyze", "--depth", "1"])

    assert result.exit_code == 0
    assert "[analyze] 43/130" in result.stdout
    assert "10 reused" in result.stdout
    assert "33 new" in result.stdout


def test_train_command_prints_stage_progress(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        assert progress is not None
        progress({"stage": "train", "message": "Training LightGBM"})
        return {
            "model_path": Path("models/mistake_model.joblib"),
            "metadata_path": Path("models/mistake_model.metadata.json"),
            "metrics": {"roc_auc": 0.75},
        }

    monkeypatch.setattr(cli, "_run_train", fake_run)
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    assert "[train] Training LightGBM" in result.stdout
    assert "Saved model" in result.stdout
