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
