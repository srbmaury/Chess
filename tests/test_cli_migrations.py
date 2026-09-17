from typer.testing import CliRunner

from chess_ml_coach import cli

runner = CliRunner()


class FakeDatabase:
    opened = False
    closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True


def test_db_migrate_requires_database_url():
    result = runner.invoke(cli.app, ["db-migrate"], env={"DATABASE_URL": ""})

    assert result.exit_code != 0
    assert "DATABASE_URL" in result.stdout


def test_db_migrate_applies_and_closes(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(cli, "_database_factory", lambda _settings: database)
    monkeypatch.setattr(cli, "_apply_migrations", lambda _database: ("0001_hosted_schema",))

    result = runner.invoke(
        cli.app,
        ["db-migrate", "--database-url", "postgresql://example.invalid/chess"],
    )

    assert result.exit_code == 0
    assert "0001_hosted_schema" in result.stdout
    assert database.opened is True
    assert database.closed is True
