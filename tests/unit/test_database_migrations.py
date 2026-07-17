"""Regression tests for ordered and recoverable SQLite migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from olympianim.database import sqlite as database


def test_initialized_database_records_every_schema_migration(tmp_path: Path) -> None:
    path = tmp_path / "olympianim.db"
    database.initialize_database(path)

    with database.connect(path) as connection:
        versions = [
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]

    assert versions == list(range(1, database.LATEST_SCHEMA_VERSION + 1))


def test_unversioned_existing_database_is_backed_up_before_migration(tmp_path: Path) -> None:
    path = tmp_path / "olympianim.db"
    database.initialize_database(path)
    with database.connect(path) as connection:
        connection.execute("DELETE FROM schema_migrations")

    database.initialize_database(path)

    backups = list((tmp_path / ".backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_failed_migration_rolls_back_its_schema_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "olympianim.db"
    database.initialize_database(path)

    def broken(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE should_rollback (id INTEGER)")
        raise RuntimeError("migration failed")

    migration = database.SchemaMigration(
        database.LATEST_SCHEMA_VERSION + 1,
        "broken_test_migration",
        "test-checksum",
        broken,
    )
    monkeypatch.setattr(database, "MIGRATIONS", (*database.MIGRATIONS, migration))
    with database.connect(path) as connection:
        with pytest.raises(RuntimeError, match="migration failed"):
            database.migrate(connection)
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'should_rollback'"
        ).fetchone()
        recorded = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?", (migration.version,)
        ).fetchone()

    assert table is None
    assert recorded is None
