import sqlite3
from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_34 import build_migration_34
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_30_repair_projects_table import (
    RepairProjectsTableMigrationCallback,
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator


def _table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cursor.fetchone() is not None


def test_creates_the_projects_table_and_its_trigger() -> None:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    # Created by migration_27, which the projects FK references.
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")

    RepairProjectsTableMigrationCallback()(db.cursor())

    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute("INSERT INTO projects (project_id, user_id, name, data) VALUES ('p1', 'u1', 'n', '{}');")
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='tg_projects_updated_at';")
    assert cursor.fetchone() is not None
    db.close()


def test_is_idempotent_on_a_database_that_already_has_the_table() -> None:
    """The common case: a fork-originated database where migration_33 already ran. The repair must
    no-op rather than fail on an existing table/trigger, and must not disturb existing rows."""
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")
    RepairProjectsTableMigrationCallback()(db.cursor())
    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute("INSERT INTO projects (project_id, user_id, name, data) VALUES ('p1', 'u1', 'kept', '{}');")

    RepairProjectsTableMigrationCallback()(db.cursor())

    cursor = db.cursor()
    cursor.execute("SELECT name FROM projects;")
    assert cursor.fetchall() == [("kept",)]
    db.close()


def test_repairs_a_database_that_came_from_an_upstream_build(tmp_path: Path) -> None:
    """End-to-end regression guard for the migration_33 id collision.

    This fork's migration_33 creates `projects`; upstream's migration_33 creates the
    image-subfolder move tables (carried here as migration_34). They share one id, so a database
    that was ever opened by an upstream build records `migration_33` and this fork then skips its
    own projects migration permanently, leaving `no such table: projects`.

    The dated repair migration is immune to that collision and must fix such a database.
    """
    logger = Logger("test")
    context = MigrationBuildContext(app_config=MagicMock(), logger=logger, image_files=MagicMock())
    all_migrations = build_migrations(context)

    db = SqliteDatabase(db_path=tmp_path / "upstream.db", logger=logger, verbose=False)

    # 1. Bring the database to legacy version 32 using only the migrations at or below 32.
    to_v32 = SqliteMigrator(db=db)
    for migration in all_migrations:
        if migration.to_version is not None and migration.to_version <= 32:
            to_v32.register_migration(migration)
    to_v32.run_migrations()

    # 2. Simulate UPSTREAM's migration_33 having run. Its body is what this fork carries as
    #    migration_34, so run that and stamp the legacy version/id upstream would have recorded.
    cursor = db._conn.cursor()
    build_migration_34().callback(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (33);")
    cursor.execute("INSERT INTO applied_migrations (migration_id, legacy_version) VALUES ('migration_33', 33);")
    db._conn.commit()
    assert not _table_exists(cursor, "projects")

    # 3. Open it with the full fork migrator, as the app does on startup.
    full = SqliteMigrator(db=db)
    for migration in all_migrations:
        full.register_migration(migration)
    full.run_migrations()

    # Without the repair migration the fork's migration_33 stays skipped and this is still missing.
    assert _table_exists(cursor, "projects")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='tg_projects_updated_at';")
    assert cursor.fetchone() is not None


def test_migration_has_a_dated_id_so_it_cannot_collide_with_an_upstream_number() -> None:
    migration = build_migration()
    assert migration.id == "2026_07_30_repair_projects_table"
    assert migration.to_version is None
