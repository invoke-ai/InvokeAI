"""Existing projects need document schema 3 and have no workspaces yet; workspaces go with their project and account."""

from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator

MIGRATION_ID = "2026_10_10_project_workspaces"


def _run(db: Database, migrations: list[MigrationBase]) -> bool:
    migrator = Migrator(db)
    for migration in migrations:
        migrator.register_migration(migration)
    return migrator.run_migrations()


def _previous_state(migrations: list[MigrationBase]) -> list[MigrationBase]:
    """Every migration except this one and those that depend on it, directly or not."""
    by_id = {migration.id: migration for migration in migrations}

    def depends_on_this(migration: MigrationBase) -> bool:
        current: str | None = migration.id
        while current is not None:
            if current == MIGRATION_ID:
                return True
            current = by_id[current].depends_on
        return False

    return [migration for migration in migrations if not depends_on_this(migration)]


@pytest.fixture
def migrated(tmp_path: Path) -> Database:
    """A project of Alice's from before this migration, then the migration."""
    config = DefaultInvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    config._root = tmp_path
    migrations = build_migrations(
        MigrationBuildContext(app_config=config, logger=Logger("test"), image_files=MagicMock())
    )
    db = Database.open_sqlite(None, Logger("test"))
    assert _run(db, _previous_state(migrations))
    with db.begin(write=True) as conn:
        conn.exec_driver_sql(
            "INSERT INTO users (user_id, email, display_name, password_hash) "
            "VALUES ('alice', 'alice@example.com', 'Alice', 'x');"
        )
        conn.exec_driver_sql("INSERT INTO boards (board_id, board_name, user_id) VALUES ('inbox', 'P', 'alice');")
        conn.exec_driver_sql(
            "INSERT INTO projects (project_id, user_id, name, data, board_id) VALUES ('p', 'alice', 'P', '{}', 'inbox');"
        )
    assert _run(db, migrations)
    assert not _run(db, migrations)
    return db


def test_existing_projects_need_document_schema_3_and_have_no_workspaces(migrated: Database) -> None:
    conn = migrated.sqlite.conn
    assert conn.execute("SELECT minimum_document_schema_version FROM projects;").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM project_workspaces;").fetchone()[0] == 0
    with pytest.raises(Exception, match="CHECK"):
        conn.execute("UPDATE projects SET minimum_document_schema_version = 0;")
    migrated.dispose()


@pytest.mark.parametrize("deleted", ["DELETE FROM projects;", "DELETE FROM users WHERE user_id = 'alice';"])
def test_a_workspace_goes_with_its_project_and_its_account(migrated: Database, deleted: str) -> None:
    conn = migrated.sqlite.conn
    conn.execute("INSERT INTO project_workspaces (project_id, user_id, data) VALUES ('p', 'alice', '{}');")
    conn.execute(deleted)
    assert conn.execute("SELECT COUNT(*) FROM project_workspaces;").fetchone()[0] == 0
    migrated.dispose()
