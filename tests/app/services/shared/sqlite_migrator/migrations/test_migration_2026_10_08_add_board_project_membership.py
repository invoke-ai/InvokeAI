"""Existing boards gain membership: every inbox joins its project, everything else is Library."""

from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import delete, insert, inspect, select

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import boards
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from tests.fixtures.database import migrate_to_newest

MIGRATION_ID = "2026_10_08_add_board_project_membership"


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


def _has_column(db: Database) -> bool:
    return any(row[1] == "project_id" for row in db.sqlite.conn.execute("PRAGMA table_info(boards);").fetchall())


def _has_index(db: Database) -> bool:
    return (
        db.sqlite.conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_boards_project_id';").fetchone()
        is not None
    )


def _memberships(db: Database) -> dict[str, str | None]:
    rows = db.sqlite.conn.execute("SELECT board_id, project_id FROM boards ORDER BY board_id;").fetchall()
    return {row[0]: row[1] for row in rows}


def test_inboxes_join_their_projects_and_other_boards_stay_in_the_library(tmp_path: Path) -> None:
    logger = Logger("test")
    config = DefaultInvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    config._root = tmp_path
    context = MigrationBuildContext(app_config=config, logger=logger, image_files=MagicMock())
    migrations = build_migrations(context)
    db = Database.open_sqlite(None, logger)
    assert _run(db, _previous_state(migrations))
    assert not _has_column(db)

    # Two accounts, each with a project: every inbox joins the project that names it.
    alice = UserService(database=db).create(
        UserCreateRequest(email="alice@example.com", display_name="Alice", password="TestPass123", is_admin=False)
    )
    bob = UserService(database=db).create(
        UserCreateRequest(email="bob@example.com", display_name="Bob", password="TestPass123", is_admin=False)
    )
    with db.begin(write=True) as conn:
        conn.exec_driver_sql(
            "INSERT INTO boards (board_id, board_name, user_id) VALUES (?, ?, ?);",
            [
                ("alice-inbox", "Alice's project", alice.user_id),
                ("bob-inbox", "Bob's project", bob.user_id),
                ("alice-loose", "References", alice.user_id),
                ("system-loose", "Legacy", "system"),
            ],
        )
        conn.exec_driver_sql(
            "INSERT INTO projects (project_id, user_id, name, data, board_id) VALUES (?, ?, ?, '{}', ?);",
            [
                ("alice-project", alice.user_id, "Alice's project", "alice-inbox"),
                ("bob-project", bob.user_id, "Bob's project", "bob-inbox"),
            ],
        )

    assert _run(db, migrations)
    assert _has_column(db)
    assert _memberships(db) == {
        "alice-inbox": "alice-project",
        "alice-loose": None,
        "bob-inbox": "bob-project",
        "system-loose": None,
    }
    assert _has_index(db)
    assert not _run(db, migrations)
    db.dispose()


def test_rerunning_leaves_existing_memberships_alone(database: Database, tmp_path: Path) -> None:
    """A retry fills missing inbox membership, preserving existing member and inbox values."""
    with database.begin(write=True) as conn:
        conn.execute(
            insert(boards),
            [
                {"board_id": "inbox", "board_name": "P", "user_id": "system", "project_id": "other"},
                {"board_id": "member", "board_name": "M", "user_id": "system", "project_id": "elsewhere"},
                {"board_id": "unfilled-inbox", "board_name": "U", "user_id": "system", "project_id": None},
            ],
        )
        conn.execute(
            insert(projects),
            [
                {"project_id": "p", "user_id": "system", "name": "P", "data": "{}", "board_id": "inbox"},
                {"project_id": "u", "user_id": "system", "name": "U", "data": "{}", "board_id": "unfilled-inbox"},
            ],
        )
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))
    migrate_to_newest(database, tmp_path)
    with database.begin(write=False) as conn:
        assert dict(conn.execute(select(boards.c.board_id, boards.c.project_id)).all()) == {
            "inbox": "other",
            "member": "elsewhere",
            "unfilled-inbox": "u",
        }
        assert "idx_boards_project_id" in {index["name"] for index in inspect(conn).get_indexes("boards")}
