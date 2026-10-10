"""Board visibility and shares become general access and members, from 6.13 and alpha.1 data."""

from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_members
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator

MIGRATION_ID = "2026_10_10_board_roles"


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
def migrations(tmp_path: Path) -> list[MigrationBase]:
    config = DefaultInvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    config._root = tmp_path
    return build_migrations(MigrationBuildContext(app_config=config, logger=Logger("test"), image_files=MagicMock()))


@pytest.fixture
def before(migrations: list[MigrationBase]) -> Database:
    """A database just before this migration, holding boards as 6.13 and the alpha left them."""
    db = Database.open_sqlite(None, Logger("test"))
    assert _run(db, _previous_state(migrations))
    with db.begin(write=True) as conn:
        conn.exec_driver_sql(
            "INSERT INTO users (user_id, email, display_name, password_hash) VALUES (?, ?, ?, 'x');",
            [("alice", "alice@example.com", "Alice"), ("bob", "bob@example.com", "Bob")],
        )
        conn.exec_driver_sql(
            "INSERT INTO boards (board_id, board_name, user_id, is_public, board_visibility) VALUES (?, ?, ?, ?, ?);",
            [
                ("private", "Private", "alice", 0, "private"),
                ("shared", "Shared", "alice", 0, "shared"),
                ("public", "Public", "alice", 1, "public"),
                ("lent", "Lent", "alice", 0, "private"),
            ],
        )
        conn.exec_driver_sql(
            "INSERT INTO shared_boards (board_id, user_id, can_edit, shared_at) VALUES (?, ?, ?, ?);",
            [
                ("lent", "bob", 1, "2026-01-02 03:04:05.000"),
                # A share naming the board's own owner grants nothing.
                ("lent", "alice", 0, "2026-01-02 03:04:05.000"),
            ],
        )
    return db


def _general_access(db: Database) -> dict[str, str]:
    return {row[0]: row[1] for row in db.sqlite.conn.execute("SELECT board_id, general_access FROM boards;")}


def _members(db: Database) -> list[tuple[str, str, str, str | None, str]]:
    rows = db.sqlite.conn.execute(
        "SELECT board_id, user_id, role, granted_by, granted_at FROM board_members ORDER BY board_id, user_id;"
    ).fetchall()
    return [tuple(row) for row in rows]


def _columns(db: Database) -> set[str]:
    return {row[1] for row in db.sqlite.conn.execute("PRAGMA table_info(boards);").fetchall()}


def test_visibility_becomes_general_access_and_shares_become_viewer_members(
    before: Database, migrations: list[MigrationBase]
) -> None:
    assert _run(before, migrations)

    assert _general_access(before) == {
        "private": "restricted",
        "shared": "viewer",
        "public": "contributor",
        "lent": "restricted",
    }
    assert _members(before) == [("lent", "bob", "viewer", None, "2026-01-02 03:04:05.000")]
    assert not {"board_visibility", "is_public"} & _columns(before)
    tables = {row[0] for row in before.sqlite.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    assert "shared_boards" not in tables
    assert not _run(before, migrations)
    before.dispose()


def test_general_access_and_roles_are_checked(before: Database, migrations: list[MigrationBase]) -> None:
    assert _run(before, migrations)

    with pytest.raises(Exception, match="CHECK"):
        before.sqlite.conn.execute("UPDATE boards SET general_access = 'editor' WHERE board_id = 'private';")
    with pytest.raises(Exception, match="CHECK"):
        before.sqlite.conn.execute(
            "INSERT INTO board_members (board_id, user_id, role) VALUES ('private', 'bob', 'owner');"
        )
    before.dispose()


def test_a_run_that_stopped_halfway_finishes_the_work(before: Database, migrations: list[MigrationBase]) -> None:
    """On a server each DDL statement commits on its own: a failed run can leave the new column behind, still at
    its default, with everything else left to do."""
    with before.begin(write=True) as conn:
        conn.exec_driver_sql("ALTER TABLE boards ADD COLUMN general_access VARCHAR(32) NOT NULL DEFAULT 'restricted';")

    assert _run(before, migrations)

    assert _general_access(before)["public"] == "contributor"
    assert _members(before) == [("lent", "bob", "viewer", None, "2026-01-02 03:04:05.000")]
    assert not {"board_visibility", "is_public"} & _columns(before)
    before.dispose()


def test_a_run_that_stopped_after_copying_members_does_not_copy_them_twice(
    before: Database, migrations: list[MigrationBase]
) -> None:
    """On a server the new table commits as it is created: a run that failed after the copy leaves the members in
    it and `shared_boards` still there."""
    with before.begin(write=True) as conn:
        board_members.create(conn)
        conn.exec_driver_sql(
            "INSERT INTO board_members (board_id, user_id, role, granted_at) "
            "VALUES ('lent', 'bob', 'viewer', '2026-01-02 03:04:05.000');"
        )

    assert _run(before, migrations)

    assert _members(before) == [("lent", "bob", "viewer", None, "2026-01-02 03:04:05.000")]
    before.dispose()
