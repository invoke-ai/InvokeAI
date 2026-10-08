"""Existing boards gain membership: every inbox joins its project, everything else is Library."""

from logging import Logger
from unittest.mock import MagicMock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_10_08_add_board_project_membership import (
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService

MIGRATION_ID = "2026_10_08_add_board_project_membership"


def _run(db: SqliteDatabase, migrations: list[Migration]) -> bool:
    migrator = SqliteMigrator(db=db)
    for migration in migrations:
        migrator.register_migration(migration)
    return migrator.run_migrations()


def _previous_state(migrations: list[Migration]) -> list[Migration]:
    """Every migration except this one and those that depend on it, directly or not."""
    by_id = {migration.id: migration for migration in migrations}

    def depends_on_this(migration: Migration) -> bool:
        current: str | None = migration.id
        while current is not None:
            if current == MIGRATION_ID:
                return True
            current = by_id[current].depends_on
        return False

    return [migration for migration in migrations if not depends_on_this(migration)]


def _has_column(db: SqliteDatabase) -> bool:
    return any(row[1] == "project_id" for row in db._conn.execute("PRAGMA table_info(boards);").fetchall())


def _has_index(db: SqliteDatabase) -> bool:
    return db._conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_boards_project_id';").fetchone() is not None


def _memberships(db: SqliteDatabase) -> dict[str, str | None]:
    rows = db._conn.execute("SELECT board_id, project_id FROM boards ORDER BY board_id;").fetchall()
    return {row[0]: row[1] for row in rows}


def test_inboxes_join_their_projects_and_other_boards_stay_in_the_library() -> None:
    logger = Logger("test")
    context = MigrationBuildContext(
        app_config=InvokeAIAppConfig(use_memory_db=True, node_cache_size=0), logger=logger, image_files=MagicMock()
    )
    migrations = build_migrations(context)
    db = SqliteDatabase(db_path=None, logger=logger)
    assert _run(db, _previous_state(migrations))
    assert not _has_column(db)

    # Two accounts, each with a project: every inbox joins the project that names it.
    alice = UserService(db=db).create(
        UserCreateRequest(email="alice@example.com", display_name="Alice", password="TestPass123", is_admin=False)
    )
    bob = UserService(db=db).create(
        UserCreateRequest(email="bob@example.com", display_name="Bob", password="TestPass123", is_admin=False)
    )
    with db.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO boards (board_id, board_name, user_id) VALUES (?, ?, ?);",
            [
                ("alice-inbox", "Alice's project", alice.user_id),
                ("bob-inbox", "Bob's project", bob.user_id),
                ("alice-loose", "References", alice.user_id),
                ("system-loose", "Legacy", "system"),
            ],
        )
        cursor.executemany(
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


def test_rerunning_leaves_existing_memberships_alone() -> None:
    """The callback is safe to apply again: it only fills in what is missing, never overwrites.

    The inbox seeded as a member of `other` is a state the services never produce; it is here to prove
    the backfill is gated on `project_id IS NULL` rather than on being an inbox.
    """
    logger = Logger("test")
    context = MigrationBuildContext(
        app_config=InvokeAIAppConfig(use_memory_db=True, node_cache_size=0), logger=logger, image_files=MagicMock()
    )
    db = SqliteDatabase(db_path=None, logger=logger)
    assert _run(db, build_migrations(context))
    user = UserService(db=db).create(
        UserCreateRequest(email="carol@example.com", display_name="Carol", password="TestPass123", is_admin=False)
    )
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO boards (board_id, board_name, user_id, project_id) VALUES ('inbox', 'P', ?, 'other');",
            (user.user_id,),
        )
        cursor.execute(
            "INSERT INTO boards (board_id, board_name, user_id, project_id) VALUES ('member', 'M', ?, 'elsewhere');",
            (user.user_id,),
        )
        cursor.execute(
            "INSERT INTO boards (board_id, board_name, user_id) VALUES ('unfilled-inbox', 'U', ?);", (user.user_id,)
        )
        cursor.execute(
            "INSERT INTO projects (project_id, user_id, name, data, board_id) VALUES ('p', ?, 'P', '{}', 'inbox');",
            (user.user_id,),
        )
        cursor.execute(
            "INSERT INTO projects (project_id, user_id, name, data, board_id) VALUES ('u', ?, 'U', '{}', 'unfilled-inbox');",
            (user.user_id,),
        )

    with db.transaction() as cursor:
        build_migration().callback(cursor)

    assert _memberships(db) == {"inbox": "other", "member": "elsewhere", "unfilled-inbox": "u"}
