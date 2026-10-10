"""Projects become keyed by id alone: a shared id is re-keyed everywhere, and memberships no project resolves go."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator

MIGRATION_ID = "2026_10_10_global_project_ids"
LOGGER = "test_global_project_ids"


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
    return build_migrations(
        MigrationBuildContext(app_config=config, logger=logging.getLogger(LOGGER), image_files=MagicMock())
    )


@pytest.fixture
def before(migrations: list[MigrationBase]) -> Database:
    """Alice and Bob each hold a project called `dup`. Bob's is the older one although Alice sorts first and her row
    was written first; a board of Alice's names a project that went with a deleted account."""
    db = Database.open_sqlite(None, logging.getLogger(LOGGER))
    assert _run(db, _previous_state(migrations))
    with db.begin(write=True) as conn:
        execute = conn.exec_driver_sql
        execute(
            "INSERT INTO users (user_id, email, display_name, password_hash) VALUES (?, ?, ?, 'x');",
            [("alice", "alice@example.com", "Alice"), ("bob", "bob@example.com", "Bob")],
        )
        execute(
            "INSERT INTO boards (board_id, board_name, user_id, project_id) VALUES (?, ?, ?, ?);",
            [
                ("alice-inbox", "Dup", "alice", "dup"),
                ("bob-inbox", "Dup", "bob", "dup"),
                ("bob-member", "Refs", "bob", "dup"),
                ("alice-stray", "Stray", "alice", "gone"),
            ],
        )
        execute(
            "INSERT INTO projects (project_id, user_id, name, data, board_id, revision, created_at, updated_at, "
            "minimum_canvas_schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
            [
                (
                    "dup",
                    "alice",
                    "Alice's dup",
                    json.dumps({"id": "dup", "canvas": {"layers": [1]}}),
                    "alice-inbox",
                    7,
                    "2026-02-01 00:00:00.000",
                    "2026-03-01 00:00:00.000",
                    3,
                ),
                (
                    "dup",
                    "bob",
                    "Bob's dup",
                    json.dumps({"id": "dup", "name": "dup"}),
                    "bob-inbox",
                    4,
                    "2026-01-01 00:00:00.000",
                    "2026-01-02 00:00:00.000",
                    2,
                ),
            ],
        )
        for user_id in ("alice", "bob"):
            execute(
                "INSERT INTO images (image_name, image_origin, image_category, width, height, user_id, project_id) "
                "VALUES (?, 'internal', 'general', 1, 1, ?, 'dup');",
                (f"{user_id}.png", user_id),
            )
            execute(
                "INSERT INTO videos (video_name, video_origin, video_category, width, height, user_id, project_id) "
                "VALUES (?, 'internal', 'general', 1, 1, ?, 'dup');",
                (f"{user_id}.mp4", user_id),
            )
            execute(
                "INSERT INTO session_queue (batch_id, queue_id, session_id, session, user_id, project_id, origin) "
                "VALUES ('b', 'default', ?, '{}', ?, 'dup', 'webv2:p:dup:q:1');",
                (f"{user_id}-session", user_id),
            )
            execute(
                "INSERT INTO media_references (owner_kind, user_id, owner_id, media_kind, media_name) "
                "VALUES ('project', ?, 'dup', 'image', ?);",
                (user_id, f"{user_id}.png"),
            )
            execute(
                "INSERT INTO client_state (user_id, key, value) VALUES (?, 'webv2:workbench-account', ?);",
                (user_id, json.dumps({"activeProjectId": "dup", "openProjectIds": ["dup", "other"]})),
            )
            execute(
                "INSERT INTO client_state (user_id, key, value) VALUES (?, 'webv2:project-covers', ?);",
                (user_id, json.dumps({"coverImageNames": {"dup": f"{user_id}.png"}})),
            )
    return db


def _scalar(db: Database, sql: str, *parameters: str) -> object:
    return db.sqlite.conn.execute(sql, parameters).fetchone()[0]


def _projects(db: Database) -> dict[str, tuple[object, ...]]:
    rows = db.sqlite.conn.execute(
        "SELECT user_id, project_id, name, data, board_id, revision, created_at, updated_at, "
        "minimum_canvas_schema_version FROM projects;"
    ).fetchall()
    return {row[0]: tuple(row[1:]) for row in rows}


def _client_state(db: Database, user_id: str) -> dict[str, object]:
    rows = db.sqlite.conn.execute("SELECT key, value FROM client_state WHERE user_id = ?;", (user_id,)).fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def test_a_shared_id_stays_with_the_oldest_project_and_the_other_is_rekeyed_everywhere(
    before: Database, migrations: list[MigrationBase], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(before, migrations)

    projects = _projects(before)
    alice_id = projects["alice"][0]
    assert projects["bob"][0] == "dup"
    assert isinstance(alice_id, str) and alice_id.startswith("project-")
    # Every value survives the rebuild; the re-keyed document names its new id.
    assert projects == {
        "alice": (
            alice_id,
            "Alice's dup",
            json.dumps({"id": alice_id, "canvas": {"layers": [1]}}),
            "alice-inbox",
            7,
            "2026-02-01 00:00:00.000",
            "2026-03-01 00:00:00.000",
            3,
        ),
        "bob": (
            "dup",
            "Bob's dup",
            json.dumps({"id": "dup", "name": "dup"}),
            "bob-inbox",
            4,
            "2026-01-01 00:00:00.000",
            "2026-01-02 00:00:00.000",
            2,
        ),
    }

    for user_id, expected in (("alice", alice_id), ("bob", "dup")):
        for sql in (
            "SELECT project_id FROM images WHERE user_id = ?;",
            "SELECT project_id FROM videos WHERE user_id = ?;",
            "SELECT project_id FROM session_queue WHERE user_id = ?;",
            "SELECT owner_id FROM media_references WHERE user_id = ?;",
        ):
            assert _scalar(before, sql, user_id) == expected, sql
        assert _scalar(before, "SELECT origin FROM session_queue WHERE user_id = ?;", user_id) == (
            f"webv2:p:{expected}:q:1"
        )
        assert _client_state(before, user_id) == {
            "webv2:workbench-account": {"activeProjectId": expected, "openProjectIds": [expected, "other"]},
            "webv2:project-covers": {"coverImageNames": {expected: f"{user_id}.png"}},
        }
    boards = dict(before.sqlite.conn.execute("SELECT board_id, project_id FROM boards;").fetchall())
    assert boards == {"alice-inbox": alice_id, "bob-inbox": "dup", "bob-member": "dup", "alice-stray": None}
    assert f"the copy owned by alice is now {alice_id}" in caplog.text
    assert "1 boards named a project that no longer exists" in caplog.text
    before.dispose()


def test_a_run_that_stopped_after_the_data_changes_finishes_the_work(
    before: Database, migrations: list[MigrationBase]
) -> None:
    """On a server the data changes commit with the first DDL: a failed run can leave the owner index created and
    the per-account key still in place, with no duplicate left to re-key."""
    with before.begin(write=True) as conn:
        conn.exec_driver_sql("UPDATE projects SET project_id = 'renamed' WHERE user_id = 'alice';")
        conn.exec_driver_sql("UPDATE boards SET project_id = 'renamed' WHERE board_id = 'alice-inbox';")
        conn.exec_driver_sql("CREATE INDEX idx_projects_user_id ON projects (user_id);")

    assert _run(before, migrations)

    assert {user_id: values[0] for user_id, values in _projects(before).items()} == {"alice": "renamed", "bob": "dup"}
    key = [row[1] for row in before.sqlite.conn.execute("PRAGMA table_info(projects);").fetchall() if row[5]]
    assert key == ["project_id"]
    before.dispose()
