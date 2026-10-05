"""An existing queue gains the listing index without any queue read changing what it returns."""

import uuid
from logging import Logger
from typing import Any
from unittest.mock import MagicMock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_10_04_add_session_queue_listing_index import (
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator

MIGRATION_ID = "2026_10_04_add_session_queue_listing_index"
PREFIX = "webv2:p:project-1:q:"

# (created_at, user_id, origin, status), inserted in this order. One enqueue writes its rows within the
# same millisecond, so equal timestamps are ordinary; the fourth row's clock is behind the third's.
ROWS = [
    ("2026-10-01 10:00:00.000", "alice", f"{PREFIX}a", "completed"),
    ("2026-10-01 10:00:00.000", "bob", "webv2:q:b", "failed"),
    ("2026-10-01 10:00:00.000", "alice", f"{PREFIX}c", "completed"),
    ("2026-09-30 09:00:00.000", "bob", f"{PREFIX}d", "canceled"),
    ("2026-10-01 10:00:05.000", "alice", "webv2:util:e", "in_progress"),
    ("2026-10-01 10:00:05.000", "bob", f"{PREFIX}f", "pending"),
    ("2026-10-01 10:00:05.000", "alice", f"{PREFIX}g", "pending"),
]


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


def _has_index(db: SqliteDatabase) -> bool:
    row = db._conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_session_queue_listing';").fetchone()
    return row is not None


def _reads(queue: SqliteSessionQueue) -> dict[str, Any]:
    def ids(direction: SQLiteDirection, **filters: str) -> list[int]:
        return queue.get_queue_item_ids("default", direction, **filters).item_ids

    return {
        "desc": ids(SQLiteDirection.Descending),
        "asc": ids(SQLiteDirection.Ascending),
        "desc_origin": ids(SQLiteDirection.Descending, origin_prefix=PREFIX),
        "asc_origin": ids(SQLiteDirection.Ascending, origin_prefix=PREFIX),
        "desc_user": ids(SQLiteDirection.Descending, user_id="alice"),
        "asc_user": ids(SQLiteDirection.Ascending, user_id="alice"),
        "status": queue.get_queue_status("default", user_id="alice", acting_user_id="alice"),
        "status_origin": queue.get_queue_status("default", user_id="bob", acting_user_id="bob", origin_prefix=PREFIX),
        "summaries": [
            (summary.item_id, summary.status, summary.origin)
            for summary in queue.get_queue_item_summaries_by_ids("default", [3, 1, 99])
        ],
    }


def test_indexes_an_existing_queue_without_changing_its_reads() -> None:
    logger = Logger("test")
    context = MigrationBuildContext(
        app_config=InvokeAIAppConfig(use_memory_db=True, node_cache_size=0), logger=logger, image_files=MagicMock()
    )
    migrations = build_migrations(context)
    db = SqliteDatabase(db_path=None, logger=logger)
    assert _run(db, _previous_state(migrations))
    assert not _has_index(db)

    with db.transaction() as cursor:
        cursor.executemany(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, created_at, user_id, origin, status)
            VALUES ('default', '{}', ?, 'batch', ?, ?, ?, ?);
            """,
            [(str(uuid.uuid4()), *row) for row in ROWS],
        )
    queue = SqliteSessionQueue(db=db)
    before = _reads(queue)

    # Newest first; rows enqueued together keep their enqueue order in either direction.
    assert before["desc"] == [5, 6, 7, 1, 2, 3, 4]
    assert before["asc"] == [4, 1, 2, 3, 5, 6, 7]
    assert before["desc_origin"] == [6, 7, 1, 3, 4]
    assert before["asc_origin"] == [4, 1, 3, 6, 7]
    assert before["desc_user"] == [5, 7, 1, 3]
    assert before["asc_user"] == [1, 3, 5, 7]
    # In the requested order; an unknown id is skipped.
    assert before["summaries"] == [(3, "completed", f"{PREFIX}c"), (1, "completed", f"{PREFIX}a")]

    assert _run(db, migrations)
    assert _has_index(db)
    assert _reads(queue) == before

    assert not _run(db, migrations)
    build_migration().callback(db._conn.cursor())
    assert _reads(queue) == before
