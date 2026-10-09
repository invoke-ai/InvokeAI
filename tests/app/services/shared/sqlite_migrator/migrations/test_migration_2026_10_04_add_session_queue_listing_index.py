"""An existing queue gains the listing index without any queue read changing what it returns.

The index exists only on SQLite; on a server the migration does nothing.
"""

import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import delete, insert, inspect

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.session_queue import session_queue as session_queue_table
from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from invokeai.backend.util.logging import InvokeAILogger

MIGRATION_ID = "2026_10_04_add_session_queue_listing_index"
INDEX = "idx_session_queue_listing"
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


@pytest.fixture
def migrations(tmp_path: Path) -> list[MigrationBase]:
    # Migrations clean up legacy files under the root. No settings from the environment or a config file may
    # point them anywhere else.
    config = DefaultInvokeAIAppConfig()
    config._root = tmp_path
    context = MigrationBuildContext(
        app_config=config,
        logger=InvokeAILogger.get_logger("test_listing_index_migration"),
        image_files=mock.Mock(spec=ImageFileStorageBase),
    )
    return build_migrations(context)


@pytest.fixture
def sqlite_database() -> Iterator[Database]:
    database = Database.open_sqlite(None, InvokeAILogger.get_logger("test_listing_index_migration"))
    try:
        yield database
    finally:
        database.dispose()


def _run(database: Database, migrations: Sequence[MigrationBase]) -> bool:
    migrator = Migrator(database)
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


def _has_index(database: Database) -> bool:
    with database.begin(write=False) as conn:
        return any(index["name"] == INDEX for index in inspect(conn).get_indexes("session_queue"))


def _reads(queue: SessionQueue) -> dict[str, Any]:
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


def test_indexes_an_existing_queue_without_changing_its_reads(
    sqlite_database: Database, migrations: list[MigrationBase]
) -> None:
    assert _run(sqlite_database, _previous_state(migrations))
    assert not _has_index(sqlite_database)

    with sqlite_database.begin(write=True) as conn:
        conn.execute(
            insert(session_queue_table),
            [
                {
                    "queue_id": "default",
                    "session": "{}",
                    "session_id": str(uuid.uuid4()),
                    "batch_id": "batch",
                    "created_at": created_at,
                    "user_id": user_id,
                    "origin": origin,
                    "status": status,
                }
                for created_at, user_id, origin, status in ROWS
            ],
        )
    queue = SessionQueue(sqlite_database)
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

    assert _run(sqlite_database, migrations)
    assert _has_index(sqlite_database)
    assert _reads(queue) == before

    assert not _run(sqlite_database, migrations)
    # Run again with its record lost, as a run that failed after creating the index leaves it: the migration finds
    # the index and keeps it.
    with sqlite_database.begin(write=True) as conn:
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))
    assert _run(sqlite_database, migrations)
    assert _has_index(sqlite_database)
    assert _reads(queue) == before
