from collections.abc import Iterator
from logging import getLogger

import pytest

from invokeai.app.services.intermediates.intermediates_records_sqlite import IntermediatesRecordsSqlite, MediaKind
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_22_add_intermediates_management import (
    build_migration as build_intermediates_migration,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_22_index_unmeasured_intermediates import (
    build_migration,
)


@pytest.fixture
def db() -> Iterator[SqliteDatabase]:
    database = SqliteDatabase(db_path=None, logger=getLogger(__name__))
    with database.transaction() as cursor:
        cursor.execute("CREATE TABLE session_queue (item_id INTEGER PRIMARY KEY);")
        for table, kind in (("images", "image"), ("videos", "video")):
            cursor.execute(
                f"""CREATE TABLE {table} (
                    {kind}_name TEXT PRIMARY KEY,
                    {kind}_subfolder TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT 'system',
                    is_intermediate BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TEXT NOT NULL DEFAULT '2020-01-01 00:00:00.000'
                );"""
            )
        # Existing installations have already applied this migration before receiving the index.
        build_intermediates_migration(getLogger(__name__)).callback(cursor)
        build_migration().callback(cursor)
    yield database
    database._conn.close()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_late_measurement_batches_do_not_rescan_completed_work(db: SqliteDatabase, kind: MediaKind) -> None:
    table = f"{kind}s"
    with db.transaction() as cursor:
        cursor.executemany(
            f"INSERT INTO {table} ({kind}_name) VALUES (?);",
            [(f"asset-{index}",) for index in range(100_000)],
        )
    records = IntermediatesRecordsSqlite(db)

    def next_batch() -> tuple[list[tuple[str, str]], int]:
        instructions = 0

        def count_instructions() -> int:
            nonlocal instructions
            instructions += 100
            return 0

        db._conn.set_progress_handler(count_instructions, 100)
        try:
            return records.next_unmeasured(kind, 200, min_age_seconds=10), instructions
        finally:
            db._conn.set_progress_handler(None, 0)

    first, first_work = next_batch()
    assert first == [(f"asset-{index}", "") for index in range(200)]
    with db.transaction() as cursor:
        cursor.execute(f"UPDATE {table} SET file_size_bytes = 100 WHERE rowid <= 99800;")
    late, late_work = next_batch()
    assert late == [(f"asset-{index}", "") for index in range(99800, 100_000)]
    # SQLite VM work is deterministic enough for a generous growth bound and independent of
    # machine speed. A table scan exceeds this by two orders of magnitude on the late batch.
    assert 0 < first_work < 10_000
    assert late_work < 2 * first_work


@pytest.mark.parametrize("kind", ["image", "video"])
def test_backfill_preserves_young_and_unmeasurable_rows(db: SqliteDatabase, kind: MediaKind) -> None:
    table = f"{kind}s"
    with db.transaction() as cursor:
        cursor.executemany(
            f"INSERT INTO {table} ({kind}_name, is_intermediate, file_size_bytes, created_at) VALUES (?, ?, ?, ?);",
            [
                ("durable", False, None, "2020-01-01"),
                ("measured", True, 100, "2020-01-01"),
                ("unreadable", True, None, "2020-01-01"),
                ("young", True, None, "2999-01-01"),
                ("z-first", True, None, "2020-01-01"),
                ("a-second", True, None, "2020-01-01"),
            ],
        )
        # The index migration can be replayed without rewriting measurement state.
        build_migration().callback(cursor)
    records = IntermediatesRecordsSqlite(db)
    records.mark_unmeasurable(kind, ["unreadable"])

    assert records.next_unmeasured(kind, 200, min_age_seconds=10) == [("z-first", ""), ("a-second", "")]
    with db.transaction() as cursor:
        cursor.execute(f"UPDATE {table} SET file_size_bytes = 100 WHERE {kind}_name IN ('z-first', 'a-second');")
        cursor.execute(f"UPDATE {table} SET created_at = '2020-01-01' WHERE {kind}_name = 'young';")
    assert records.next_unmeasured(kind, 200, min_age_seconds=10) == [("young", "")]
