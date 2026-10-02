import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_04_add_queue_enqueue_receipts import (
    AddQueueEnqueueReceiptsCallback,
    build_migration,
)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _index_names(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA index_list(session_queue_enqueue_receipts)")}


def _column_names(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(session_queue_enqueue_receipts)")}


def _index_columns(connection: sqlite3.Connection, index_name: str) -> list[str]:
    return [row[2] for row in connection.execute(f"PRAGMA index_info({index_name})")]


def test_adds_queue_enqueue_receipts_table_and_retention_index() -> None:
    connection = sqlite3.connect(":memory:")

    AddQueueEnqueueReceiptsCallback()(connection.cursor())

    assert "session_queue_enqueue_receipts" in _table_names(connection)
    assert "idx_session_queue_enqueue_receipts_owner_ack" in _index_names(connection)
    assert _index_columns(connection, "idx_session_queue_enqueue_receipts_owner_ack") == [
        "user_id",
        "acknowledged_at",
        "byte_size",
    ]
    assert {"acknowledged_at", "byte_size"}.issubset(_column_names(connection))


def test_receipts_are_deleted_with_their_owner() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY);")
    connection.execute("INSERT INTO users (user_id) VALUES ('user-1');")
    AddQueueEnqueueReceiptsCallback()(connection.cursor())
    connection.execute(
        """
        INSERT INTO session_queue_enqueue_receipts (
            queue_id, user_id, idempotency_key, payload_hash, batch_id,
            requested, enqueued, priority, item_ids, byte_size
        ) VALUES ('default', 'user-1', 'key', 'hash', 'batch', 1, 1, 0, '[1]', 1);
        """
    )

    connection.execute("DELETE FROM users WHERE user_id = 'user-1';")

    assert connection.execute("SELECT COUNT(*) FROM session_queue_enqueue_receipts;").fetchone()[0] == 0


def test_migration_is_idempotent() -> None:
    connection = sqlite3.connect(":memory:")
    callback = AddQueueEnqueueReceiptsCallback()

    callback(connection.cursor())
    callback(connection.cursor())

    assert "session_queue_enqueue_receipts" in _table_names(connection)


def test_builder_metadata() -> None:
    migration = build_migration()

    assert migration.id == "2026_09_04_add_queue_enqueue_receipts"
    assert migration.depends_on == "2026_07_01_add_session_queue_device"
    assert migration.from_version is None
    assert migration.to_version is None
