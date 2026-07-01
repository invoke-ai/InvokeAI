"""Tests for migration 2026_07_01_add_session_queue_device: add device column to session_queue."""

import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_01_add_session_queue_device import (
    AddSessionQueueDeviceCallback,
    build_migration,
)


def _create_session_queue_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE session_queue (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        """
    )


def _get_columns(conn: sqlite3.Connection) -> list[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(session_queue);").fetchall()]


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_session_queue_table(conn)
    return conn


class TestAddSessionQueueDevice:
    def test_adds_device_column(self, db: sqlite3.Connection):
        AddSessionQueueDeviceCallback()(db.cursor())
        db.commit()

        assert "device" in _get_columns(db)

    def test_existing_rows_get_null_device(self, db: sqlite3.Connection):
        db.execute("INSERT INTO session_queue (status) VALUES ('completed')")
        AddSessionQueueDeviceCallback()(db.cursor())
        db.commit()

        device = db.execute("SELECT device FROM session_queue").fetchone()[0]
        assert device is None

    def test_idempotent_when_column_exists(self, db: sqlite3.Connection):
        cursor = db.cursor()
        AddSessionQueueDeviceCallback()(cursor)
        AddSessionQueueDeviceCallback()(cursor)
        db.commit()

        assert _get_columns(db).count("device") == 1

    def test_tolerates_missing_session_queue_table(self):
        conn = sqlite3.connect(":memory:")
        AddSessionQueueDeviceCallback()(conn.cursor())

    def test_builder_metadata(self):
        migration = build_migration()
        assert migration.id == "2026_07_01_add_session_queue_device"
        assert migration.depends_on == "migration_30"
        assert migration.from_version is None
        assert migration.to_version is None
