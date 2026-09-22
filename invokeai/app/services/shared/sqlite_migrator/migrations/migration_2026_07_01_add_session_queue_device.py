"""Migration: Add device column to session_queue table.

This records which device (e.g. 'cuda:1') processed a queue item, so the UI can show a per-item
GPU number in the Session Queue. Existing rows get NULL (unknown device).
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddSessionQueueDeviceCallback:
    """Migration to add a device column to the session_queue table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(session_queue);")
        columns = [row[1] for row in cursor.fetchall()]

        if "device" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN device TEXT;")


def build_migration() -> Migration:
    """Builds the migration that adds a device column to the session_queue table.

    Depends on migration_30, the most recent migration to alter the session_queue schema.
    """
    return Migration(
        id="2026_07_01_add_session_queue_device",
        depends_on="migration_30",
        callback=AddSessionQueueDeviceCallback(),
    )
