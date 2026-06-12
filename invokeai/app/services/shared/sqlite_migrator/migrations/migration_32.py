"""Migration 32: Add device column to session_queue table.

This records which device (e.g. 'cuda:1') processed a queue item, so the UI can show a per-item
GPU number in the Session Queue. Existing rows get NULL (unknown device).
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration32Callback:
    """Migration to add a device column to the session_queue table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(session_queue);")
        columns = [row[1] for row in cursor.fetchall()]

        if "device" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN device TEXT;")


def build_migration_32() -> Migration:
    """Builds the migration object for migrating from version 31 to version 32.

    This migration adds a device column to the session_queue table.
    """
    return Migration(
        from_version=31,
        to_version=32,
        callback=Migration32Callback(),
    )
