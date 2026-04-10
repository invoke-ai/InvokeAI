"""Migration 28: Add per-item queue status sequencing.

This migration adds a `status_sequence` column to `session_queue` so queue item
status updates can be ordered across asynchronous event and snapshot channels.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration28Callback:
    """Add a per-queue-item status sequence for cross-channel ordering."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(session_queue);")
        columns = [row[1] for row in cursor.fetchall()]

        if "status_sequence" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN status_sequence INTEGER DEFAULT 0;")
            cursor.execute("UPDATE session_queue SET status_sequence = 0 WHERE status_sequence IS NULL;")


def build_migration_28() -> Migration:
    return Migration(
        from_version=27,
        to_version=28,
        callback=Migration28Callback(),
    )
