"""Migration 28: Add image_subfolder column to images table.

This migration adds an image_subfolder column to the images table to support
configurable image subfolder strategies (flat, date, type, hash).
Existing images get an empty string (flat/root directory).
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration28Callback:
    """Migration to add image_subfolder column to images table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_image_subfolder_column(cursor)

    def _add_image_subfolder_column(self, cursor: sqlite3.Cursor) -> None:
        """Add image_subfolder column to images table."""
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(images);")
        columns = [row[1] for row in cursor.fetchall()]

        if "image_subfolder" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN image_subfolder TEXT NOT NULL DEFAULT '';")


def build_migration_28() -> Migration:
    """Builds the migration object for migrating from version 27 to version 28.

    This migration adds an image_subfolder column to the images table.
    """
    return Migration(
        from_version=27,
        to_version=28,
        callback=Migration28Callback(),
    )
