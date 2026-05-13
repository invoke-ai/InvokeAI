"""Migration 29: Add board_visibility column to boards table.

This migration adds a board_visibility column to the boards table to support
three visibility levels:
  - 'private': only the board owner (and admins) can view/modify
  - 'shared': all users can view, but only the owner (and admins) can modify
  - 'public': all users can view; only the owner (and admins) can modify the
    board structure (rename/archive/delete)

Existing boards with is_public = 1 are migrated to 'public'.
All other existing boards default to 'private'.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration29Callback:
    """Migration to add board_visibility column to the boards table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._update_boards_table(cursor)

    def _update_boards_table(self, cursor: sqlite3.Cursor) -> None:
        """Add board_visibility column to boards table."""
        # Check if boards table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='boards';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(boards);")
        columns = [row[1] for row in cursor.fetchall()]

        if "board_visibility" not in columns:
            cursor.execute("ALTER TABLE boards ADD COLUMN board_visibility TEXT NOT NULL DEFAULT 'private';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_boards_board_visibility ON boards(board_visibility);")
            # Migrate existing is_public = 1 boards to 'public'
            if "is_public" in columns:
                cursor.execute("UPDATE boards SET board_visibility = 'public' WHERE is_public = 1;")


def build_migration_29() -> Migration:
    """Builds the migration object for migrating from version 28 to version 29.

    This migration adds the board_visibility column to the boards table,
    supporting 'private', 'shared', and 'public' visibility levels.
    """
    return Migration(
        from_version=28,
        to_version=29,
        callback=Migration29Callback(),
    )
