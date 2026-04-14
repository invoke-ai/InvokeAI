"""Migration 28: Add per-user workflow isolation columns to workflow_library.

This migration adds the database columns required for multiuser workflow isolation
to the workflow_library table:
- user_id: the owner of the workflow (defaults to 'system' for existing workflows)
- is_public: whether the workflow is shared with all users
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration28Callback:
    """Migration to add user_id and is_public to the workflow_library table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._update_workflow_library_table(cursor)

    def _update_workflow_library_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id and is_public columns to workflow_library table."""
        cursor.execute("PRAGMA table_info(workflow_library);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE workflow_library ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_library_user_id ON workflow_library(user_id);")

        if "is_public" not in columns:
            cursor.execute("ALTER TABLE workflow_library ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_library_is_public ON workflow_library(is_public);")


def build_migration_28() -> Migration:
    """Builds the migration object for migrating from version 27 to version 28.

    This migration adds per-user workflow isolation to the workflow_library table:
    - user_id column: identifies the owner of each workflow
    - is_public column: controls whether a workflow is shared with all users
    """
    return Migration(
        from_version=27,
        to_version=28,
        callback=Migration28Callback(),
    )
