import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration18Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._make_workflow_opened_at_nullable(cursor)

    def _make_workflow_opened_at_nullable(self, cursor: sqlite3.Cursor) -> None:
        """
        Make the `opened_at` column nullable in the `workflow_library` table. This is accomplished by:
            - Dropping the existing `idx_workflow_library_opened_at` index (must be done before dropping the column)
            - Dropping the existing `opened_at` column
            - Adding a new nullable column `opened_at` (no data migration needed, all values will be NULL)
            - Adding a new `idx_workflow_library_opened_at` index on the `opened_at` column
        """
        # For index renaming in SQLite, we need to drop and recreate
        cursor.execute("DROP INDEX IF EXISTS idx_workflow_library_opened_at;")
        # Rename existing column to deprecated
        cursor.execute("ALTER TABLE workflow_library DROP COLUMN opened_at;")
        # Add new nullable column - all values will be NULL - no migration of data needed
        cursor.execute("ALTER TABLE workflow_library ADD COLUMN opened_at DATETIME;")
        # Create new index on the new column
        cursor.execute(
            "CREATE INDEX idx_workflow_library_opened_at ON workflow_library(opened_at);",
        )


def build_migration_18() -> Migration:
    """
    Build the migration from database version 17 to 18.

    This migration does the following:
        - Make the `opened_at` column nullable in the `workflow_library` table. This is accomplished by:
            - Dropping the existing `idx_workflow_library_opened_at` index (must be done before dropping the column)
            - Dropping the existing `opened_at` column
            - Adding a new nullable column `opened_at` (no data migration needed, all values will be NULL)
            - Adding a new `idx_workflow_library_opened_at` index on the `opened_at` column
    """
    migration_18 = Migration(
        from_version=17,
        to_version=18,
        callback=Migration18Callback(),
    )

    return migration_18
