import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration18Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._make_workflow_opened_at_nullable(cursor)

    def _make_workflow_opened_at_nullable(self, cursor: sqlite3.Cursor) -> None:
        """
        - Makes the `opened_at` column on workflow library table nullable by adding a new column
        and deprecating the old one.
        """
        # Rename existing column to deprecated
        cursor.execute("ALTER TABLE workflow_library RENAME COLUMN opened_at TO opened_at_deprecated;")
        # Add new nullable column
        cursor.execute("ALTER TABLE workflow_library ADD COLUMN opened_at DATETIME;")


def build_migration_18() -> Migration:
    """
    Build the migration from database version 17 to 18.

    This migration does the following:
        - Makes the `opened_at` column on workflow library table nullable.
    """
    migration_18 = Migration(
        from_version=17,
        to_version=18,
        callback=Migration18Callback(),
    )

    return migration_18
