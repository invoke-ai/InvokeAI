import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration16Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_retried_from_item_id_col(cursor)

    def _add_retried_from_item_id_col(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `retried_from_item_id` column to the session queue table.
        """

        cursor.execute("ALTER TABLE session_queue ADD COLUMN retried_from_item_id INTEGER;")


def build_migration_16() -> Migration:
    """
    Build the migration from database version 15 to 16.

    This migration does the following:
        - Adds `retried_from_item_id` column to the session queue table.
    """
    migration_16 = Migration(
        from_version=15,
        to_version=16,
        callback=Migration16Callback(),
    )

    return migration_16
