import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration14Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_origin_col(cursor)

    def _add_origin_col(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `origin` column to the session queue table.
        """

        cursor.execute("ALTER TABLE session_queue ADD COLUMN origin TEXT;")


def build_migration_14() -> Migration:
    """
    Build the migration from database version 13 to 14.

    This migration does the following:
        - Adds `origin` column to the session queue table.
    """
    migration_14 = Migration(
        from_version=13,
        to_version=14,
        callback=Migration14Callback(),
    )

    return migration_14
