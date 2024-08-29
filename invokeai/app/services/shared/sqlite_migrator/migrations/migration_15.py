import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration15Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_origin_col(cursor)

    def _add_origin_col(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `origin` column to the session queue table.
        - Adds `destination` column to the session queue table.
        """

        cursor.execute("ALTER TABLE session_queue ADD COLUMN origin TEXT;")
        cursor.execute("ALTER TABLE session_queue ADD COLUMN destination TEXT;")


def build_migration_15() -> Migration:
    """
    Build the migration from database version 14 to 15.

    This migration does the following:
        - Adds `origin` column to the session queue table.
        - Adds `destination` column to the session queue table.
    """
    migration_15 = Migration(
        from_version=14,
        to_version=15,
        callback=Migration15Callback(),
    )

    return migration_15
