import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration13Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_archived_col(cursor)

    def _add_archived_col(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `archived` columns to the board table.
        """

        cursor.execute("ALTER TABLE boards ADD COLUMN archived BOOLEAN DEFAULT FALSE;")


def build_migration_13() -> Migration:
    """
    Build the migration from database version 12 to 13..

    This migration does the following:
    - Adds `archived` columns to the board table.
    """
    migration_13 = Migration(
        from_version=12,
        to_version=13,
        callback=Migration13Callback(),
    )

    return migration_13
