import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration10Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._update_error_cols(cursor)

    def _update_error_cols(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `error_type` and `error_message` columns to the session queue table.
        - Renames the `error` column to `error_traceback`.
        """

        cursor.execute("ALTER TABLE session_queue ADD COLUMN error_type TEXT;")
        cursor.execute("ALTER TABLE session_queue ADD COLUMN error_message TEXT;")
        cursor.execute("ALTER TABLE session_queue RENAME COLUMN error TO error_traceback;")


def build_migration_10() -> Migration:
    """
    Build the migration from database version 9 to 10.

    This migration does the following:
    - Adds `error_type` and `error_message` columns to the session queue table.
    - Renames the `error` column to `error_traceback`.
    """
    migration_10 = Migration(
        from_version=9,
        to_version=10,
        callback=Migration10Callback(),
    )

    return migration_10
