import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration9Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._empty_session_queue(cursor)

    def _empty_session_queue(self, cursor: sqlite3.Cursor) -> None:
        """Empties the session queue. This is done to prevent any lingering session queue items from causing pydantic errors due to changed schemas."""

        cursor.execute("DELETE FROM session_queue;")


def build_migration_9() -> Migration:
    """
    Build the migration from database version 8 to 9.

    This migration does the following:
    - Empties the session queue. This is done to prevent any lingering session queue items from causing pydantic errors due to changed schemas.
    """
    migration_9 = Migration(
        from_version=8,
        to_version=9,
        callback=Migration9Callback(),
    )

    return migration_9
