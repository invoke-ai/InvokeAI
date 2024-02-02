import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration5Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._drop_graph_executions(cursor)

    def _drop_graph_executions(self, cursor: sqlite3.Cursor) -> None:
        """Drops the `graph_executions` table."""

        cursor.execute(
            """--sql
            DROP TABLE IF EXISTS graph_executions;
            """
        )


def build_migration_5() -> Migration:
    """
    Build the migration from database version 4 to 5.

    Introduced in v3.6.3, this migration:
    - Drops the `graph_executions` table. We are able to do this because we are moving the graph storage
      to be purely in-memory.
    """
    migration_5 = Migration(
        from_version=4,
        to_version=5,
        callback=Migration5Callback(),
    )

    return migration_5
