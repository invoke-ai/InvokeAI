import sqlite3
from logging import Logger
from typing import cast

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration5Callback:
    def __init__(self, logger: Logger):
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._drop_graph_executions(cursor)

    def _drop_graph_executions(self, cursor: sqlite3.Cursor) -> None:
        """Drops the `graph_executions` table."""

        # Check if the table exists first. This shouldn't ever be an issue for users, but it is for tests.
        #
        # Unlike other database tables, whose schemas are managed by the migrator, `graph_executions`
        # is created and managed by an instance of `SqliteItemStorage``. Specifically, it is created in the
        # `__init__` of `SqliteItemStorage`.
        #
        # In tests, we don't always create an instance of `SqliteItemStorage`, so the table may not exist in
        # the test memory db fixture.
        #
        # We _do_, however, run the migrator on the test memory db fixtures. So, we need to check if the
        # table exists before dropping it, else unrelated tests will fail.
        #
        # This song and dance won't be necessary again, because we aren't using `SqliteItemStorage` for anything
        # moving forward.

        cursor.execute(
            """--sql
            SELECT name FROM sqlite_master WHERE type='table' AND name='graph_executions';
            """
        )

        if cursor.fetchone():
            cursor.execute(
                """--sql
                SELECT COUNT(*) FROM graph_executions;
                """
            )
            count = cast(int, cursor.fetchone()[0])
            self._logger.info(f"Clearing {count} old sessions from database")
            cursor.execute(
                """--sql
                DROP TABLE IF EXISTS graph_executions;
                """
            )
        else:
            self._logger.info("No 'graph_executions' table found.")


def build_migration_5(logger: Logger) -> Migration:
    """
    Build the migration from database version 4 to 5.

    Introduced in v3.6.3, this migration:
    - Drops the `graph_executions` table. We are able to do this because we are moving the graph storage
      to be purely in-memory.
    """
    migration_5 = Migration(
        from_version=4,
        to_version=5,
        callback=Migration5Callback(logger=logger),
    )

    return migration_5
