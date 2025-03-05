import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration17Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_workflows_tags_col(cursor)

    def _add_workflows_tags_col(self, cursor: sqlite3.Cursor) -> None:
        """
        - Adds `tags` column to the workflow_library table. It is a generated column that extracts the tags from the
            workflow JSON.
        """

        cursor.execute(
            "ALTER TABLE workflow_library ADD COLUMN tags TEXT GENERATED ALWAYS AS (json_extract(workflow, '$.tags')) VIRTUAL;"
        )


def build_migration_17() -> Migration:
    """
    Build the migration from database version 16 to 17.

    This migration does the following:
        - Adds `tags` column to the workflow_library table. It is a generated column that extracts the tags from the
            workflow JSON.
    """
    migration_17 = Migration(
        from_version=16,
        to_version=17,
        callback=Migration17Callback(),
    )

    return migration_17
