import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration6Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._recreate_model_triggers(cursor)

    def _recreate_model_triggers(self, cursor: sqlite3.Cursor) -> None:
        """
        Adds the timestamp trigger to the model_config table.

        This trigger was inadvertently dropped in earlier migration scripts.
        """

        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS model_config_updated_at
            AFTER UPDATE
            ON model_config FOR EACH ROW
            BEGIN
                UPDATE model_config SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        )


def build_migration_6() -> Migration:
    """
    Build the migration from database version 5 to 6.

    This migration does the following:
    - Adds the model_config_updated_at trigger if it does not exist
    """
    migration_6 = Migration(
        from_version=5,
        to_version=6,
        callback=Migration6Callback(),
    )

    return migration_6
