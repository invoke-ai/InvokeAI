import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration21Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE client_state (
              id          INTEGER PRIMARY KEY CHECK(id = 1),
              data        TEXT    NOT NULL, -- Frontend will handle the shape of this data
              updated_at  DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
            );
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER tg_client_state_updated_at
            AFTER UPDATE ON client_state
            FOR EACH ROW
            BEGIN
              UPDATE client_state
                SET updated_at = CURRENT_TIMESTAMP
              WHERE id = OLD.id;
            END;
            """
        )


def build_migration_21() -> Migration:
    """Builds the migration object for migrating from version 20 to version 21. This includes:
    - Creating the `client_state` table.
    - Adding a trigger to update the `updated_at` field on updates.
    """
    return Migration(
        from_version=20,
        to_version=21,
        callback=Migration21Callback(),
    )
