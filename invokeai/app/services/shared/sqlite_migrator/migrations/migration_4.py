import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration4Callback:
    """Callback to do step 4 of migration."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:  # noqa D102
        self._create_model_metadata(cursor)
        self._create_model_tags(cursor)
        self._create_tags(cursor)
        self._create_triggers(cursor)

    def _create_model_metadata(self, cursor: sqlite3.Cursor) -> None:
        """Create the table used to store model metadata downloaded from remote sources."""
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_metadata (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.name')) VIRTUAL NOT NULL,
                author TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.author')) VIRTUAL NOT NULL,
                -- Serialized JSON representation of the whole metadata object,
                -- which will contain additional fields from subclasses
                metadata TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                FOREIGN KEY(id) REFERENCES model_config(id) ON DELETE CASCADE
            );
            """
        )

    def _create_model_tags(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_tags (
                model_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                FOREIGN KEY(model_id) REFERENCES model_config(id) ON DELETE CASCADE,
                FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE,
                UNIQUE(model_id,tag_id)
            );
            """
        )

    def _create_tags(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER NOT NULL PRIMARY KEY,
                tag_text TEXT NOT NULL UNIQUE
            );
            """
        )

    def _create_triggers(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS model_metadata_updated_at
            AFTER UPDATE
            ON model_metadata FOR EACH ROW
            BEGIN
                UPDATE model_metadata SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        )


def build_migration_4() -> Migration:
    """
    Build the migration from database version 3 to 4.

    Adds the tables needed to store model metadata and tags.
    """
    migration_4 = Migration(
        from_version=3,
        to_version=4,
        callback=Migration4Callback(),
    )

    return migration_4
