import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration7Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_models_table(cursor)

    def _create_models_table(self, cursor: sqlite3.Cursor) -> None:
        """
        Adds the timestamp trigger to the model_config table.

        This trigger was inadvertently dropped in earlier migration scripts.
        """

        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS models (
                id TEXT NOT NULL PRIMARY KEY,
                base TEXT GENERATED ALWAYS as (json_extract(config, '$.base')) VIRTUAL NOT NULL,
                type TEXT GENERATED ALWAYS as (json_extract(config, '$.type')) VIRTUAL NOT NULL,
                path TEXT GENERATED ALWAYS as (json_extract(config, '$.path')) VIRTUAL NOT NULL,
                format TEXT GENERATED ALWAYS as (json_extract(config, '$.format')) VIRTUAL NOT NULL,
                name TEXT GENERATED ALWAYS as (json_extract(config, '$.name')) VIRTUAL NOT NULL,
                description TEXT GENERATED ALWAYS as (json_extract(config, '$.description')) VIRTUAL,
                source TEXT GENERATED ALWAYS as (json_extract(config, '$.source')) VIRTUAL NOT NULL,
                source_type TEXT GENERATED ALWAYS as (json_extract(config, '$.source_type')) VIRTUAL NOT NULL,
                source_api_response TEXT GENERATED ALWAYS as (json_extract(config, '$.source_api_response')) VIRTUAL,
                hash TEXT NOT NULL, -- could be null
                -- Serialized JSON representation of the whole config object, which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- unique constraint on combo of name, base and type
                UNIQUE(name, base, type)
            );
            """
        ]

        # Add trigger for `updated_at`.
        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS models_updated_at
            AFTER UPDATE
            ON models FOR EACH ROW
            BEGIN
                UPDATE models SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        ]

        # Add indexes for searchable fields
        indices = [
            "CREATE INDEX IF NOT EXISTS base_index ON models(base);",
            "CREATE INDEX IF NOT EXISTS type_index ON models(type);",
            "CREATE INDEX IF NOT EXISTS name_index ON models(name);",
            "CREATE UNIQUE INDEX IF NOT EXISTS path_index ON models(path);",
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)


def build_migration_7() -> Migration:
    """
    Build the migration from database version 5 to 6.

    This migration does the following:
    - Adds the model_config_updated_at trigger if it does not exist
    - Delete all ip_adapter models so that the model prober can find and
      update with the correct image processor model.
    """
    migration_7 = Migration(
        from_version=6,
        to_version=7,
        callback=Migration7Callback(),
    )

    return migration_7
