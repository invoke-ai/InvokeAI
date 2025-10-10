import sqlite3
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration22Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger
        self._models_dir = app_config.models_path.resolve()

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._logger.info("Removing UNIQUE(name, base, type) constraint from models table")

        # Step 1: Rename the existing models table
        cursor.execute("ALTER TABLE models RENAME TO models_old;")

        # Step 2: Create the new models table without the UNIQUE(name, base, type) constraint
        cursor.execute(
            """--sql
            CREATE TABLE models (
                id TEXT NOT NULL PRIMARY KEY,
                hash TEXT GENERATED ALWAYS as (json_extract(config, '$.hash')) VIRTUAL NOT NULL,
                base TEXT GENERATED ALWAYS as (json_extract(config, '$.base')) VIRTUAL NOT NULL,
                type TEXT GENERATED ALWAYS as (json_extract(config, '$.type')) VIRTUAL NOT NULL,
                path TEXT GENERATED ALWAYS as (json_extract(config, '$.path')) VIRTUAL NOT NULL,
                format TEXT GENERATED ALWAYS as (json_extract(config, '$.format')) VIRTUAL NOT NULL,
                name TEXT GENERATED ALWAYS as (json_extract(config, '$.name')) VIRTUAL NOT NULL,
                description TEXT GENERATED ALWAYS as (json_extract(config, '$.description')) VIRTUAL,
                source TEXT GENERATED ALWAYS as (json_extract(config, '$.source')) VIRTUAL NOT NULL,
                source_type TEXT GENERATED ALWAYS as (json_extract(config, '$.source_type')) VIRTUAL NOT NULL,
                source_api_response TEXT GENERATED ALWAYS as (json_extract(config, '$.source_api_response')) VIRTUAL,
                trigger_phrases TEXT GENERATED ALWAYS as (json_extract(config, '$.trigger_phrases')) VIRTUAL,
                file_size INTEGER GENERATED ALWAYS as (json_extract(config, '$.file_size')) VIRTUAL NOT NULL,
                -- Serialized JSON representation of the whole config object, which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Explicit unique constraint on path
                UNIQUE(path)
            );
            """
        )

        # Step 3: Copy all data from the old table to the new table
        # Only copy the stored columns (id, config, created_at, updated_at), not the virtual columns
        cursor.execute(
            "INSERT INTO models (id, config, created_at, updated_at) "
            "SELECT id, config, created_at, updated_at FROM models_old;"
        )

        # Step 4: Drop the old table
        cursor.execute("DROP TABLE models_old;")

        # Step 5: Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS base_index ON models(base);")
        cursor.execute("CREATE INDEX IF NOT EXISTS type_index ON models(type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS name_index ON models(name);")

        # Step 6: Recreate the updated_at trigger
        cursor.execute(
            """--sql
            CREATE TRIGGER models_updated_at
            AFTER UPDATE
            ON models FOR EACH ROW
            BEGIN
                UPDATE models SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        )


def build_migration_22(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """Builds the migration object for migrating from version 21 to version 22.

    This migration:
    - Removes the UNIQUE constraint on the combination of (base, name, type) columns in the models table
    - Adds an explicit UNIQUE contraint on the path column
    """

    return Migration(
        from_version=21,
        to_version=22,
        callback=Migration22Callback(app_config=app_config, logger=logger),
    )
