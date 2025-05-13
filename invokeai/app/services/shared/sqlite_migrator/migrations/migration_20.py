import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration20Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            -- many-to-many relationship table for models
            CREATE TABLE IF NOT EXISTS model_relationships (
                -- model_key_1 and model_key_2 are the same as the key(primary key) in the models table
                model_key_1 TEXT NOT NULL,
                model_key_2 TEXT NOT NULL,
                created_at TEXT DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (model_key_1, model_key_2),
                -- model_key_1 < model_key_2, to ensure uniqueness and prevent duplicates
                FOREIGN KEY (model_key_1) REFERENCES models(id) ON DELETE CASCADE,
                FOREIGN KEY (model_key_2) REFERENCES models(id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """
            -- Creates an index to keep performance equal when searching for model_key_1 or model_key_2
            CREATE INDEX IF NOT EXISTS keyx_model_relationships_model_key_2
            ON model_relationships(model_key_2)
            """
        )


def build_migration_20() -> Migration:
    return Migration(
        from_version=19,
        to_version=20,
        callback=Migration20Callback(),
    )
