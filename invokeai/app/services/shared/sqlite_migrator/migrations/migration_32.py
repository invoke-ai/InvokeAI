"""Migration 32: Rebuild the model_relationships table with valid foreign keys.

Migration 22 recreated the models table via `ALTER TABLE models RENAME TO
models_old` + `DROP TABLE models_old`. SQLite rewrites foreign key clauses in
referencing tables on RENAME, so model_relationships was left with foreign
keys pointing at the dropped models_old table. Reads still worked (foreign
keys are only enforced on writes), but every INSERT failed with
"no such table: main.models_old".

This migration rebuilds model_relationships with foreign keys referencing
models(id), preserving any relationships whose models still exist.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration32Callback:
    """Migration to repair model_relationships foreign keys broken by migration 22."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._rebuild_model_relationships(cursor)

    def _rebuild_model_relationships(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_relationships';")
        if cursor.fetchone() is None:
            return

        cursor.execute("ALTER TABLE model_relationships RENAME TO model_relationships_old;")
        cursor.execute(
            """--sql
            CREATE TABLE model_relationships (
                -- model_key_1 and model_key_2 are the same as the id (primary key) in the models table
                model_key_1 TEXT NOT NULL,
                model_key_2 TEXT NOT NULL,
                created_at TEXT DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- model_key_1 < model_key_2, to ensure uniqueness and prevent duplicates
                PRIMARY KEY (model_key_1, model_key_2),
                FOREIGN KEY (model_key_1) REFERENCES models(id) ON DELETE CASCADE,
                FOREIGN KEY (model_key_2) REFERENCES models(id) ON DELETE CASCADE
            );
            """
        )
        # Carry over relationships whose models still exist; anything else is
        # unrecoverable garbage left behind by deletes that the broken foreign
        # keys could not cascade.
        cursor.execute(
            """--sql
            INSERT OR IGNORE INTO model_relationships (model_key_1, model_key_2, created_at)
            SELECT model_key_1, model_key_2, created_at FROM model_relationships_old
            WHERE model_key_1 IN (SELECT id FROM models)
              AND model_key_2 IN (SELECT id FROM models);
            """
        )
        cursor.execute("DROP TABLE model_relationships_old;")
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS keyx_model_relationships_model_key_2
            ON model_relationships(model_key_2);
            """
        )


def build_migration_32() -> Migration:
    return Migration(
        from_version=31,
        to_version=32,
        callback=Migration32Callback(),
    )
