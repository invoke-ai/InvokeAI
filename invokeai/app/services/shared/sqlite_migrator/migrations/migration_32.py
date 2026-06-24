"""Migration 32: Repair model_relationships foreign keys.

Migration 22 rebuilt the `models` table by renaming it to `models_old`, creating a
fresh `models` table, copying the data over, and dropping `models_old`. Because
modern SQLite (with `legacy_alter_table` off) rewrites foreign-key references in
*other* tables when a table is renamed, the foreign keys in `model_relationships`
were silently repointed at `models_old` -- which was then dropped.

This left the related-models links referencing a table that no longer exists,
breaking `ON DELETE CASCADE` and foreign-key integrity for related models.

This migration rebuilds `model_relationships` so its foreign keys reference
`models(id)` again, preserving existing links and dropping any orphaned rows whose
model keys no longer exist (those would violate the restored foreign keys).
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration32Callback:
    """Migration to repair the broken foreign keys on the model_relationships table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._repair_model_relationships_fks(cursor)

    def _repair_model_relationships_fks(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_relationships';")
        row = cursor.fetchone()
        if row is None:
            # Table does not exist (fresh db will create it correctly), nothing to repair.
            return

        existing_sql: str = row[0]
        if "models_old" not in existing_sql:
            # Foreign keys already point at the correct table, nothing to repair.
            return

        # Rebuild the table with the correct foreign keys referencing models(id).
        cursor.execute("ALTER TABLE model_relationships RENAME TO model_relationships_old;")
        cursor.execute(
            """
            -- many-to-many relationship table for models
            CREATE TABLE model_relationships (
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

        # Copy over the existing links, dropping any orphaned rows whose model keys no
        # longer exist -- these would violate the restored foreign keys.
        cursor.execute(
            """
            INSERT INTO model_relationships (model_key_1, model_key_2, created_at)
            SELECT model_key_1, model_key_2, created_at
            FROM model_relationships_old
            WHERE model_key_1 IN (SELECT id FROM models)
              AND model_key_2 IN (SELECT id FROM models);
            """
        )

        # Drop the old table first so its index name is freed before we recreate it.
        cursor.execute("DROP TABLE model_relationships_old;")
        cursor.execute(
            """
            -- Creates an index to keep performance equal when searching for model_key_1 or model_key_2
            CREATE INDEX IF NOT EXISTS keyx_model_relationships_model_key_2
            ON model_relationships(model_key_2);
            """
        )


def build_migration_32() -> Migration:
    """Builds the migration object for migrating from version 31 to version 32.

    This migration repairs the foreign keys on the model_relationships table, which were
    broken by migration 22 rebuilding the models table.
    """
    return Migration(
        from_version=31,
        to_version=32,
        callback=Migration32Callback(),
    )
