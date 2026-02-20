"""Migration 28: Add user_id to client_state table for multi-user support.

This migration updates the client_state table to support per-user state isolation:
- Drops the single-row constraint (CHECK(id = 1))
- Adds user_id column
- Creates unique constraint on (user_id, key) pairs
- Migrates existing data to 'system' user
"""

import json
import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration28Callback:
    """Migration to add per-user client state support."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._update_client_state_table(cursor)

    def _update_client_state_table(self, cursor: sqlite3.Cursor) -> None:
        """Restructure client_state table to support per-user storage."""
        # Check if client_state table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='client_state';")
        if cursor.fetchone() is None:
            # Table doesn't exist, create it with the new schema
            cursor.execute(
                """
                CREATE TABLE client_state (
                  user_id     TEXT NOT NULL,
                  key         TEXT NOT NULL,
                  value       TEXT NOT NULL,
                  updated_at  DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                  PRIMARY KEY (user_id, key),
                  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_client_state_user_id ON client_state(user_id);")
            cursor.execute(
                """
                CREATE TRIGGER tg_client_state_updated_at
                AFTER UPDATE ON client_state
                FOR EACH ROW
                BEGIN
                  UPDATE client_state
                    SET updated_at = CURRENT_TIMESTAMP
                  WHERE user_id = OLD.user_id AND key = OLD.key;
                END;
                """
            )
            return

        # Table exists with old schema - migrate it
        # Get existing data if the data column is present (it may be absent if an older
        # version of migration 21 was deployed without the column)
        cursor.execute("PRAGMA table_info(client_state);")
        columns = [row[1] for row in cursor.fetchall()]
        existing_data = {}
        if "data" in columns:
            cursor.execute("SELECT data FROM client_state WHERE id = 1;")
            row = cursor.fetchone()
            if row is not None:
                try:
                    existing_data = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    # If data is corrupt, just start fresh
                    pass

        # Drop the old table
        cursor.execute("DROP TABLE IF EXISTS client_state;")

        # Create new table with per-user schema
        cursor.execute(
            """
            CREATE TABLE client_state (
              user_id     TEXT NOT NULL,
              key         TEXT NOT NULL,
              value       TEXT NOT NULL,
              updated_at  DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
              PRIMARY KEY (user_id, key),
              FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_client_state_user_id ON client_state(user_id);")

        cursor.execute(
            """
            CREATE TRIGGER tg_client_state_updated_at
            AFTER UPDATE ON client_state
            FOR EACH ROW
            BEGIN
              UPDATE client_state
                SET updated_at = CURRENT_TIMESTAMP
              WHERE user_id = OLD.user_id AND key = OLD.key;
            END;
            """
        )

        # Migrate existing data to 'system' user
        for key, value in existing_data.items():
            cursor.execute(
                """
                INSERT INTO client_state (user_id, key, value)
                VALUES ('system', ?, ?);
                """,
                (key, value),
            )


def build_migration_28() -> Migration:
    """Builds the migration object for migrating from version 27 to version 28.

    This migration adds per-user client state support to prevent data leakage between users.
    """
    return Migration(
        from_version=27,
        to_version=28,
        callback=Migration28Callback(),
    )
