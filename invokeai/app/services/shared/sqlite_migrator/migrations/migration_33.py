"""Migration 33: Add the projects table for server-side workbench project persistence.

Projects are the v7 workbench's primary unit of work (spec: State Ownership).
Each row stores one user-owned project document as opaque JSON, plus a
monotonic revision used for optimistic concurrency: clients send the revision
they loaded, and a save against a stale revision is rejected so concurrent
tabs/devices cannot silently overwrite each other.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration33Callback:
    """Migration to add the projects table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_projects_table(cursor)

    def _create_projects_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE projects (
                -- Client-generated identifier; unique per user, not globally.
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                -- Opaque client-owned project document (JSON).
                data TEXT NOT NULL,
                -- Incremented on every update; used for optimistic concurrency.
                revision INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, project_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE TRIGGER tg_projects_updated_at
            AFTER UPDATE ON projects
            FOR EACH ROW
            BEGIN
              UPDATE projects
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
              WHERE user_id = OLD.user_id AND project_id = OLD.project_id;
            END;
            """
        )


def build_migration_33() -> Migration:
    """Builds the migration object for migrating from version 32 to version 33. This includes:
    - Creating the `projects` table for per-user workbench project persistence.
    - Adding a trigger to keep `updated_at` current.
    """
    return Migration(
        from_version=32,
        to_version=33,
        callback=Migration33Callback(),
    )
