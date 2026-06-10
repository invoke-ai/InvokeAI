"""Migration 33: Add `canvas_projects` and `board_canvas_projects` tables for Canvas Project (.invk) support.

The `canvas_projects` table stores metadata for server-persisted Canvas Project ZIP files (`.invk`).
The `board_canvas_projects` table parallels `board_images` / `board_videos`, providing one-to-many
board↔project association. Foreign-key cascades from `boards` mirror the image/video sides.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration33Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_canvas_projects(cursor)
        self._create_board_canvas_projects(cursor)

    def _create_canvas_projects(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS canvas_projects (
                project_name TEXT NOT NULL PRIMARY KEY,
                project_origin TEXT NOT NULL,
                name TEXT NOT NULL,
                app_version TEXT NOT NULL DEFAULT 'unknown',
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                image_count INTEGER NOT NULL DEFAULT 0,
                has_thumbnail BOOLEAN DEFAULT FALSE,
                starred BOOLEAN DEFAULT FALSE,
                is_intermediate BOOLEAN DEFAULT FALSE,
                user_id TEXT NOT NULL DEFAULT 'system',
                project_subfolder TEXT NOT NULL DEFAULT '',
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_canvas_projects_project_name ON canvas_projects(project_name);",
            "CREATE INDEX IF NOT EXISTS idx_canvas_projects_project_origin ON canvas_projects(project_origin);",
            "CREATE INDEX IF NOT EXISTS idx_canvas_projects_created_at ON canvas_projects(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_canvas_projects_starred ON canvas_projects(starred);",
            "CREATE INDEX IF NOT EXISTS idx_canvas_projects_user_id ON canvas_projects(user_id);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_canvas_projects_updated_at
            AFTER UPDATE
            ON canvas_projects FOR EACH ROW
            BEGIN
                UPDATE canvas_projects SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE project_name = old.project_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_board_canvas_projects(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS board_canvas_projects (
                board_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many board↔project using PK on project_name
                PRIMARY KEY (project_name),
                FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
                FOREIGN KEY (project_name) REFERENCES canvas_projects (project_name) ON DELETE CASCADE
            );
            """
        ]

        indices = [
            "CREATE INDEX IF NOT EXISTS idx_board_canvas_projects_board_id ON board_canvas_projects (board_id);",
            "CREATE INDEX IF NOT EXISTS idx_board_canvas_projects_board_id_created_at ON board_canvas_projects (board_id, created_at);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_board_canvas_projects_updated_at
            AFTER UPDATE
            ON board_canvas_projects FOR EACH ROW
            BEGIN
                UPDATE board_canvas_projects SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE board_id = old.board_id AND project_name = old.project_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)


def build_migration_33() -> Migration:
    return Migration(
        from_version=32,
        to_version=33,
        callback=Migration33Callback(),
    )
