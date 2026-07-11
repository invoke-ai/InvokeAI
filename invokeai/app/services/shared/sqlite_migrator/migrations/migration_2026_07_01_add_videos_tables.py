"""Migration: Add `videos` and `board_videos` tables for minimal video support.

The `videos` table parallels `images` but with extra `duration` and `fps` columns.
The `board_videos` table parallels `board_images`, providing one-to-many board↔video association.
Foreign-key cascades from `boards` mirror the image side, so deleting a board also removes its videos' associations.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddVideosTablesCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_videos(cursor)
        self._create_board_videos(cursor)

    def _create_videos(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS videos (
                video_name TEXT NOT NULL PRIMARY KEY,
                video_origin TEXT NOT NULL,
                video_category TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                duration REAL NOT NULL DEFAULT 0.0,
                fps REAL,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                is_intermediate BOOLEAN DEFAULT FALSE,
                starred BOOLEAN DEFAULT FALSE,
                has_workflow BOOLEAN DEFAULT FALSE,
                -- Deliberately no FK to users(user_id), matching images/boards/workflows
                -- (migration_27 adds those user_id columns with an index only): deleting a
                -- user leaves their media in place for admin review/cleanup rather than
                -- cascading a row delete that would strand the files on disk.
                user_id TEXT NOT NULL DEFAULT 'system',
                video_subfolder TEXT NOT NULL DEFAULT '',
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_video_name ON videos(video_name);",
            "CREATE INDEX IF NOT EXISTS idx_videos_video_origin ON videos(video_origin);",
            "CREATE INDEX IF NOT EXISTS idx_videos_video_category ON videos(video_category);",
            "CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_videos_starred ON videos(starred);",
            "CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_videos_updated_at
            AFTER UPDATE
            ON videos FOR EACH ROW
            BEGIN
                UPDATE videos SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE video_name = old.video_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_board_videos(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS board_videos (
                board_id TEXT NOT NULL,
                video_name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many board↔video using PK on video_name
                PRIMARY KEY (video_name),
                FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
                FOREIGN KEY (video_name) REFERENCES videos (video_name) ON DELETE CASCADE
            );
            """
        ]

        indices = [
            "CREATE INDEX IF NOT EXISTS idx_board_videos_board_id ON board_videos (board_id);",
            "CREATE INDEX IF NOT EXISTS idx_board_videos_board_id_created_at ON board_videos (board_id, created_at);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_board_videos_updated_at
            AFTER UPDATE
            ON board_videos FOR EACH ROW
            BEGIN
                UPDATE board_videos SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE board_id = old.board_id AND video_name = old.video_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)


def build_migration() -> Migration:
    """Builds the migration that adds the videos and board_videos tables.

    Depends on migration_27, which last reshaped the boards table and created the 'system' user
    that videos.user_id defaults to.
    """
    return Migration(
        id="2026_07_01_add_videos_tables",
        depends_on="migration_27",
        callback=AddVideosTablesCallback(),
    )
