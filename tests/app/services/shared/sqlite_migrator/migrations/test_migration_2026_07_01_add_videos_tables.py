"""Tests for migration 2026_07_01_add_videos_tables: add videos and board_videos tables."""

import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_01_add_videos_tables import (
    AddVideosTablesCallback,
    build_migration,
)


def _create_boards_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE boards (
            board_id TEXT NOT NULL PRIMARY KEY,
            board_name TEXT NOT NULL
        );
        """
    )


def _get_table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    _create_boards_table(conn)
    return conn


class TestAddVideosTables:
    def test_creates_videos_and_board_videos_tables(self, db: sqlite3.Connection):
        AddVideosTablesCallback()(db.cursor())
        db.commit()

        assert {"videos", "board_videos"} <= _get_table_names(db)

    def test_videos_updated_at_trigger(self, db: sqlite3.Connection):
        AddVideosTablesCallback()(db.cursor())
        db.commit()

        db.execute(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height)"
            " VALUES ('v1', 'internal', 'general', 640, 480)"
        )
        before = db.execute("SELECT updated_at FROM videos WHERE video_name='v1'").fetchone()[0]
        db.execute("UPDATE videos SET starred = TRUE WHERE video_name='v1'")
        after = db.execute("SELECT updated_at FROM videos WHERE video_name='v1'").fetchone()[0]
        assert after >= before

    def test_board_delete_cascades_to_board_videos(self, db: sqlite3.Connection):
        AddVideosTablesCallback()(db.cursor())
        db.commit()

        db.execute("INSERT INTO boards (board_id, board_name) VALUES ('b1', 'Board 1')")
        db.execute(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height)"
            " VALUES ('v1', 'internal', 'general', 640, 480)"
        )
        db.execute("INSERT INTO board_videos (board_id, video_name) VALUES ('b1', 'v1')")
        db.execute("DELETE FROM boards WHERE board_id='b1'")

        assert db.execute("SELECT COUNT(*) FROM board_videos").fetchone()[0] == 0
        # The video itself is not deleted, only its board association.
        assert db.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1

    def test_idempotent_when_tables_exist(self, db: sqlite3.Connection):
        cursor = db.cursor()
        AddVideosTablesCallback()(cursor)
        db.execute(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height)"
            " VALUES ('v1', 'internal', 'general', 640, 480)"
        )
        AddVideosTablesCallback()(cursor)
        db.commit()

        # Re-running must not drop or recreate existing tables/data.
        assert db.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1

    def test_builder_metadata(self):
        migration = build_migration()
        assert migration.id == "2026_07_01_add_videos_tables"
        assert migration.depends_on == "migration_27"
        assert migration.from_version is None
        assert migration.to_version is None
