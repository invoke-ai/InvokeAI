"""The cleanup cache must notice session rewrites even when their timestamps collide."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_23_track_queue_session_revision import (
    build_migration,
)


def test_session_revision_advances_only_for_changed_session_content() -> None:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE session_queue (item_id INTEGER PRIMARY KEY, session TEXT NOT NULL, updated_at TEXT);")
    cursor.execute("INSERT INTO session_queue VALUES (1, '{\"image_name\":\"old.png\"}', 'same-timestamp');")
    build_migration().callback(cursor)

    cursor.execute('UPDATE session_queue SET session = \'{"image_name":"new.png"}\' WHERE item_id = 1;')
    cursor.execute("SELECT session_revision, updated_at FROM session_queue WHERE item_id = 1;")
    assert cursor.fetchone() == (1, "same-timestamp")

    cursor.execute("UPDATE session_queue SET session = session WHERE item_id = 1;")
    cursor.execute("SELECT session_revision FROM session_queue WHERE item_id = 1;")
    assert cursor.fetchone()[0] == 1
    build_migration().callback(cursor)
    cursor.execute('UPDATE session_queue SET session = \'{"image_name":"old.png"}\' WHERE item_id = 1;')
    cursor.execute("SELECT session_revision FROM session_queue WHERE item_id = 1;")
    assert cursor.fetchone()[0] == 2
