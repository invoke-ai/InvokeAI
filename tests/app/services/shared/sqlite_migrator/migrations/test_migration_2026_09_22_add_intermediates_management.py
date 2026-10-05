"""The intermediates-management schema: provenance/size columns and the media reference index."""

import json
import sqlite3
from logging import getLogger

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_22_add_intermediates_management import (
    build_migration,
)


@pytest.fixture
def cursor() -> sqlite3.Cursor:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")
    connection.execute(
        "CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY, user_id TEXT DEFAULT 'system',"
        " is_intermediate BOOLEAN DEFAULT FALSE);"
    )
    connection.execute(
        "CREATE TABLE videos (video_name TEXT NOT NULL PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'system',"
        " is_intermediate BOOLEAN DEFAULT FALSE);"
    )
    connection.execute(
        "CREATE TABLE session_queue (item_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL);"
    )
    connection.execute(
        "CREATE TABLE projects (project_id TEXT NOT NULL, user_id TEXT NOT NULL, data TEXT NOT NULL,"
        " PRIMARY KEY (user_id, project_id));"
    )
    connection.execute(
        "CREATE TABLE workflow_library (workflow_id TEXT NOT NULL PRIMARY KEY, workflow TEXT NOT NULL,"
        " user_id TEXT DEFAULT 'system',"
        " category TEXT GENERATED ALWAYS AS (json_extract(workflow, '$.meta.category')) VIRTUAL NOT NULL);"
    )
    return connection.cursor()


def _run(cursor: sqlite3.Cursor) -> None:
    build_migration(getLogger(__name__)).callback(cursor)  # type: ignore[misc]


def _columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cursor.fetchall()}


def _references(cursor: sqlite3.Cursor) -> set[tuple[str, str, str, str, str]]:
    cursor.execute("SELECT owner_kind, user_id, owner_id, media_kind, media_name FROM media_references;")
    return set(cursor.fetchall())


def test_adds_provenance_and_size_columns(cursor: sqlite3.Cursor) -> None:
    _run(cursor)

    assert {"project_id", "file_size_bytes"} <= _columns(cursor, "images")
    assert {"project_id", "file_size_bytes"} <= _columns(cursor, "videos")
    assert "project_id" in _columns(cursor, "session_queue")
    assert {"operation_id", "caller_user_id", "idempotency_key", "state_json"} <= _columns(
        cursor, "intermediates_operations"
    )
    assert {"user_id", "lease_id", "media_kind", "media_name", "expires_at"} <= _columns(
        cursor, "intermediates_browser_holds"
    )
    assert {"operation_id", "media_kind", "media_name", "size_bytes", "confirmed_refs_json"} <= _columns(
        cursor, "intermediates_operation_targets"
    )

    # Existing media stays unassigned and unmeasured rather than guessed.
    cursor.execute("INSERT INTO images (image_name) VALUES ('old.png');")
    cursor.execute("SELECT project_id, file_size_bytes FROM images WHERE image_name = 'old.png';")
    assert cursor.fetchone() == (None, None)


def test_backfills_references_from_projects_and_workflows(cursor: sqlite3.Cursor) -> None:
    project = {
        "canvas": {
            "stagingArea": {"pendingImages": [{"imageName": "staged.png"}, {"imageName": "staged.png"}]},
            "layers": [{"image": {"image_name": "layer.png"}}],
        },
        "video": {"clips": [{"video_name": "clip.mp4"}]},
        "unrelated": {"imageName": "", "name": "not-a-ref.png"},
    }
    workflow = {
        "meta": {"category": "user"},
        "nodes": [{"data": {"inputs": {"image": {"value": {"image_name": "input.png"}}}}}],
    }
    default_workflow = {"meta": {"category": "default"}, "nodes": [{"data": {"image_name": "bundled.png"}}]}
    cursor.execute("INSERT INTO projects VALUES ('p1', 'u1', ?);", (json.dumps(project),))
    cursor.execute("INSERT INTO projects VALUES ('p2', 'u2', 'not json');")
    cursor.execute("INSERT INTO workflow_library VALUES ('w1', ?, 'u1');", (json.dumps(workflow),))
    # Default workflows are synced from bundled files; runtime never indexes them, so neither does the backfill.
    cursor.execute("INSERT INTO workflow_library VALUES ('d1', ?, 'system');", (json.dumps(default_workflow),))

    _run(cursor)

    assert _references(cursor) == {
        ("project", "u1", "p1", "image", "staged.png"),
        ("project", "u1", "p1", "image", "layer.png"),
        ("project", "u1", "p1", "video", "clip.mp4"),
        ("workflow", "u1", "w1", "image", "input.png"),
    }


def test_rerunning_does_not_disturb_a_maintained_index(cursor: sqlite3.Cursor) -> None:
    cursor.execute("INSERT INTO projects VALUES ('p1', 'u1', ?);", (json.dumps({"imageName": "a.png"}),))
    _run(cursor)
    # The writer has since replaced the project's references; a second run must not resurrect the old
    # ones from the document nor fail on the existing columns.
    cursor.execute("DELETE FROM media_references;")
    cursor.execute("INSERT INTO media_references VALUES ('project', 'u1', 'p1', 'image', 'b.png');")

    _run(cursor)

    assert _references(cursor) == {("project", "u1", "p1", "image", "b.png")}
    assert {"project_id", "file_size_bytes"} <= _columns(cursor, "images")
