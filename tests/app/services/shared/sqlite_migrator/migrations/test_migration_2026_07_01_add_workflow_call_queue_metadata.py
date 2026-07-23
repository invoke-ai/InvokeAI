import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_01_add_workflow_call_queue_metadata import (
    AddWorkflowCallQueueMetadataCallback,
    build_migration,
)


def _get_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cursor.fetchall()}


def _get_indexes(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index';")
    return {row[0] for row in cursor.fetchall()}


def test_adds_workflow_call_columns_and_indexes_to_session_queue() -> None:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE session_queue (item_id INTEGER PRIMARY KEY);")

    AddWorkflowCallQueueMetadataCallback()(cursor)

    assert _get_columns(cursor, "session_queue") >= {
        "workflow_call_id",
        "parent_item_id",
        "parent_session_id",
        "root_item_id",
        "workflow_call_depth",
    }
    assert _get_indexes(cursor) >= {
        "idx_session_queue_workflow_call_id",
        "idx_session_queue_parent_item_id",
        "idx_session_queue_parent_session_id",
        "idx_session_queue_root_item_id",
        "idx_session_queue_workflow_call_depth",
    }

    db.close()


def test_migration_is_idempotent_and_tolerates_missing_session_queue() -> None:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()

    AddWorkflowCallQueueMetadataCallback()(cursor)
    cursor.execute("CREATE TABLE session_queue (item_id INTEGER PRIMARY KEY, workflow_call_id TEXT);")
    AddWorkflowCallQueueMetadataCallback()(cursor)
    AddWorkflowCallQueueMetadataCallback()(cursor)

    assert _get_columns(cursor, "session_queue") >= {
        "workflow_call_id",
        "parent_item_id",
        "parent_session_id",
        "root_item_id",
        "workflow_call_depth",
    }

    db.close()


def test_build_migration_declares_stable_id_and_dependency() -> None:
    migration = build_migration()

    assert migration.id == "2026_07_01_add_workflow_call_queue_metadata"
    assert migration.depends_on == "migration_33"
    assert migration.from_version is None
    assert migration.to_version is None
