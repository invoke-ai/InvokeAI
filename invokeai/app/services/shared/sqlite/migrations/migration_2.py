import sqlite3

from invokeai.app.services.shared.sqlite.sqlite_migrator import Migration


def _migrate(cursor: sqlite3.Cursor) -> None:
    """Migration callback for database version 2."""

    _add_images_has_workflow(cursor)
    _add_session_queue_workflow(cursor)
    _drop_old_workflow_tables(cursor)
    _add_workflow_library(cursor)
    _drop_model_manager_metadata(cursor)


def _add_images_has_workflow(cursor: sqlite3.Cursor) -> None:
    """Add the `has_workflow` column to `images` table."""
    cursor.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]

    if "has_workflow" not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN has_workflow BOOLEAN DEFAULT FALSE;")


def _add_session_queue_workflow(cursor: sqlite3.Cursor) -> None:
    """Add the `workflow` column to `session_queue` table."""

    cursor.execute("PRAGMA table_info(session_queue)")
    columns = [column[1] for column in cursor.fetchall()]

    if "workflow" not in columns:
        cursor.execute("ALTER TABLE session_queue ADD COLUMN workflow TEXT;")


def _drop_old_workflow_tables(cursor: sqlite3.Cursor) -> None:
    """Drops the `workflows` and `workflow_images` tables."""
    cursor.execute("DROP TABLE IF EXISTS workflow_images;")
    cursor.execute("DROP TABLE IF EXISTS workflows;")


def _add_workflow_library(cursor: sqlite3.Cursor) -> None:
    """Adds the `workflow_library` table and drops the `workflows` and `workflow_images` tables."""
    tables = [
        """--sql
        CREATE TABLE IF NOT EXISTS workflow_library (
            workflow_id TEXT NOT NULL PRIMARY KEY,
            workflow TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            -- updated via trigger
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            -- updated manually when retrieving workflow
            opened_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            -- Generated columns, needed for indexing and searching
            category TEXT GENERATED ALWAYS as (json_extract(workflow, '$.meta.category')) VIRTUAL NOT NULL,
            name TEXT GENERATED ALWAYS as (json_extract(workflow, '$.name')) VIRTUAL NOT NULL,
            description TEXT GENERATED ALWAYS as (json_extract(workflow, '$.description')) VIRTUAL NOT NULL
        );
        """,
    ]

    indices = [
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_created_at ON workflow_library(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_updated_at ON workflow_library(updated_at);",
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_opened_at ON workflow_library(opened_at);",
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_category ON workflow_library(category);",
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_name ON workflow_library(name);",
        "CREATE INDEX IF NOT EXISTS idx_workflow_library_description ON workflow_library(description);",
    ]

    triggers = [
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_workflow_library_updated_at
        AFTER UPDATE
        ON workflow_library FOR EACH ROW
        BEGIN
            UPDATE workflow_library
            SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE workflow_id = old.workflow_id;
        END;
        """
    ]

    for stmt in tables + indices + triggers:
        cursor.execute(stmt)


def _drop_model_manager_metadata(cursor: sqlite3.Cursor) -> None:
    """Drops the `model_manager_metadata` table."""
    cursor.execute("DROP TABLE IF EXISTS model_manager_metadata;")


migration_2 = Migration(
    from_version=1,
    to_version=2,
    migrate=_migrate,
)
"""
Database version 2.

Introduced in v3.5.0 for the new workflow library.

Migration:
- Add `has_workflow` column to `images` table
- Add `workflow` column to `session_queue` table
- Drop `workflows` and `workflow_images` tables
- Add `workflow_library` table
"""
