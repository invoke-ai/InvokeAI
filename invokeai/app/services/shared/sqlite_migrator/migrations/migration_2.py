import sqlite3
from logging import Logger

from tqdm import tqdm

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationDependency

# This migration requires an ImageFileStorageBase service and logger
image_files_dependency = MigrationDependency(name="image_files", dependency_type=ImageFileStorageBase)
logger_dependency = MigrationDependency(name="logger", dependency_type=Logger)


def migrate_callback(cursor: sqlite3.Cursor, **kwargs) -> None:
    """Migration callback for database version 2."""

    logger = kwargs[logger_dependency.name]
    image_files = kwargs[image_files_dependency.name]

    _add_images_has_workflow(cursor)
    _add_session_queue_workflow(cursor)
    _drop_old_workflow_tables(cursor)
    _add_workflow_library(cursor)
    _drop_model_manager_metadata(cursor)
    _recreate_model_config(cursor)
    _migrate_embedded_workflows(cursor, logger, image_files)


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


def _recreate_model_config(cursor: sqlite3.Cursor) -> None:
    """
    Drops the `model_config` table, recreating it.

    In 3.4.0, this table used explicit columns but was changed to use json_extract 3.5.0.

    Because this table is not used in production, we are able to simply drop it and recreate it.
    """

    cursor.execute("DROP TABLE IF EXISTS model_config;")

    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS model_config (
            id TEXT NOT NULL PRIMARY KEY,
            -- The next 3 fields are enums in python, unrestricted string here
            base TEXT GENERATED ALWAYS as (json_extract(config, '$.base')) VIRTUAL NOT NULL,
            type TEXT GENERATED ALWAYS as (json_extract(config, '$.type')) VIRTUAL NOT NULL,
            name TEXT GENERATED ALWAYS as (json_extract(config, '$.name')) VIRTUAL NOT NULL,
            path TEXT GENERATED ALWAYS as (json_extract(config, '$.path')) VIRTUAL NOT NULL,
            format TEXT GENERATED ALWAYS as (json_extract(config, '$.format')) VIRTUAL NOT NULL,
            original_hash TEXT, -- could be null
            -- Serialized JSON representation of the whole config object,
            -- which will contain additional fields from subclasses
            config TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            -- Updated via trigger
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            -- unique constraint on combo of name, base and type
            UNIQUE(name, base, type)
        );
        """
    )


def _migrate_embedded_workflows(
    cursor: sqlite3.Cursor,
    logger: Logger,
    image_files: ImageFileStorageBase,
) -> None:
    """
    In the v3.5.0 release, InvokeAI changed how it handles embedded workflows. The `images` table in
    the database now has a `has_workflow` column, indicating if an image has a workflow embedded.

    This migrate callback checks each image for the presence of an embedded workflow, then updates its entry
    in the database accordingly.
    """
    # Get the total number of images and chunk it into pages
    cursor.execute("SELECT image_name FROM images")
    image_names: list[str] = [image[0] for image in cursor.fetchall()]
    total_image_names = len(image_names)

    if not total_image_names:
        return

    logger.info(f"Migrating workflows for {total_image_names} images")

    # Migrate the images
    to_migrate: list[tuple[bool, str]] = []
    pbar = tqdm(image_names)
    for idx, image_name in enumerate(pbar):
        pbar.set_description(f"Checking image {idx + 1}/{total_image_names} for workflow")
        pil_image = image_files.get(image_name)
        if "invokeai_workflow" in pil_image.info:
            to_migrate.append((True, image_name))

    logger.info(f"Adding {len(to_migrate)} embedded workflows to database")
    cursor.executemany("UPDATE images SET has_workflow = ? WHERE image_name = ?", to_migrate)


migration_2 = Migration(
    from_version=1,
    to_version=2,
    migrate_callback=migrate_callback,
    dependencies={image_files_dependency.name: image_files_dependency, logger_dependency.name: logger_dependency},
)
"""
Database version 2.

Introduced in v3.5.0 for the new workflow library.

Dependencies:
- image_files: ImageFileStorageBase
- logger: Logger

Migration:
- Add `has_workflow` column to `images` table
- Add `workflow` column to `session_queue` table
- Drop `workflows` and `workflow_images` tables
- Add `workflow_library` table
- Populates the `has_workflow` column in the `images` table (requires `image_files` & `logger` dependencies)
"""
