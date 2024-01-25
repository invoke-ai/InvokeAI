import sqlite3
from logging import Logger

from pydantic import ValidationError
from tqdm import tqdm

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.app.services.workflow_records.workflow_records_common import (
    UnsafeWorkflowWithVersionValidator,
)


class Migration2Callback:
    def __init__(self, image_files: ImageFileStorageBase, logger: Logger):
        self._image_files = image_files
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor):
        self._add_images_has_workflow(cursor)
        self._add_session_queue_workflow(cursor)
        self._drop_old_workflow_tables(cursor)
        self._add_workflow_library(cursor)
        self._drop_model_manager_metadata(cursor)
        self._migrate_embedded_workflows(cursor)

    def _add_images_has_workflow(self, cursor: sqlite3.Cursor) -> None:
        """Add the `has_workflow` column to `images` table."""
        cursor.execute("PRAGMA table_info(images)")
        columns = [column[1] for column in cursor.fetchall()]

        if "has_workflow" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN has_workflow BOOLEAN DEFAULT FALSE;")

    def _add_session_queue_workflow(self, cursor: sqlite3.Cursor) -> None:
        """Add the `workflow` column to `session_queue` table."""

        cursor.execute("PRAGMA table_info(session_queue)")
        columns = [column[1] for column in cursor.fetchall()]

        if "workflow" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN workflow TEXT;")

    def _drop_old_workflow_tables(self, cursor: sqlite3.Cursor) -> None:
        """Drops the `workflows` and `workflow_images` tables."""
        cursor.execute("DROP TABLE IF EXISTS workflow_images;")
        cursor.execute("DROP TABLE IF EXISTS workflows;")

    def _add_workflow_library(self, cursor: sqlite3.Cursor) -> None:
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

    def _drop_model_manager_metadata(self, cursor: sqlite3.Cursor) -> None:
        """Drops the `model_manager_metadata` table."""
        cursor.execute("DROP TABLE IF EXISTS model_manager_metadata;")

    def _migrate_embedded_workflows(self, cursor: sqlite3.Cursor) -> None:
        """
        In the v3.5.0 release, InvokeAI changed how it handles embedded workflows. The `images` table in
        the database now has a `has_workflow` column, indicating if an image has a workflow embedded.

        This migrate callback checks each image for the presence of an embedded workflow, then updates its entry
        in the database accordingly.
        """
        # Get all image names
        cursor.execute("SELECT image_name FROM images")
        image_names: list[str] = [image[0] for image in cursor.fetchall()]
        total_image_names = len(image_names)

        if not total_image_names:
            return

        self._logger.info(f"Migrating workflows for {total_image_names} images")

        # Migrate the images
        to_migrate: list[tuple[bool, str]] = []
        pbar = tqdm(image_names)
        for idx, image_name in enumerate(pbar):
            pbar.set_description(f"Checking image {idx + 1}/{total_image_names} for workflow")
            try:
                pil_image = self._image_files.get(image_name)
            except ImageFileNotFoundException:
                self._logger.warning(f"Image {image_name} not found, skipping")
                continue
            except Exception as e:
                self._logger.warning(f"Error while checking image {image_name}, skipping: {e}")
                continue
            if "invokeai_workflow" in pil_image.info:
                try:
                    UnsafeWorkflowWithVersionValidator.validate_json(pil_image.info.get("invokeai_workflow", ""))
                except ValidationError:
                    self._logger.warning(f"Image {image_name} has invalid embedded workflow, skipping")
                    continue
                to_migrate.append((True, image_name))

        self._logger.info(f"Adding {len(to_migrate)} embedded workflows to database")
        cursor.executemany("UPDATE images SET has_workflow = ? WHERE image_name = ?", to_migrate)


def build_migration_2(image_files: ImageFileStorageBase, logger: Logger) -> Migration:
    """
    Builds the migration from database version 1 to 2.

    Introduced in v3.5.0 for the new workflow library.

    :param image_files: The image files service, used to check for embedded workflows
    :param logger: The logger, used to log progress during embedded workflows handling

    This migration does the following:
    - Add `has_workflow` column to `images` table
    - Add `workflow` column to `session_queue` table
    - Drop `workflows` and `workflow_images` tables
    - Add `workflow_library` table
    - Drops the `model_manager_metadata` table
    - Drops the `model_config` table, recreating it (at this point, there is no user data in this table)
    - Populates the `has_workflow` column in the `images` table (requires `image_files` & `logger` dependencies)
    """
    migration_2 = Migration(
        from_version=1,
        to_version=2,
        callback=Migration2Callback(image_files=image_files, logger=logger),
    )

    return migration_2
