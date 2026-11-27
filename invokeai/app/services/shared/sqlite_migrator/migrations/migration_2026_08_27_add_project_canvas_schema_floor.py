"""Add the monotonic canvas-schema compatibility floor to project metadata.

Project documents are client-owned and intentionally opaque to the server. The stored floor lets
the server refuse an older client before returning or replacing a document written with a newer
canvas schema. Existing projects are v2, so they are backfilled to schema 2.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddProjectCanvasSchemaFloorCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            ALTER TABLE projects
            ADD COLUMN minimum_canvas_schema_version INTEGER NOT NULL DEFAULT 2
                CHECK (minimum_canvas_schema_version >= 1);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_08_27_add_project_canvas_schema_floor",
        # This migration alters the table that the project-board migration rebuilds. Depending on
        # that table owner prevents the rebuild from dropping this column on databases where the
        # independent dated-migration branches are planned together.
        depends_on="2026_08_06_add_project_boards",
        callback=AddProjectCanvasSchemaFloorCallback(),
    )
