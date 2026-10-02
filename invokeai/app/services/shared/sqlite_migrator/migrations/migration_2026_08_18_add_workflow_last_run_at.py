"""Add the `last_run_at` column to `workflow_library`.

Mirrors `opened_at`: a nullable timestamp, updated by a dedicated "touch" endpoint rather than as
a side effect of any existing write. `opened_at` records when a workflow was last viewed in the
editor; `last_run_at` records when a workflow was last completed as a run, so the frontend can
show "Your last run · 2 days ago" on library cards.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddWorkflowLastRunAtCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("ALTER TABLE workflow_library ADD COLUMN last_run_at DATETIME;")


def build_migration() -> Migration:
    """Build the migration that adds the nullable `last_run_at` column to `workflow_library`."""
    return Migration(
        id="2026_08_18_add_workflow_last_run_at",
        depends_on="2026_08_08_repair_image_subfolder_move_tables",
        callback=AddWorkflowLastRunAtCallback(),
    )
