"""Give every library workflow a monotonic content revision so explicit template updates can refuse stale writes."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


def _add_workflow_revision(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(workflow_library);")
    columns = {row[1] for row in cursor.fetchall()}
    if "revision" not in columns:
        cursor.execute("ALTER TABLE workflow_library ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;")


def build_migration() -> Migration:
    return Migration(
        id="2026_09_26_add_workflow_revision",
        depends_on="2026_09_24_drop_intermediates_operations",
        callback=_add_workflow_revision,
    )
