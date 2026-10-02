"""Cleanup operations no longer persist: a run lives in server memory and the live policy is its retry.

The applied `2026_09_22_add_intermediates_management` migration still creates these tables and is
left untouched; this one removes them wherever it ran.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


def _drop_intermediates_operations(cursor: sqlite3.Cursor) -> None:
    cursor.execute("DROP TABLE IF EXISTS intermediates_operation_targets;")
    cursor.execute("DROP TABLE IF EXISTS intermediates_operations;")


def build_migration() -> Migration:
    return Migration(
        id="2026_09_24_drop_intermediates_operations",
        depends_on="2026_09_23_index_client_state_references",
        callback=_drop_intermediates_operations,
    )
