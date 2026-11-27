"""Keep size backfill queries proportional to the intermediates still awaiting measurement."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


def _index_unmeasured_intermediates(cursor: sqlite3.Cursor) -> None:
    for table in ("images", "videos"):
        # The query fixes is_intermediate to TRUE, so this index's implicit rowid suffix
        # supplies insertion order without sorting. Measured rows leave the partial index;
        # later batches therefore do not repeatedly scan all the completed work.
        cursor.execute(
            f"""--sql
            CREATE INDEX IF NOT EXISTS idx_{table}_unmeasured_intermediates
            ON {table}(is_intermediate)
            WHERE is_intermediate = TRUE AND file_size_bytes IS NULL;
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_09_22_index_unmeasured_intermediates",
        depends_on="2026_09_22_add_intermediates_management",
        callback=_index_unmeasured_intermediates,
    )
