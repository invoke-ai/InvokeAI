"""Schema for the intermediates manager.

Intermediates could only ever be cleared instance-wide, and nothing recorded which project made a
file, how large it is, or whether a saved document still needs it. This migration adds:

- `session_queue.project_id`: the project a batch was enqueued for. `Batch.project_id` was validated
  at enqueue and then dropped; persisting it lets outputs inherit it.
- `images.project_id` / `videos.project_id`: where the media originated. Existing rows stay NULL
  ("unassigned"): board membership and free-form queue origins are not provenance, so nothing is
  inferred.
- `images.file_size_bytes` / `videos.file_size_bytes`: on-disk size of the media file plus its
  thumbnail (and video sidecar). NULL means not yet measured, never zero.
- `media_references`: the assets every saved project document and user library workflow names, backfilled
  here from the existing rows and kept current by their writers from now on.
"""

import json
import sqlite3
from logging import Logger

from invokeai.app.services.shared.media_references import (
    create_media_references_table,
    extract_media_references,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddIntermediatesManagementCallback:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._add_column(cursor, "session_queue", "project_id", "TEXT")
        for table in ("images", "videos"):
            if not _table_exists(cursor, table):
                continue
            self._add_column(cursor, table, "project_id", "TEXT")
            self._add_column(cursor, table, "file_size_bytes", "INTEGER")
            # Summaries group intermediates by owner and project; the partial index keeps that off
            # the far larger durable population.
            cursor.execute(
                f"""--sql
                CREATE INDEX IF NOT EXISTS idx_{table}_intermediate_scope
                ON {table}(user_id, project_id)
                WHERE is_intermediate = TRUE;
                """
            )
        create_media_references_table(cursor)
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS intermediates_operations (
                operation_id TEXT PRIMARY KEY,
                caller_user_id TEXT NOT NULL,
                preview_id TEXT,
                idempotency_key TEXT,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(caller_user_id, idempotency_key)
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS intermediates_browser_holds (
                user_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                media_kind TEXT NOT NULL,
                media_name TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY(user_id, lease_id, media_kind, media_name)
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_intermediates_browser_holds_media "
            "ON intermediates_browser_holds(media_kind, media_name, expires_at);"
        )
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS intermediates_operation_targets (
                operation_id TEXT NOT NULL,
                media_kind TEXT NOT NULL,
                media_name TEXT NOT NULL,
                size_bytes INTEGER,
                confirmed_refs_json TEXT NOT NULL,
                PRIMARY KEY(operation_id, media_kind, media_name)
            );
            """
        )
        self._backfill_references(cursor)

    @staticmethod
    def _add_column(cursor: sqlite3.Cursor, table: str, column: str, declaration: str) -> None:
        cursor.execute(f"PRAGMA table_info({table});")
        if any(row[1] == column for row in cursor.fetchall()):
            return
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration};")

    def _backfill_references(self, cursor: sqlite3.Cursor) -> None:
        # Only a database that has never had the table needs the backfill; on one that already
        # migrated, the writers have kept it current and re-deriving would only race them.
        cursor.execute("SELECT COUNT(*) FROM media_references;")
        if cursor.fetchone()[0] != 0:
            return

        indexed = 0
        if _table_exists(cursor, "projects"):
            indexed += self._index_rows(
                cursor,
                owner_kind="project",
                select="SELECT user_id, project_id, data FROM projects ORDER BY rowid ASC;",
            )
        if _table_exists(cursor, "workflow_library"):
            # Only user workflows: default workflows are synced from files and never indexed at runtime.
            indexed += self._index_rows(
                cursor,
                owner_kind="workflow",
                select=(
                    "SELECT user_id, workflow_id, workflow FROM workflow_library"
                    " WHERE category = 'user' ORDER BY rowid ASC;"
                ),
            )
        if indexed:
            self._logger.info(f"Intermediates management migration: indexed media references of {indexed} document(s)")

    def _index_rows(self, cursor: sqlite3.Cursor, *, owner_kind: str, select: str) -> int:
        # A second cursor streams the documents one at a time: `data` is a whole project document,
        # and the writes below run on `cursor`, which would discard a result set being iterated.
        reader = cursor.connection.cursor()
        reader.execute(select)
        indexed = 0
        for user_id, owner_id, document_json in reader:
            try:
                document = json.loads(document_json)
            except (TypeError, ValueError):
                self._logger.warning(
                    f"Intermediates management migration: {owner_kind} {owner_id} of {user_id} is not valid JSON;"
                    " its media references are not indexed"
                )
                continue
            references = extract_media_references(document)
            rows = [(owner_kind, user_id or "system", owner_id, "image", name) for name in sorted(references.images)]
            rows += [(owner_kind, user_id or "system", owner_id, "video", name) for name in sorted(references.videos)]
            if rows:
                cursor.executemany(
                    """--sql
                    INSERT OR IGNORE INTO media_references (owner_kind, user_id, owner_id, media_kind, media_name)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    rows,
                )
            indexed += 1
        reader.close()
        return indexed


def _table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cursor.fetchone() is not None


def build_migration(logger: Logger) -> Migration:
    return Migration(
        id="2026_09_22_add_intermediates_management",
        depends_on="2026_09_16_add_qwen3_vl_encoder_variant",
        callback=AddIntermediatesManagementCallback(logger),
    )
