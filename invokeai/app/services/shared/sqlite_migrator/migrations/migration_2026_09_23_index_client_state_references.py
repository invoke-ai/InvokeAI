"""Index the media that persisted client state and quarantined projects name.

The legacy editor keeps its canvas layers and reference images only in `client_state`, as
intermediates, and `orphaned_projects_2026_08_06` holds project documents kept for repair. Both are
now `media_references` owners, so cleanup no longer classifies their media as unused. Replacing
each owner's rows makes a re-run harmless. Expired browser holds are swept on every hold write, so
they get an index of their own.
"""

import sqlite3
from logging import Logger

from invokeai.app.services.shared.media_references import MediaReferenceOwnerKind, extract_media_references_from_json
from invokeai.app.services.shared.sqlite_migrator.migrations._media_references_v1 import replace_media_references
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class IndexClientStateReferencesCallback:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        # This table was added to the already-released September 22 migration. Databases that
        # applied its earlier form need it created by a migration they have not yet run.
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
            "CREATE INDEX IF NOT EXISTS idx_intermediates_browser_holds_expires_at "
            "ON intermediates_browser_holds(expires_at);"
        )
        indexed = 0
        if _table_exists(cursor, "client_state"):
            # Rows of a departed account (a cascade that never fired) would pin their media forever:
            # only account deletion clears client-state references.
            live = " WHERE user_id IN (SELECT user_id FROM users)" if _table_exists(cursor, "users") else ""
            indexed += self._index_rows(cursor, "client_state", f"SELECT user_id, key, value FROM client_state{live};")
        if _table_exists(cursor, "orphaned_projects_2026_08_06"):
            indexed += self._index_rows(
                cursor, "quarantined_project", "SELECT user_id, project_id, data FROM orphaned_projects_2026_08_06;"
            )
        if indexed:
            self._logger.info(f"Indexed media references of {indexed} client state value(s) and quarantined project(s)")

    @staticmethod
    def _index_rows(cursor: sqlite3.Cursor, owner_kind: MediaReferenceOwnerKind, select: str) -> int:
        # A second cursor streams the rows; the writes run on `cursor`, which would discard a
        # result set being iterated.
        reader = cursor.connection.cursor()
        reader.execute(select)
        indexed = 0
        for user_id, owner_id, document_json in reader:
            references = extract_media_references_from_json(document_json if isinstance(document_json, str) else "")
            if references.is_empty():
                continue
            replace_media_references(
                cursor, owner_kind=owner_kind, user_id=user_id, owner_id=owner_id, references=references
            )
            indexed += 1
        reader.close()
        return indexed


def _table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cursor.fetchone() is not None


def build_migration(logger: Logger) -> Migration:
    return Migration(
        id="2026_09_23_index_client_state_references",
        depends_on="2026_09_23_track_queue_session_revision",
        callback=IndexClientStateReferencesCallback(logger),
    )
