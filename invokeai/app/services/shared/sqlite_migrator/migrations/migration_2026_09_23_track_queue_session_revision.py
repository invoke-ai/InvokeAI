"""Give active queue sessions a monotonic revision for cleanup's media-input cache."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


def _track_queue_session_revision(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(session_queue);")
    if "session_revision" not in {row[1] for row in cursor.fetchall()}:
        cursor.execute("ALTER TABLE session_queue ADD COLUMN session_revision INTEGER NOT NULL DEFAULT 0;")
    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_session_queue_session_revision
        AFTER UPDATE OF session ON session_queue
        FOR EACH ROW
        WHEN NEW.session IS NOT OLD.session
        BEGIN
            UPDATE session_queue SET session_revision = OLD.session_revision + 1
            WHERE item_id = NEW.item_id;
        END;
        """
    )


def build_migration() -> Migration:
    return Migration(
        id="2026_09_23_track_queue_session_revision",
        depends_on="2026_09_22_index_unmeasured_intermediates",
        callback=_track_queue_session_revision,
    )
