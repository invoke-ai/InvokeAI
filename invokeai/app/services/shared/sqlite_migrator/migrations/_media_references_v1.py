"""The `media_references` table as the SQLite-only migrations create and fill it, frozen with them.

Changing this changes what a fresh install's migrations do. The application writes through
`database.queries.media_references`.
"""

import sqlite3

from invokeai.app.services.shared.media_references import MediaReferenceOwnerKind, MediaReferences


def replace_media_references(
    cursor: sqlite3.Cursor,
    *,
    owner_kind: MediaReferenceOwnerKind,
    user_id: str,
    owner_id: str,
    references: MediaReferences,
) -> None:
    """Makes the index for one owner equal to `references`, on the caller's transaction."""
    delete_media_references(cursor, owner_kind=owner_kind, user_id=user_id, owner_id=owner_id)
    rows: list[tuple[str, str, str, str, str]] = []
    rows.extend((owner_kind, user_id, owner_id, "image", name) for name in sorted(references.images))
    rows.extend((owner_kind, user_id, owner_id, "video", name) for name in sorted(references.videos))
    if rows:
        cursor.executemany(
            """--sql
            INSERT OR IGNORE INTO media_references (owner_kind, user_id, owner_id, media_kind, media_name)
            VALUES (?, ?, ?, ?, ?);
            """,
            rows,
        )


def delete_media_references(
    cursor: sqlite3.Cursor, *, owner_kind: MediaReferenceOwnerKind, user_id: str, owner_id: str
) -> None:
    cursor.execute(
        """--sql
        DELETE FROM media_references
        WHERE owner_kind = ? AND user_id = ? AND owner_id = ?;
        """,
        (owner_kind, user_id, owner_id),
    )


def create_media_references_table(cursor: sqlite3.Cursor) -> None:
    """DDL shared by the migration and tests that need the table without the whole chain."""
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS media_references (
            -- A MediaReferenceOwnerKind
            owner_kind TEXT NOT NULL,
            -- The owning account; project ids are unique per user, not globally.
            user_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            -- 'image' or 'video'
            media_kind TEXT NOT NULL,
            media_name TEXT NOT NULL,
            PRIMARY KEY (owner_kind, user_id, owner_id, media_kind, media_name)
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_media_references_media
        ON media_references(media_kind, media_name);
        """
    )
