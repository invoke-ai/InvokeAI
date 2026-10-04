"""Media references named by saved documents.

Project documents, library workflows, persisted client state (the legacy editor keeps its canvas
and reference images only there) and quarantined project documents are opaque JSON to the server,
but the assets they name are not opaque to cleanup: an intermediate a saved document still points
at (a canvas staging candidate, a cropped reference image) must not be collected as garbage. The
`media_references` table is that index. It is written in the same transaction as the document it describes, so a
save and its references are never observed apart, and the extractor here is the one definition of
"names a media asset" for both writers and the migration that backfills existing documents.

Extraction is deliberately over-inclusive: every string under an image- or video-name key counts,
history included. A false reference retains a file; a missed one deletes something in use.

Applied migrations call the table DDL, the extractors and `replace_media_references`: keep their DDL
and signatures frozen, since changing them changes what a fresh install's migrations do.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Literal

MediaReferenceOwnerKind = Literal["project", "workflow", "client_state", "quarantined_project"]
MediaKind = Literal["image", "video"]

IMAGE_NAME_KEYS = frozenset({"imageName", "image_name"})
VIDEO_NAME_KEYS = frozenset({"videoName", "video_name", "source_video_name"})

# Bounds a pathological document: names are generated UUID filenames, so anything longer is not
# one and is not worth a row.
_MAX_NAME_LENGTH = 255

_NAME_KEY_MARKERS = tuple(f'"{key}"' for key in (*IMAGE_NAME_KEYS, *VIDEO_NAME_KEYS))


@dataclass
class MediaReferences:
    images: set[str] = field(default_factory=set)
    videos: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not self.images and not self.videos


def extract_media_references(document: object) -> MediaReferences:
    """Collects every image and video name a JSON-like document names, at any depth."""
    references = MediaReferences()
    # Iterative: project documents nest deeply enough that recursion depth is not a bound worth
    # trusting.
    pending: list[object] = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if isinstance(value, str):
                if not value or len(value) > _MAX_NAME_LENGTH:
                    continue
                if key in IMAGE_NAME_KEYS:
                    references.images.add(value)
                elif key in VIDEO_NAME_KEYS:
                    references.videos.add(value)
                continue
            if isinstance(value, (dict, list)):
                pending.append(value)
    return references


def extract_media_references_from_json(document_json: str) -> MediaReferences:
    """Like `extract_media_references` for serialized JSON; malformed JSON names nothing."""
    # Most client-state values name no media; skip parsing them.
    if not any(marker in document_json for marker in _NAME_KEY_MARKERS):
        return MediaReferences()
    try:
        return extract_media_references(json.loads(document_json))
    except ValueError:
        return MediaReferences()


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
