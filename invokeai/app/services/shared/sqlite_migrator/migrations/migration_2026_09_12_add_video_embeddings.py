"""Extend the semantic index to gallery videos.

A video is indexed by its thumbnail — the representative frame the video file
store already extracts when the video is created — so a video takes part in the
image map and in semantic search exactly as an image does.

Videos keep their own embedding table rather than sharing `image_embeddings`:
the FK that deletes an embedding with its media can only reference one table,
and eligibility and board access are answered from `videos`/`board_videos` with
their own columns. One table per namespace keeps both honest.

`image_projections` gains `item_kinds`, a JSON array row-aligned with
`image_names` naming the namespace each name belongs to. It is nullable, and a
NULL means what every projection cached before this migration was: all images.
So existing caches stay valid instead of every user paying for a recomputed
projection on first load after an upgrade.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddVideoEmbeddingsCallback:
    """Migration to add the video_embeddings table and the projection item_kinds column."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._assert_videos_table_exists(cursor)
        self._create_video_embeddings_table(cursor)
        self._add_projection_item_kinds(cursor)

    def _assert_videos_table_exists(self, cursor: sqlite3.Cursor) -> None:
        """Fail here rather than at the first insert if `videos` has not been created yet.

        A migration declares one `depends_on`, and this one names the migration whose column it
        alters. Its other prerequisite — `videos`, which the new foreign key references — is
        satisfied by run order (dated migrations run in date order, and the videos tables are
        two months older). SQLite resolves foreign-key targets lazily, so getting that wrong
        would create the table here and fail obscurely on the first embedding written months
        later; this turns it into a migration failure naming the cause.
        """
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'videos';")
        if cursor.fetchone() is None:
            raise RuntimeError(
                "Cannot add video_embeddings: the videos table does not exist. "
                "2026_07_01_add_videos_tables must run first."
            )

    def _create_video_embeddings_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS video_embeddings (
                video_name TEXT NOT NULL,
                -- Content hash of the embedding model, not its install key.
                model_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                -- float32, L2-normalized, dim * 4 bytes.
                embedding BLOB NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (video_name, model_id),
                FOREIGN KEY (video_name) REFERENCES videos(video_name) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_video_embeddings_model_id ON video_embeddings(model_id);
            """
        )

    def _add_projection_item_kinds(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(image_projections);")
        if any(row[1] == "item_kinds" for row in cursor.fetchall()):
            return
        cursor.execute(
            """--sql
            ALTER TABLE image_projections ADD COLUMN item_kinds TEXT;
            """
        )


def build_migration() -> Migration:
    """Build the migration that extends the semantic index to videos:
    - `video_embeddings` (global per-video embedding index).
    - `image_projections.item_kinds` (which namespace each projected name belongs to).
    """
    return Migration(
        id="2026_09_12_add_video_embeddings",
        # Adds the sibling of `image_embeddings` and a column to `image_projections`,
        # both created by this migration. `videos`, which the new FK references, is
        # older still (2026_07_01_add_videos_tables).
        depends_on="2026_08_03_add_image_index_tables",
        callback=AddVideoEmbeddingsCallback(),
    )
