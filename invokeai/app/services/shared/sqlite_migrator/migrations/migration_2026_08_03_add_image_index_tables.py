"""Add tables for the semantic image index.

The image map / semantic search feature keeps a vision-model embedding for
each gallery image plus a cached 2D UMAP projection per user:

- `image_embeddings` is global (embeddings are user-independent): one row per
  (image_name, model_id), where model_id is the embedding model's content
  hash so re-installs of the same weights keep the index valid.
- `image_projections` caches one UMAP projection per (user_id, model_id) over
  the set of images that user can access. `scope_hash` fingerprints that set
  so staleness is detected by re-deriving it, never by bookkeeping.

The DDL uses IF NOT EXISTS so databases that acquired these tables from a
pre-rename build of this feature (which shipped them as a numeric migration)
no-op cleanly.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddImageIndexTablesCallback:
    """Migration to add the image_embeddings and image_projections tables."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_image_embeddings_table(cursor)
        self._create_image_projections_table(cursor)

    def _create_image_embeddings_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS image_embeddings (
                image_name TEXT NOT NULL,
                -- Content hash of the embedding model, not its install key.
                model_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                -- float32, L2-normalized, dim * 4 bytes.
                embedding BLOB NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (image_name, model_id),
                FOREIGN KEY (image_name) REFERENCES images(image_name) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_image_embeddings_model_id ON image_embeddings(model_id);
            """
        )

    def _create_image_projections_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS image_projections (
                user_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                -- Fingerprint of the accessible image set the projection was
                -- computed over; a mismatch against the current set marks the
                -- cached projection stale.
                scope_hash TEXT NOT NULL,
                -- JSON of the projection parameters (n_neighbors, min_dist, ...).
                params TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                -- JSON array of image names, row-aligned with coords.
                image_names TEXT NOT NULL,
                -- float32, shape (point_count, 2).
                coords BLOB NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, model_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_image_projections_updated_at
            AFTER UPDATE ON image_projections
            FOR EACH ROW
            BEGIN
              UPDATE image_projections
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
              WHERE user_id = OLD.user_id AND model_id = OLD.model_id;
            END;
            """
        )


def build_migration() -> Migration:
    """Build the migration that adds the image index tables:
    - `image_embeddings` (global per-image embedding index).
    - `image_projections` (per-user cached UMAP projections).
    """
    return Migration(
        id="2026_08_03_add_image_index_tables",
        # images(image_name) predates migration_33; migration_33 also transitively
        # guarantees migration_27's `users` table, which image_projections' FK
        # references. Same dependency reasoning as 2026_07_30_repair_projects_table.
        depends_on="migration_33",
        callback=AddImageIndexTablesCallback(),
    )
