"""Add the supplementary cluster-labeling vocabulary table.

The semantic image map labels clusters by zero-shot similarity against a
bundled vocabulary (`cluster_vocab.txt`). `image_index_vocab_terms` holds the
admin-maintained supplementary terms that are merged with the bundled list at
embedding-build time.

The table is server-wide rather than per-user: phrase embeddings are built
once per embedding model and shared by every user's cluster labels, so a
per-user vocabulary would need a per-user embedding build.

Terms are stored normalized (lowercased, whitespace-collapsed) by the API
layer; COLLATE NOCASE on the primary key is a backstop against case-variant
duplicates reaching the table by any other route.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddImageIndexVocabTermsCallback:
    """Migration to add the image_index_vocab_terms table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS image_index_vocab_terms (
                term TEXT NOT NULL COLLATE NOCASE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (term)
            );
            """
        )


def build_migration() -> Migration:
    """Build the migration that adds the supplementary vocabulary table for
    semantic image map cluster labeling."""
    return Migration(
        id="2026_08_23_add_image_index_vocab_terms",
        depends_on="2026_08_18_add_workflow_last_run_at",
        callback=AddImageIndexVocabTermsCallback(),
    )
