"""Store image and video semantic embeddings as little-endian float16 BLOBs.

After this migration commits, rollback to older application code requires restoring the pre-migration database backup.
"""

import sqlite3
import time
from logging import Logger

from invokeai.app.services.image_index.image_index_common import blob_to_embedding, embedding_to_blob
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

_BATCH_SIZE = 256
_PROGRESS_INTERVAL = 10_000
_EMBEDDING_TABLES = (("image_embeddings", "image_name"), ("video_embeddings", "video_name"))


class StoreEmbeddingsFp16Callback:
    """Add explicit encoding metadata and convert legacy float32 embedding rows."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        started_at = time.perf_counter()
        self._logger.info("Starting image and video embedding fp16 migration")

        converted_by_table: dict[str, int] = {}
        converted_total = 0
        try:
            for table, _ in _EMBEDDING_TABLES:
                if not self._has_encoding_column(cursor, table):
                    cursor.execute(
                        f"""ALTER TABLE {table}
                            ADD COLUMN encoding TEXT NOT NULL DEFAULT 'float32'
                            CHECK (encoding IN ('float32', 'float16'));"""
                    )

            for table, name_column in _EMBEDDING_TABLES:
                converted_by_table[table] = self._convert_table(cursor, table, name_column, converted_total)
                converted_total += converted_by_table[table]
        except Exception:
            elapsed = time.perf_counter() - started_at
            image_count = converted_by_table.get("image_embeddings", 0)
            video_count = converted_by_table.get("video_embeddings", 0)
            self._logger.info(
                "Embedding fp16 migration failed after %.2f seconds; converted image rows=%d, video rows=%d",
                elapsed,
                image_count,
                video_count,
            )
            raise

        elapsed = time.perf_counter() - started_at
        self._logger.info(
            "Completed image and video embedding fp16 migration: image rows=%d, video rows=%d, total rows=%d, "
            "elapsed=%.2f seconds",
            converted_by_table["image_embeddings"],
            converted_by_table["video_embeddings"],
            converted_total,
            elapsed,
        )

    @staticmethod
    def _has_encoding_column(cursor: sqlite3.Cursor, table: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table});")
        return any(row[1] == "encoding" for row in cursor.fetchall())

    def _convert_table(self, cursor: sqlite3.Cursor, table: str, name_column: str, converted_before: int) -> int:
        last_rowid: int | None = None
        converted = 0
        next_progress = (converted_before // _PROGRESS_INTERVAL + 1) * _PROGRESS_INTERVAL
        while True:
            if last_rowid is None:
                cursor.execute(
                    f"""SELECT rowid, {name_column}, model_id, dim, embedding
                        FROM {table}
                        WHERE encoding = 'float32'
                        ORDER BY rowid
                        LIMIT {_BATCH_SIZE};"""
                )
            else:
                cursor.execute(
                    f"""SELECT rowid, {name_column}, model_id, dim, embedding
                        FROM {table}
                        WHERE rowid > ? AND encoding = 'float32'
                        ORDER BY rowid
                        LIMIT {_BATCH_SIZE};""",
                    (last_rowid,),
                )
            batch = cursor.fetchall()
            if not batch:
                return converted

            for row in batch:
                rowid, record_name, model_id, dim, embedding = row
                last_rowid = rowid
                identity = f"{record_name} (model_id={model_id})"
                try:
                    decoded = blob_to_embedding(embedding, dim, "float32")
                    encoded = embedding_to_blob(decoded)
                    cursor.execute(
                        f"UPDATE {table} SET embedding = ?, encoding = 'float16' "
                        "WHERE rowid = ? AND encoding = 'float32';",
                        (encoded, rowid),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("row changed while migration was running")
                except Exception as e:
                    raise RuntimeError(f"Failed converting {table} record {identity!r}: {e}") from e

                converted += 1
                total = converted_before + converted
                if total >= next_progress:
                    self._logger.info(f"Converted {total} image and video embeddings to fp16")
                    next_progress = (total // _PROGRESS_INTERVAL + 1) * _PROGRESS_INTERVAL


def build_migration(logger: Logger) -> Migration:
    """Build the transactionally safe conversion from legacy float32 embedding BLOBs."""
    return Migration(
        id="2026_09_30_store_embeddings_fp16",
        depends_on="2026_09_12_add_video_embeddings",
        callback=StoreEmbeddingsFp16Callback(logger=logger),
    )
