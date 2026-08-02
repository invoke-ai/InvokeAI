import json
import sqlite3

import numpy as np

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_index.image_index_common import (
    EMBEDDING_DTYPE,
    ImageIndexStatus,
    ProjectionRecord,
    blob_to_coords,
    blob_to_embedding,
    coords_to_blob,
    embedding_to_blob,
)
from invokeai.app.services.image_index.image_index_records_base import ImageIndexRecordsBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

# SQLite's default variable limit is 999; stay well under it when chunking IN clauses.
_IN_CLAUSE_CHUNK = 500

# Conditions defining a "gallery" image, i.e. one worth indexing.
_ELIGIBLE_IMAGE_CONDITIONS = "images.is_intermediate = ? AND images.image_category = ?"


def _eligible_params() -> tuple[bool, str]:
    return (False, ImageCategory.GENERAL.value)


class ImageIndexRecordsSqlite(ImageIndexRecordsBase):
    """SQLite implementation of semantic image index storage."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def upsert_embedding(self, image_name: str, model_id: str, embedding: np.ndarray) -> None:
        blob = embedding_to_blob(embedding)
        try:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    INSERT INTO image_embeddings (image_name, model_id, dim, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (image_name, model_id)
                    DO UPDATE SET dim = excluded.dim, embedding = excluded.embedding;
                    """,
                    (image_name, model_id, embedding.shape[0], blob),
                )
        except sqlite3.IntegrityError:
            # The image was deleted between being scheduled and embedded (the
            # FK parent is gone). Its embedding is pointless; drop it silently.
            pass

    def get_embeddings(self, image_names: list[str], model_id: str) -> tuple[list[str], np.ndarray]:
        # Dedupe while preserving order so repeated input names cannot
        # double-count rows in downstream projection/similarity math.
        image_names = list(dict.fromkeys(image_names))
        found_names: list[str] = []
        vectors: list[np.ndarray] = []
        dim: int | None = None

        with self._db.transaction() as cursor:
            for start in range(0, len(image_names), _IN_CLAUSE_CHUNK):
                chunk = image_names[start : start + _IN_CLAUSE_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cursor.execute(
                    f"""--sql
                    SELECT image_name, dim, embedding
                    FROM image_embeddings
                    WHERE model_id = ? AND image_name IN ({placeholders});
                    """,
                    (model_id, *chunk),
                )
                rows = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
                # Preserve the caller's ordering within each chunk.
                for name in chunk:
                    if name not in rows:
                        continue
                    row_dim, blob = rows[name]
                    if dim is None:
                        dim = row_dim
                    elif row_dim != dim:
                        raise ValueError(f"Inconsistent embedding dims for model {model_id}: found {row_dim} and {dim}")
                    found_names.append(name)
                    vectors.append(blob_to_embedding(blob, row_dim))

        if not vectors:
            return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)
        return found_names, np.stack(vectors)

    def delete_embedding(self, image_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM image_embeddings WHERE image_name = ?;
                """,
                (image_name,),
            )

    def delete_embeddings_for_other_models(self, model_id: str) -> int:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM image_embeddings WHERE model_id != ?;
                """,
                (model_id,),
            )
            return cursor.rowcount

    def list_unembedded_image_names(self, model_id: str, limit: int) -> list[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT images.image_name
                FROM images
                LEFT JOIN image_embeddings
                  ON image_embeddings.image_name = images.image_name
                  AND image_embeddings.model_id = ?
                WHERE {_ELIGIBLE_IMAGE_CONDITIONS}
                  AND image_embeddings.image_name IS NULL
                ORDER BY images.created_at ASC, images.image_name ASC
                LIMIT ?;
                """,
                (model_id, *_eligible_params(), limit),
            )
            return [row[0] for row in cursor.fetchall()]

    def count_index_status(self, model_id: str) -> ImageIndexStatus:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT
                  COUNT(*),
                  COUNT(image_embeddings.image_name)
                FROM images
                LEFT JOIN image_embeddings
                  ON image_embeddings.image_name = images.image_name
                  AND image_embeddings.model_id = ?
                WHERE {_ELIGIBLE_IMAGE_CONDITIONS};
                """,
                (model_id, *_eligible_params()),
            )
            total, embedded = cursor.fetchone()
        return ImageIndexStatus(total=total, embedded=embedded)

    def list_accessible_embedded_images(self, user_id: str | None, model_id: str) -> list[str]:
        # Both clauses mirror the gallery "all" listing semantics
        # (image_records_sqlite): images on archived boards are hidden from
        # every scope, and a scoped user sees their own unboarded images plus
        # images on active boards they own, that are shared/public, or that
        # were individually shared with them via shared_boards. The projection
        # scope_hash and semantic search both derive from this listing, so any
        # change here must keep matching the gallery's access model.
        params: list[object] = [model_id, *_eligible_params()]
        if user_id is None:
            # Administrative scope: everything except archived-board images.
            access_clause = """AND (
                    board_images.board_id IS NULL
                    OR EXISTS (
                      SELECT 1 FROM boards
                      WHERE boards.board_id = board_images.board_id
                        AND boards.archived = 0
                    )
                  )"""
        else:
            access_clause = """AND (
                    (board_images.board_id IS NULL AND images.user_id = ?)
                    OR EXISTS (
                      SELECT 1 FROM boards
                      WHERE boards.board_id = board_images.board_id
                        AND boards.archived = 0
                        AND (
                          boards.user_id = ?
                          OR boards.board_visibility IN (?, ?)
                          OR EXISTS (
                            SELECT 1 FROM shared_boards
                            WHERE shared_boards.board_id = boards.board_id
                              AND shared_boards.user_id = ?
                          )
                        )
                    )
                  )"""
            params.extend([user_id, user_id, BoardVisibility.Shared.value, BoardVisibility.Public.value, user_id])

        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT DISTINCT image_embeddings.image_name
                FROM image_embeddings
                JOIN images ON images.image_name = image_embeddings.image_name
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE image_embeddings.model_id = ?
                  AND {_ELIGIBLE_IMAGE_CONDITIONS}
                  {access_clause}
                ORDER BY image_embeddings.image_name ASC;
                """,
                params,
            )
            return [row[0] for row in cursor.fetchall()]

    def get_projection(self, user_id: str, model_id: str) -> ProjectionRecord | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT scope_hash, params, point_count, image_names, coords, created_at, updated_at
                FROM image_projections
                WHERE user_id = ? AND model_id = ?;
                """,
                (user_id, model_id),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return ProjectionRecord(
            user_id=user_id,
            model_id=model_id,
            scope_hash=row[0],
            params=row[1],
            point_count=row[2],
            image_names=json.loads(row[3]),
            coords=blob_to_coords(row[4], row[2]),
            created_at=row[5],
            updated_at=row[6],
        )

    def set_projection(
        self,
        user_id: str,
        model_id: str,
        scope_hash: str,
        params: str,
        image_names: list[str],
        coords: np.ndarray,
    ) -> None:
        if len(image_names) != coords.shape[0]:
            raise ValueError(f"Got {len(image_names)} image names but {coords.shape[0]} coordinate rows")
        blob = coords_to_blob(coords)
        try:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    INSERT INTO image_projections (user_id, model_id, scope_hash, params, point_count, image_names, coords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, model_id)
                    DO UPDATE SET
                      scope_hash = excluded.scope_hash,
                      params = excluded.params,
                      point_count = excluded.point_count,
                      image_names = excluded.image_names,
                      coords = excluded.coords;
                    """,
                    (user_id, model_id, scope_hash, params, len(image_names), json.dumps(image_names), blob),
                )
        except sqlite3.IntegrityError:
            # The user was deleted while their projection was being computed;
            # there is nobody left to serve it to.
            pass

    def delete_projection(self, user_id: str, model_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM image_projections WHERE user_id = ? AND model_id = ?;
                """,
                (user_id, model_id),
            )
