import json

import numpy as np

from invokeai.app.services.image_index.image_index_common import (
    EMBEDDING_DTYPE,
    ImageIndexStatus,
    IndexedItem,
    MediaKind,
    ProjectionRecord,
    blob_to_coords,
    blobs_to_embeddings,
    coords_to_blob,
    embedding_to_blob,
)
from invokeai.app.services.image_index.image_index_records_base import ImageIndexRecordsBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.backend.util.logging import InvokeAILogger

# What `embedding_to_blob` writes.
_STORED_ENCODING = "float16"


class ImageIndexRecords(ImageIndexRecordsBase):
    """Semantic image index storage."""

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries
        self._logger = InvokeAILogger.get_logger(self.__class__.__name__)

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def upsert_embedding(self, item: IndexedItem, model_id: str, embedding: np.ndarray) -> None:
        blob = embedding_to_blob(embedding)
        # The item may be deleted between being scheduled and embedded, which makes the embedding pointless.
        if not self._queries.image_index.upsert_embedding(item, model_id, embedding.shape[0], blob, _STORED_ENCODING):
            self._logger.debug(f"Skipped embedding for missing {item.kind} {item.name}")

    def get_embeddings(self, items: list[IndexedItem], model_id: str) -> tuple[list[IndexedItem], np.ndarray]:
        # Dedupe while preserving order so repeated input items cannot
        # double-count rows in downstream projection/similarity math.
        items = list(dict.fromkeys(items))
        rows = self._queries.image_index.embeddings(items, model_id)

        # Read back in the caller's order: the returned matrix's rows align with it.
        found_items = [item for item in items if item in rows]

        if not found_items:
            return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)

        dim = rows[found_items[0]][0]
        for item in found_items:
            row_dim = rows[item][0]
            if row_dim != dim:
                raise ValueError(f"Inconsistent embedding dims for model {model_id}: found {row_dim} and {dim}")
        matrix = blobs_to_embeddings(
            [rows[item][1] for item in found_items], [rows[item][2] for item in found_items], dim
        )
        return found_items, matrix

    def delete_embedding(self, item: IndexedItem) -> None:
        self._queries.image_index.delete_embedding(item)

    def delete_embeddings_for_other_models(self, model_id: str) -> int:
        return self._queries.image_index.delete_embeddings_for_other_models(model_id)

    def list_unembedded_items(self, model_id: str, limit: int) -> list[IndexedItem]:
        if limit < 0:
            # SQLite reads a negative LIMIT as unbounded, which would turn a backfill batch into a full-table load,
            # and MySQL rejects it. Fail on a miscomputed limit instead.
            raise ValueError(f"limit must be non-negative, got {limit}")
        return self._queries.image_index.unembedded(model_id, limit)

    def count_index_status(self, model_id: str) -> ImageIndexStatus:
        total, embedded = self._queries.image_index.status(model_id)
        return ImageIndexStatus(total=total, embedded=embedded)

    def list_accessible_embedded_items(self, user_id: str | None, model_id: str) -> list[IndexedItem]:
        # The projection scope_hash and semantic search both derive from this listing, so it must keep matching the
        # gallery's access model.
        return self._queries.image_index.accessible_embedded(user_id, model_id)

    def get_custom_vocab_terms(self) -> list[str]:
        return self._queries.image_index.vocab_terms()

    def set_custom_vocab_terms(self, terms: list[str]) -> None:
        def replace(q: Queries) -> None:
            # On a server another replace would otherwise delete only what it sees committed, leaving both lists.
            q.locks.acquire(DatabaseLock.IMAGE_INDEX_VOCABULARY)
            q.image_index.replace_vocab_terms(terms)

        self._queries.run(replace)

    def get_projection(self, user_id: str, model_id: str) -> ProjectionRecord | None:
        row = self._queries.image_index.projection(user_id, model_id)
        if row is None:
            return None

        names: list[str] = json.loads(row.image_names)
        # A projection cached before videos were indexable has no kinds: every name in it is an
        # image, which is what a NULL column means. Same reading for a row written by this code,
        # where the column is always present.
        kinds: list[MediaKind] = json.loads(row.item_kinds) if row.item_kinds is not None else ["image"] * len(names)
        if len(kinds) != len(names):
            # Unreachable from this code — both arrays are written by one statement from one
            # list — but reachable by downgrading to a build that rewrites `image_names` and
            # leaves `item_kinds` behind. Reported as "no cached projection" rather than
            # raised: every reader is a request handler or the worker's own fit, so raising
            # would 500 the map and spin the worker on a row that only `set_projection`
            # repairs — which is never reached while the read keeps failing.
            self._logger.warning(
                f"Discarding the cached projection for user {user_id}: {len(names)} names but {len(kinds)} kinds"
            )
            return None

        return ProjectionRecord(
            user_id=user_id,
            model_id=model_id,
            scope_hash=row.scope_hash,
            params=row.params,
            point_count=row.point_count,
            items=[IndexedItem(kind, name) for kind, name in zip(kinds, names, strict=True)],
            coords=blob_to_coords(row.coords, row.point_count),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def set_projection(
        self,
        user_id: str,
        model_id: str,
        scope_hash: str,
        params: str,
        items: list[IndexedItem],
        coords: np.ndarray,
    ) -> None:
        if len(items) != coords.shape[0]:
            raise ValueError(f"Got {len(items)} items but {coords.shape[0]} coordinate rows")
        stored = self._queries.image_index.set_projection(
            user_id=user_id,
            model_id=model_id,
            scope_hash=scope_hash,
            params=params,
            point_count=len(items),
            image_names=json.dumps([item.name for item in items]),
            item_kinds=json.dumps([item.kind for item in items]),
            coords=coords_to_blob(coords),
        )
        # There is nobody left to serve the projection of an account deleted while it was computed.
        if not stored:
            self._logger.debug(f"Skipped projection for missing user {user_id}")

    def delete_projection(self, user_id: str, model_id: str) -> None:
        self._queries.image_index.delete_projection(user_id, model_id)
