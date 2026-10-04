import json
from dataclasses import dataclass

import numpy as np

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_index.image_index_common import (
    EMBEDDING_DTYPE,
    ImageIndexStatus,
    IndexedItem,
    MediaKind,
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
from invokeai.backend.util.logging import InvokeAILogger

# SQLite's default variable limit is 999; stay well under it when chunking IN clauses.
_IN_CLAUSE_CHUNK = 500


@dataclass(frozen=True)
class _Namespace:
    """The tables and columns one media kind is indexed through.

    Images and videos are indexed by the same rules over different tables, so the rules are
    written once and the SQL is built per namespace. Every field is a constant of this module;
    nothing here is ever derived from a request.
    """

    kind: MediaKind
    table: str
    name_column: str
    category_column: str
    embeddings_table: str
    board_table: str

    @property
    def eligible_conditions(self) -> str:
        """Conditions defining a "gallery" item, i.e. one worth indexing."""
        return f"{self.table}.is_intermediate = ? AND {self.table}.{self.category_column} = ?"


_IMAGES = _Namespace(
    kind="image",
    table="images",
    name_column="image_name",
    category_column="image_category",
    embeddings_table="image_embeddings",
    board_table="board_images",
)
_VIDEOS = _Namespace(
    kind="video",
    table="videos",
    name_column="video_name",
    category_column="video_category",
    embeddings_table="video_embeddings",
    board_table="board_videos",
)
# Images first, so every listing that concatenates namespaces has one stable order.
_NAMESPACES: tuple[_Namespace, ...] = (_IMAGES, _VIDEOS)
_NAMESPACE_BY_KIND: dict[MediaKind, _Namespace] = {namespace.kind: namespace for namespace in _NAMESPACES}


def _eligible_params() -> tuple[bool, str]:
    return (False, ImageCategory.GENERAL.value)


class ImageIndexRecordsSqlite(ImageIndexRecordsBase):
    """SQLite implementation of semantic image index storage."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._logger = InvokeAILogger.get_logger(self.__class__.__name__)

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def upsert_embedding(self, item: IndexedItem, model_id: str, embedding: np.ndarray) -> None:
        namespace = _NAMESPACE_BY_KIND[item.kind]
        blob = embedding_to_blob(embedding)
        with self._db.transaction() as cursor:
            # The item may be deleted between being scheduled and embedded, in which case the
            # embedding is pointless and this is a no-op. `WHERE EXISTS` rather than catching the
            # foreign-key IntegrityError: the guard and the insert are one atomic statement, and
            # no exception is raised on the expected race, so nothing can leave the transaction
            # in a state the caller did not ask for. An IntegrityError from here now means a
            # genuine bug and is allowed to propagate.
            cursor.execute(
                f"""--sql
                INSERT INTO {namespace.embeddings_table} ({namespace.name_column}, model_id, dim, embedding)
                SELECT ?, ?, ?, ?
                WHERE EXISTS (
                  SELECT 1 FROM {namespace.table} WHERE {namespace.table}.{namespace.name_column} = ?
                )
                ON CONFLICT ({namespace.name_column}, model_id)
                DO UPDATE SET dim = excluded.dim, embedding = excluded.embedding;
                """,
                (item.name, model_id, embedding.shape[0], blob, item.name),
            )
            if cursor.rowcount == 0:
                self._logger.debug(f"Skipped embedding for missing {item.kind} {item.name}")

    def get_embeddings(self, items: list[IndexedItem], model_id: str) -> tuple[list[IndexedItem], np.ndarray]:
        # Dedupe while preserving order so repeated input items cannot
        # double-count rows in downstream projection/similarity math.
        items = list(dict.fromkeys(items))
        rows: dict[IndexedItem, tuple[int, bytes]] = {}

        # Only the reads happen under the transaction. Deserialization and validation are done
        # afterwards so a malformed row raises outside it: `transaction()` rolls the shared
        # connection back on any exception, which would be an unpleasant side effect of what the
        # caller asked to be a read.
        with self._db.transaction() as cursor:
            for namespace in _NAMESPACES:
                names = [item.name for item in items if item.kind == namespace.kind]
                for start in range(0, len(names), _IN_CLAUSE_CHUNK):
                    chunk = names[start : start + _IN_CLAUSE_CHUNK]
                    placeholders = ",".join("?" * len(chunk))
                    cursor.execute(
                        f"""--sql
                        SELECT {namespace.name_column}, dim, embedding
                        FROM {namespace.embeddings_table}
                        WHERE model_id = ? AND {namespace.name_column} IN ({placeholders});
                        """,
                        (model_id, *chunk),
                    )
                    for name, dim, blob in cursor.fetchall():
                        rows[IndexedItem(namespace.kind, name)] = (dim, blob)

        # Read back in the caller's order: the returned matrix's rows align with it.
        found_items = [item for item in items if item in rows]

        dim: int | None = None
        vectors: list[np.ndarray] = []
        for item in found_items:
            row_dim, blob = rows[item]
            if dim is None:
                dim = row_dim
            elif row_dim != dim:
                raise ValueError(f"Inconsistent embedding dims for model {model_id}: found {row_dim} and {dim}")
            vectors.append(blob_to_embedding(blob, row_dim))

        if not vectors:
            return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)
        return found_items, np.stack(vectors)

    def delete_embedding(self, item: IndexedItem) -> None:
        namespace = _NAMESPACE_BY_KIND[item.kind]
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                DELETE FROM {namespace.embeddings_table} WHERE {namespace.name_column} = ?;
                """,
                (item.name,),
            )

    def delete_embeddings_for_other_models(self, model_id: str) -> int:
        deleted = 0
        with self._db.transaction() as cursor:
            for namespace in _NAMESPACES:
                cursor.execute(
                    f"""--sql
                    DELETE FROM {namespace.embeddings_table} WHERE model_id != ?;
                    """,
                    (model_id,),
                )
                deleted += cursor.rowcount
        return deleted

    def list_unembedded_items(self, model_id: str, limit: int) -> list[IndexedItem]:
        if limit < 0:
            # SQLite reads a negative LIMIT as unbounded, which would turn a backfill batch into
            # a full-table load. Fail on a miscomputed limit instead of silently doing that.
            raise ValueError(f"limit must be non-negative, got {limit}")

        # Each namespace contributes its own oldest `limit` rows, merged here so a batch is the
        # globally oldest work rather than every image before any video. Taking `limit` from each
        # is what makes the merge correct: whatever the merged prefix should contain is present
        # in one of the two lists.
        candidates: list[tuple[str, str, IndexedItem]] = []
        with self._db.transaction() as cursor:
            for namespace in _NAMESPACES:
                cursor.execute(
                    f"""--sql
                    SELECT {namespace.table}.{namespace.name_column}, {namespace.table}.created_at
                    FROM {namespace.table}
                    LEFT JOIN {namespace.embeddings_table}
                      ON {namespace.embeddings_table}.{namespace.name_column}
                         = {namespace.table}.{namespace.name_column}
                      AND {namespace.embeddings_table}.model_id = ?
                    WHERE {namespace.eligible_conditions}
                      AND {namespace.embeddings_table}.{namespace.name_column} IS NULL
                    ORDER BY {namespace.table}.created_at ASC, {namespace.table}.{namespace.name_column} ASC
                    LIMIT ?;
                    """,
                    (model_id, *_eligible_params(), limit),
                )
                candidates.extend(
                    (created_at, name, IndexedItem(namespace.kind, name)) for name, created_at in cursor.fetchall()
                )

        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        return [item for _, _, item in candidates[:limit]]

    def count_index_status(self, model_id: str) -> ImageIndexStatus:
        total = 0
        embedded = 0
        with self._db.transaction() as cursor:
            for namespace in _NAMESPACES:
                cursor.execute(
                    f"""--sql
                    SELECT
                      COUNT(*),
                      COUNT({namespace.embeddings_table}.{namespace.name_column})
                    FROM {namespace.table}
                    LEFT JOIN {namespace.embeddings_table}
                      ON {namespace.embeddings_table}.{namespace.name_column}
                         = {namespace.table}.{namespace.name_column}
                      AND {namespace.embeddings_table}.model_id = ?
                    WHERE {namespace.eligible_conditions};
                    """,
                    (model_id, *_eligible_params()),
                )
                namespace_total, namespace_embedded = cursor.fetchone()
                total += namespace_total
                embedded += namespace_embedded
        return ImageIndexStatus(total=total, embedded=embedded)

    def list_accessible_embedded_items(self, user_id: str | None, model_id: str) -> list[IndexedItem]:
        # Both clauses mirror the gallery "all" listing semantics
        # (image_records_sqlite / video_records_sqlite): items on archived
        # boards are hidden from every scope, and a scoped user sees their own
        # unboarded items plus items on active boards they own, that are
        # shared/public, or that were individually shared with them via
        # shared_boards. The projection scope_hash and semantic search both
        # derive from this listing, so any change here must keep matching the
        # gallery's access model.
        items: list[IndexedItem] = []
        with self._db.transaction() as cursor:
            for namespace in _NAMESPACES:
                params: list[object] = [model_id, *_eligible_params()]
                if user_id is None:
                    # Administrative scope: everything except items on archived boards.
                    access_clause = f"""AND (
                            {namespace.board_table}.board_id IS NULL
                            OR EXISTS (
                              SELECT 1 FROM boards
                              WHERE boards.board_id = {namespace.board_table}.board_id
                                AND boards.archived = 0
                            )
                          )"""
                else:
                    access_clause = f"""AND (
                            ({namespace.board_table}.board_id IS NULL AND {namespace.table}.user_id = ?)
                            OR EXISTS (
                              SELECT 1 FROM boards
                              WHERE boards.board_id = {namespace.board_table}.board_id
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
                    params.extend(
                        [user_id, user_id, BoardVisibility.Shared.value, BoardVisibility.Public.value, user_id]
                    )

                cursor.execute(
                    f"""--sql
                    SELECT DISTINCT {namespace.embeddings_table}.{namespace.name_column}
                    FROM {namespace.embeddings_table}
                    JOIN {namespace.table}
                      ON {namespace.table}.{namespace.name_column}
                         = {namespace.embeddings_table}.{namespace.name_column}
                    LEFT JOIN {namespace.board_table}
                      ON {namespace.board_table}.{namespace.name_column} = {namespace.table}.{namespace.name_column}
                    WHERE {namespace.embeddings_table}.model_id = ?
                      AND {namespace.eligible_conditions}
                      {access_clause}
                    ORDER BY {namespace.embeddings_table}.{namespace.name_column} ASC;
                    """,
                    params,
                )
                items.extend(IndexedItem(namespace.kind, row[0]) for row in cursor.fetchall())
        return items

    def get_custom_vocab_terms(self) -> list[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT term FROM image_index_vocab_terms ORDER BY term ASC;
                """
            )
            return [row[0] for row in cursor.fetchall()]

    def set_custom_vocab_terms(self, terms: list[str]) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM image_index_vocab_terms;
                """
            )
            # OR IGNORE: callers normalize away duplicates, but the NOCASE
            # primary key would otherwise turn a case-variant pair that slipped
            # through into an IntegrityError that rolls back the whole replace.
            cursor.executemany(
                """--sql
                INSERT OR IGNORE INTO image_index_vocab_terms (term) VALUES (?);
                """,
                [(term,) for term in terms],
            )

    def get_projection(self, user_id: str, model_id: str) -> ProjectionRecord | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT scope_hash, params, point_count, image_names, item_kinds, coords, created_at, updated_at
                FROM image_projections
                WHERE user_id = ? AND model_id = ?;
                """,
                (user_id, model_id),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        names: list[str] = json.loads(row[3])
        # A projection cached before videos were indexable has no kinds: every name in it is an
        # image, which is what a NULL column means. Same reading for a row written by this code,
        # where the column is always present.
        kinds: list[MediaKind] = json.loads(row[4]) if row[4] is not None else ["image"] * len(names)
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
            scope_hash=row[0],
            params=row[1],
            point_count=row[2],
            items=[IndexedItem(kind, name) for kind, name in zip(kinds, names, strict=True)],
            coords=blob_to_coords(row[5], row[2]),
            created_at=row[6],
            updated_at=row[7],
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
        blob = coords_to_blob(coords)
        names = json.dumps([item.name for item in items])
        kinds = json.dumps([item.kind for item in items])
        with self._db.transaction() as cursor:
            # No-op if the user was deleted while their projection was being computed; there is
            # nobody left to serve it to. Guarded by `WHERE EXISTS` for the same reason as in
            # `upsert_embedding`.
            cursor.execute(
                """--sql
                INSERT INTO image_projections
                  (user_id, model_id, scope_hash, params, point_count, image_names, item_kinds, coords)
                SELECT ?, ?, ?, ?, ?, ?, ?, ?
                WHERE EXISTS (SELECT 1 FROM users WHERE users.user_id = ?)
                ON CONFLICT (user_id, model_id)
                DO UPDATE SET
                  scope_hash = excluded.scope_hash,
                  params = excluded.params,
                  point_count = excluded.point_count,
                  image_names = excluded.image_names,
                  item_kinds = excluded.item_kinds,
                  coords = excluded.coords;
                """,
                (user_id, model_id, scope_hash, params, len(items), names, kinds, blob, user_id),
            )
            if cursor.rowcount == 0:
                self._logger.debug(f"Skipped projection for missing user {user_id}")

    def delete_projection(self, user_id: str, model_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM image_projections WHERE user_id = ? AND model_id = ?;
                """,
                (user_id, model_id),
            )
