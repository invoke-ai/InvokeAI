"""The image index: embeddings of gallery images and videos under each embedding model, the accounts' cached map
projections, and the custom labeling vocabulary.

Only gallery items are indexed: images and videos in the general category that are not intermediates. Each kind has
its own embedding table, so that an embedding is deleted with its media by foreign key. Listings of both kinds give
the images first.
"""

import functools
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Delete,
    Row,
    Select,
    Table,
    and_,
    bindparam,
    delete,
    exists,
    false,
    func,
    literal,
    or_,
    select,
    union_all,
)

from invokeai.app.services.image_index.image_index_common import IndexedItem, MediaKind
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.dialect import InBoundSet, bound_set, insert_ignore, upsert
from invokeai.app.services.shared.database.queries.base import QueryModule, mapped, read, write
from invokeai.app.services.shared.database.queries.board_access import readable_board
from invokeai.app.services.shared.database.schema.boards import board_images, board_videos, boards
from invokeai.app.services.shared.database.schema.image_index import (
    image_embeddings,
    image_index_vocab_terms,
    image_projections,
    video_embeddings,
)
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.database.types import now_text


class _Kind(NamedTuple):
    media: Table
    embeddings: Table
    membership: Table
    name: str
    category: str


# Images first: every listing of both kinds has this one order.
_KINDS: dict[MediaKind, _Kind] = {
    "image": _Kind(images, image_embeddings, board_images, "image_name", "image_category"),
    "video": _Kind(videos, video_embeddings, board_videos, "video_name", "video_category"),
}

_P = image_projections.c
_THE_PROJECTION = and_(_P.user_id == bindparam("user_id"), _P.model_id == bindparam("model_id"))
_PROJECTION = select(
    _P.scope_hash, _P.params, _P.point_count, _P.image_names, _P.item_kinds, _P.coords, _P.created_at, _P.updated_at
).where(_THE_PROJECTION)
_DELETE_PROJECTION = delete(image_projections).where(_THE_PROJECTION)
# Shared, so that the account is not deleted before the projection written for it is.
_LOCK_USER = select(literal(1)).where(users.c.user_id == bindparam("user_id")).with_for_update(read=True)

_V = image_index_vocab_terms.c
_VOCAB_TERMS = select(_V.term).order_by(_V.term)
_CLEAR_VOCAB_TERMS = delete(image_index_vocab_terms)


def _eligible(kind: _Kind) -> list[ColumnElement[bool]]:
    """The conditions of a gallery item, the only kind worth indexing."""
    return [
        kind.media.c.is_intermediate == false(),
        kind.media.c[kind.category] == literal(ImageCategory.GENERAL.value),
    ]


def _with_embedding(kind: _Kind) -> Any:
    """The media with their embedding under the bound model, if they have one."""
    embeddings = kind.embeddings
    on = and_(embeddings.c[kind.name] == kind.media.c[kind.name], embeddings.c.model_id == bindparam("model_id"))
    return kind.media.outerjoin(embeddings, on)


@functools.cache
def _lock_media(kind: MediaKind) -> Select[Any]:
    # Shared, so that the media is not deleted before the embedding written for it is.
    media, name = _KINDS[kind].media, _KINDS[kind].name
    return select(literal(1)).where(media.c[name] == bindparam("name")).with_for_update(read=True)


@functools.cache
def _upsert_embedding(kind: MediaKind, dialect_name: str) -> Any:
    return upsert(dialect_name, _KINDS[kind].embeddings, update=["dim", "embedding", "encoding"])


def _named_embeddings(kind: MediaKind) -> Select[Any]:
    embeddings, name = _KINDS[kind].embeddings, _KINDS[kind].name
    return select(
        literal(kind), embeddings.c[name], embeddings.c.dim, embeddings.c.embedding, embeddings.c.encoding
    ).where(embeddings.c.model_id == bindparam("model_id"), InBoundSet(embeddings.c[name], bindparam(f"{kind}_names")))


# Both kinds in one statement, each kind's names in one parameter: an IN list would bind every name, in statements
# of IN_CHUNK.
_EMBEDDINGS = union_all(*(_named_embeddings(kind) for kind in _KINDS))


@functools.cache
def _delete_embedding(kind: MediaKind) -> Delete:
    embeddings, name = _KINDS[kind].embeddings, _KINDS[kind].name
    return delete(embeddings).where(embeddings.c[name] == bindparam("name"))


@functools.cache
def _delete_other_models(kind: MediaKind) -> Delete:
    embeddings = _KINDS[kind].embeddings
    return delete(embeddings).where(embeddings.c.model_id != bindparam("model_id"))


@functools.cache
def _unembedded(kind: MediaKind) -> Select[Any]:
    """The kind's oldest eligible items without an embedding under the model."""
    the_kind = _KINDS[kind]
    media = the_kind.media
    return (
        select(media.c[the_kind.name], media.c.created_at)
        .select_from(_with_embedding(the_kind))
        .where(*_eligible(the_kind), the_kind.embeddings.c[the_kind.name].is_(None))
        .order_by(media.c.created_at, media.c[the_kind.name])
        .limit(bindparam("limit"))
    )


@functools.cache
def _status(kind: MediaKind) -> Select[Any]:
    """The kind's eligible items, and how many of them have an embedding under the model."""
    the_kind = _KINDS[kind]
    return (
        select(func.count(), func.count(the_kind.embeddings.c[the_kind.name]))
        .select_from(_with_embedding(the_kind))
        .where(*_eligible(the_kind))
    )


@functools.cache
def _accessible(kind: MediaKind, scoped: bool) -> Select[Any]:
    """The kind's embedded items an account can list, by name: as the gallery's listing of every image, items on
    archived boards are hidden from every scope, and `scoped` lists only the account's own items on no board and those
    on boards it may read."""
    the_kind = _KINDS[kind]
    media, embeddings, membership, name = the_kind.media, the_kind.embeddings, the_kind.membership, the_kind.name
    listed_board = [boards.c.archived == false()]
    if scoped:
        listed_board.append(readable_board(bindparam("user_id")))
    on_listed_board = exists(select(literal(1)).where(boards.c.board_id == membership.c.board_id, *listed_board))
    on_no_board = membership.c.board_id.is_(None)
    if scoped:
        on_no_board = and_(on_no_board, media.c.user_id == bindparam("user_id"))
    source = embeddings.join(media, media.c[name] == embeddings.c[name]).outerjoin(
        membership, membership.c[name] == media.c[name]
    )
    # Each item once without DISTINCT: an item is on one board at most, and has one embedding under the model.
    return (
        select(embeddings.c[name])
        .select_from(source)
        .where(embeddings.c.model_id == bindparam("model_id"), *_eligible(the_kind), or_(on_no_board, on_listed_board))
        .order_by(embeddings.c[name])
    )


@functools.cache
def _set_projection(dialect_name: str) -> Any:
    replaced = ["scope_hash", "params", "point_count", "image_names", "item_kinds", "coords", "updated_at"]
    return upsert(dialect_name, image_projections, update=replaced)


@functools.cache
def _insert_vocab_term(dialect_name: str) -> Any:
    # Callers pass distinct terms, but the case-insensitive key would make a pair differing in case fail the replace.
    return insert_ignore(dialect_name, image_index_vocab_terms)


class ProjectionRow(NamedTuple):
    """A cached projection as stored: its item names and kinds as JSON arrays, its coordinates as float32 bytes."""

    scope_hash: str
    params: str
    point_count: int
    image_names: str
    # None for a projection cached before videos were indexed, all of whose items are images.
    item_kinds: Optional[str]
    coords: bytes
    created_at: str
    updated_at: str


def _found_embeddings(rows: Sequence[Sequence[Any]]) -> dict[IndexedItem, tuple[int, bytes, str]]:
    return {IndexedItem(kind, name): (dim, embedding, encoding) for kind, name, dim, embedding, encoding in rows}


def _items(names: list[tuple[MediaKind, Sequence[str]]]) -> list[IndexedItem]:
    return [IndexedItem(kind, name) for kind, kind_names in names for name in kind_names]


class ImageIndexQueries(QueryModule):
    @write
    def upsert_embedding(
        self, conn: Connection, item: IndexedItem, model_id: str, dim: int, embedding: bytes, encoding: str
    ) -> bool:
        """Stores the item's embedding, in the float type `encoding` names, under the model, replacing one stored
        before. Whether the item exists: an item deleted meanwhile gets none."""
        if conn.execute(_lock_media(item.kind), {"name": item.name}).first() is None:
            return False
        values = {
            _KINDS[item.kind].name: item.name,
            "model_id": model_id,
            "dim": dim,
            "embedding": embedding,
            "encoding": encoding,
        }
        conn.execute(_upsert_embedding(item.kind, conn.dialect.name), values)
        return True

    @mapped(_found_embeddings)
    @read
    def embeddings(self, conn: Connection, items: Sequence[IndexedItem], model_id: str) -> Sequence[Row[Any]]:
        """The dimension, bytes and encoding of each item's embedding under the model, for the items that have one."""
        if not items:
            return []
        parameters = {f"{kind}_names": bound_set(item.name for item in items if item.kind == kind) for kind in _KINDS}
        return conn.execute(_EMBEDDINGS, {"model_id": model_id, **parameters}).all()

    @write
    def delete_embedding(self, conn: Connection, item: IndexedItem) -> None:
        """Deletes the item's embeddings under every model."""
        conn.execute(_delete_embedding(item.kind), {"name": item.name})

    @write
    def delete_embeddings_for_other_models(self, conn: Connection, model_id: str) -> int:
        """Deletes the embeddings of every model but this one; how many."""
        return sum(conn.execute(_delete_other_models(kind), {"model_id": model_id}).rowcount for kind in _KINDS)

    @read
    def unembedded(self, conn: Connection, model_id: str, limit: int) -> list[IndexedItem]:
        """The `limit` oldest eligible items of both kinds without an embedding under the model, oldest first."""
        candidates: list[tuple[str, str, IndexedItem]] = []
        for kind in _KINDS:
            rows = conn.execute(_unembedded(kind), {"model_id": model_id, "limit": limit}).all()
            candidates.extend((created_at, name, IndexedItem(kind, name)) for name, created_at in rows)
        # Each kind's own oldest `limit` hold the oldest `limit` of both. The sort is stable: images first on a tie.
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        return [item for _, _, item in candidates[:limit]]

    @read
    def status(self, conn: Connection, model_id: str) -> tuple[int, int]:
        """The eligible items of both kinds, and how many of them have an embedding under the model."""
        total = embedded = 0
        for kind in _KINDS:
            kind_total, kind_embedded = conn.execute(_status(kind), {"model_id": model_id}).one()
            total += kind_total
            embedded += kind_embedded
        return total, embedded

    @mapped(_items)
    @read
    def accessible_embedded(
        self, conn: Connection, user_id: Optional[str], model_id: str
    ) -> list[tuple[MediaKind, Sequence[str]]]:
        """The embedded items the account can list, or with None every item not on an archived board: images by name,
        then videos by name."""
        parameters = {"model_id": model_id, "user_id": user_id}
        return [
            (kind, conn.execute(_accessible(kind, user_id is not None), parameters).scalars().all()) for kind in _KINDS
        ]

    @read
    def vocab_terms(self, conn: Connection) -> list[str]:
        """The custom vocabulary, in the case-insensitive order of its terms."""
        return list(conn.execute(_VOCAB_TERMS).scalars().all())

    @write
    def replace_vocab_terms(self, conn: Connection, terms: Sequence[str]) -> None:
        """Replaces the custom vocabulary with these terms; of terms differing only in case, the first is kept."""
        conn.execute(_CLEAR_VOCAB_TERMS)
        if terms:
            # Stamped once: the column's default would take the time for every term.
            now = now_text()
            rows = [{"term": term, "created_at": now} for term in terms]
            conn.execute(_insert_vocab_term(conn.dialect.name), rows)

    @read
    def projection(self, conn: Connection, user_id: str, model_id: str) -> Optional[ProjectionRow]:
        row = conn.execute(_PROJECTION, {"user_id": user_id, "model_id": model_id}).first()
        return ProjectionRow(*row) if row is not None else None

    @write
    def set_projection(
        self,
        conn: Connection,
        *,
        user_id: str,
        model_id: str,
        scope_hash: str,
        params: str,
        point_count: int,
        image_names: str,
        item_kinds: str,
        coords: bytes,
    ) -> bool:
        """Stores the account's projection, replacing one stored before. Whether the account exists: one deleted
        meanwhile gets none."""
        if conn.execute(_LOCK_USER, {"user_id": user_id}).first() is None:
            return False
        now = now_text()
        values = {
            "user_id": user_id,
            "model_id": model_id,
            "scope_hash": scope_hash,
            "params": params,
            "point_count": point_count,
            "image_names": image_names,
            "item_kinds": item_kinds,
            "coords": coords,
            # One stamp for both: on a first insert the projection was created when it was last updated.
            "created_at": now,
            "updated_at": now,
        }
        conn.execute(_set_projection(conn.dialect.name), values)
        return True

    @write
    def delete_projection(self, conn: Connection, user_id: str, model_id: str) -> None:
        conn.execute(_DELETE_PROJECTION, {"user_id": user_id, "model_id": model_id})
