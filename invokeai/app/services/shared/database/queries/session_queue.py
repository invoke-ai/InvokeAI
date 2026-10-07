"""The session queue: the items users enqueued, their workflow-call chains, and the counts the queue's views show.

An item is read as a dict of its row (every column of `session_queue`) with its owner's display name and email, the
form `SessionQueueItem.queue_item_from_dict()` and `SessionQueueItemSummary.queue_item_summary_from_dict()` take.
An origin prefix matches as SQLite's `LIKE` always has, ignoring case; its `%` and `_` are taken as written.
"""

import functools
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    and_,
    bindparam,
    func,
    literal,
    or_,
    select,
)

from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    InBoundSet,
    bound_set,
    fixed_limit,
    like_prefix,
)
from invokeai.app.services.shared.database.queries.base import QueryModule, mapped, read
from invokeai.app.services.shared.database.schema.session_queue import session_queue, session_queue_enqueue_receipts
from invokeai.app.services.shared.database.schema.users import users

_Q = session_queue.c
_R = session_queue_enqueue_receipts.c

_ITEM = (*session_queue.c, users.c.display_name.label("user_display_name"), users.c.email.label("user_email"))
_ITEM_NAMES = tuple(column.name for column in _ITEM)
_ITEMS = session_queue.outerjoin(users, users.c.user_id == _Q.user_id)

_SUMMARY = (
    _Q.item_id,
    _Q.created_at,
    _Q.status,
    _Q.device,
    _Q.started_at,
    _Q.completed_at,
    _Q.origin,
    _Q.destination,
    _Q.batch_id,
    _Q.user_id,
    users.c.display_name.label("user_display_name"),
    users.c.email.label("user_email"),
    _Q.field_values,
    _Q.parent_item_id,
)
_SUMMARY_NAMES = tuple(column.name for column in _SUMMARY)

_IN_QUEUE = _Q.queue_id == bindparam("queue_id")
_PENDING = _Q.status == literal("pending")
_IN_PROGRESS = _Q.status == literal("in_progress")

_RECEIPT = select(_R.batch_id, _R.requested, _R.enqueued, _R.item_ids).where(
    _R.queue_id == bindparam("queue_id"),
    _R.user_id == bindparam("user_id"),
    _R.idempotency_key == bindparam("idempotency_key"),
)
_ITEM_BY_ID = select(*_ITEM).select_from(_ITEMS).where(_Q.item_id == bindparam("item_id"))
_WORKFLOW = select(_Q.workflow).where(_Q.item_id == bindparam("item_id"))
_PARENT = select(_Q.parent_item_id).where(_Q.item_id == bindparam("item_id"))
_CHILDREN = select(_Q.item_id).where(_Q.parent_item_id == bindparam("item_id")).order_by(_Q.item_id)
_IN_PROGRESS_IDS = select(_Q.item_id).where(_IN_QUEUE, _IN_PROGRESS).order_by(_Q.item_id)
_PARENT_ROW = session_queue.alias("parent")
# The child that would be dequeued next while its parent waits: its chain is still running between two children.
_NEXT_WAITING_CHILD = (
    select(_Q.item_id)
    .select_from(session_queue.join(_PARENT_ROW, _PARENT_ROW.c.item_id == _Q.parent_item_id))
    .where(_IN_QUEUE, _PENDING, _PARENT_ROW.c.status == literal("waiting"))
    .order_by(_Q.priority.desc(), _Q.created_at, _Q.item_id)
    .limit(fixed_limit(1))
)
_COUNT = select(func.count()).where(_IN_QUEUE)
_SUMMARIES = select(*_SUMMARY).select_from(_ITEMS).where(_IN_QUEUE, InBoundSet(_Q.item_id, bindparam("item_ids")))


def _origin_matches() -> ColumnElement[bool]:
    return CaseInsensitiveLike(_Q.origin, bindparam("origin_pattern"))


def _origin_pattern(origin_prefix: Optional[str]) -> Optional[str]:
    return like_prefix(origin_prefix) if origin_prefix is not None else None


@functools.cache
def _item_by_status(status: str, by_origin: bool) -> Select[Any]:
    """The pending item to run next, or an item in progress (the earliest enqueued of several)."""
    conditions = [_IN_QUEUE, _Q.status == literal(status)]
    if by_origin:
        conditions.append(_origin_matches())
    ordering = [_Q.priority.desc(), _Q.created_at, _Q.item_id] if status == "pending" else [_Q.item_id]
    return select(*_ITEM).select_from(_ITEMS).where(*conditions).order_by(*ordering).limit(fixed_limit(1))


@functools.cache
def _all_items(by_destination: bool) -> Select[Any]:
    conditions = [_IN_QUEUE]
    if by_destination:
        conditions.append(_Q.destination == bindparam("destination"))
    return select(*_ITEM).select_from(_ITEMS).where(*conditions).order_by(_Q.priority.desc(), _Q.item_id)


@functools.cache
def _page(by_status: bool, by_destination: bool, after: bool) -> Select[Any]:
    """A page of the queue, highest priority first, then in enqueue order; with `after`, the items after an item."""
    conditions = [_IN_QUEUE]
    if by_status:
        conditions.append(_Q.status == bindparam("status"))
    if by_destination:
        conditions.append(_Q.destination == bindparam("destination"))
    if after:
        after_priority: ColumnElement[int] = bindparam("after_priority")
        conditions.append(
            or_(
                _Q.priority < after_priority,
                and_(_Q.priority == after_priority, _Q.item_id > bindparam("after_item_id")),
            )
        )
    return (
        select(*_ITEM)
        .select_from(_ITEMS)
        .where(*conditions)
        .order_by(_Q.priority.desc(), _Q.item_id)
        .limit(bindparam("limit"))
    )


@functools.cache
def _item_ids(descending: bool, by_user: bool, by_origin: bool) -> Select[Any]:
    conditions = [_IN_QUEUE]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    if by_origin:
        conditions.append(_origin_matches())
    # The item id breaks ties, in enqueue order either way, as SQLite returned them before: a batch's items share
    # their creation time to the millisecond, and the queue's views show a batch in that order.
    ordering = [_Q.created_at.desc() if descending else _Q.created_at, _Q.item_id]
    return select(_Q.item_id).where(*conditions).order_by(*ordering)


@functools.cache
def _status_counts(by_user: bool, by_origin: bool) -> Select[Any]:
    """Every status's count in the queue, or in the account's part of it.

    Two statements rather than one counting both: `user_id` comes after the session in the row, which SQLite reads only
    by walking the session's overflow pages, while the account's part is found through an index on `user_id`.
    """
    conditions = [_IN_QUEUE]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    if by_origin:
        conditions.append(_origin_matches())
    return select(_Q.status, func.count()).where(*conditions).group_by(_Q.status)


@functools.cache
def _current(by_origin: bool) -> Select[Any]:
    """The identifiers of the item in progress, the earliest enqueued of several."""
    conditions = [_IN_QUEUE, _IN_PROGRESS]
    if by_origin:
        conditions.append(_origin_matches())
    return (
        select(_Q.item_id, _Q.session_id, _Q.batch_id, _Q.user_id)
        .where(*conditions)
        .order_by(_Q.item_id)
        .limit(fixed_limit(1))
    )


@functools.cache
def _batch_counts(by_user: bool) -> Select[Any]:
    conditions = [_IN_QUEUE, _Q.batch_id == bindparam("batch_id")]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    # A batch's items share their origin and destination: its children and retries copy them.
    return (
        select(_Q.status, func.count(), func.min(_Q.origin), func.min(_Q.destination))
        .where(*conditions)
        .group_by(_Q.status)
    )


@functools.cache
def _destination_counts(by_user: bool) -> Select[Any]:
    conditions = [_IN_QUEUE, _Q.destination == bindparam("destination")]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    return select(_Q.status, func.count()).where(*conditions).group_by(_Q.status)


def _item(row: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(_ITEM_NAMES, row, strict=True))


def _item_or_none(row: Optional[Sequence[Any]]) -> Optional[dict[str, Any]]:
    return _item(row) if row is not None else None


def _items(rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    return [_item(row) for row in rows]


def _summaries(rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(_SUMMARY_NAMES, row, strict=True)) for row in rows]


def _counts(rows: Sequence[Sequence[Any]]) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in rows}


# The walks below stop at an item they have seen: the app never links a cycle (a parent is always enqueued before its
# children), but a corrupt one must not hold the transaction, and on SQLite every other thread, forever.


def _descendants(conn: Connection, item_id: int) -> list[int]:
    """Every descendant of the item, breadth first."""
    descendants: list[int] = []
    seen = {item_id}
    frontier = [item_id]
    while frontier:
        children: Sequence[int] = conn.execute(_CHILDREN, {"item_id": frontier.pop(0)}).scalars().all()
        new = [child for child in children if child not in seen]
        seen.update(new)
        descendants.extend(new)
        frontier.extend(new)
    return descendants


def _chain(conn: Connection, item_id: int, *, reachable: bool = False) -> Optional[list[int]]:
    """The item's ancestors from its parent up, the item, and every descendant of the root, breadth first. None if the
    item does not exist, or an ancestor does not and not `reachable`; with `reachable`, the highest ancestor that
    exists stands for the root."""
    ancestors: list[int] = []
    current = item_id
    while True:
        row = conn.execute(_PARENT, {"item_id": current}).first()
        if row is None:
            if current == item_id or not reachable:
                return None
            ancestors.pop()
            break
        parent = row[0]
        if parent is None or parent == item_id or parent in ancestors:
            break
        ancestors.append(int(parent))
        current = int(parent)
    root = ancestors[-1] if ancestors else item_id
    return list(dict.fromkeys([*ancestors, item_id, *_descendants(conn, root)]))


class EnqueueReceipt(NamedTuple):
    batch_id: str
    requested: int
    enqueued: int
    # The enqueued items' ids, as a JSON array.
    item_ids: str


class CurrentItem(NamedTuple):
    item_id: int
    session_id: str
    batch_id: str
    user_id: str


class StatusCounts(NamedTuple):
    """A queue's counts by status, the account's own, and the item in progress, read in one snapshot."""

    counts: dict[str, int]
    own_counts: dict[str, int]
    current: Optional[CurrentItem]


class BatchCounts(NamedTuple):
    counts: dict[str, int]
    origin: Optional[str]
    destination: Optional[str]


class SessionQueueQueries(QueryModule):
    @read
    def enqueue_receipt(
        self, conn: Connection, queue_id: str, user_id: str, idempotency_key: str
    ) -> Optional[EnqueueReceipt]:
        parameters = {"queue_id": queue_id, "user_id": user_id, "idempotency_key": idempotency_key}
        row = conn.execute(_RECEIPT, parameters).first()
        return EnqueueReceipt(row[0], row[1], row[2], row[3]) if row is not None else None

    @mapped(_item_or_none)
    @read
    def item(self, conn: Connection, item_id: int) -> Optional[Row[Any]]:
        return conn.execute(_ITEM_BY_ID, {"item_id": item_id}).first()

    @mapped(_item_or_none)
    @read
    def item_by_status(
        self, conn: Connection, queue_id: str, status: str, origin_prefix: Optional[str]
    ) -> Optional[Row[Any]]:
        """The `pending` item that runs next (highest priority, then oldest), or the `in_progress` one first
        claimed."""
        statement = _item_by_status(status, origin_prefix is not None)
        parameters = {"queue_id": queue_id, "origin_pattern": _origin_pattern(origin_prefix)}
        return conn.execute(statement, parameters).first()

    @read
    def workflow(self, conn: Connection, item_id: int) -> Optional[str]:
        """The item's workflow as stored: JSON text, or None also for an item that does not exist."""
        return conn.execute(_WORKFLOW, {"item_id": item_id}).scalar()

    @mapped(_items)
    @read
    def all_items(self, conn: Connection, queue_id: str, destination: Optional[str]) -> Sequence[Row[Any]]:
        """Every item of the queue, or of its destination, highest priority first, then in enqueue order."""
        parameters = {"queue_id": queue_id, "destination": destination}
        return conn.execute(_all_items(destination is not None), parameters).all()

    @read
    def count(self, conn: Connection, queue_id: str) -> int:
        """How many items the queue holds, of every status."""
        return int(conn.execute(_COUNT, {"queue_id": queue_id}).scalar_one())

    @mapped(_items)
    @read
    def page(
        self,
        conn: Connection,
        queue_id: str,
        *,
        limit: int,
        after: Optional[tuple[int, int]],
        status: Optional[str],
        destination: Optional[str],
    ) -> Sequence[Row[Any]]:
        """Up to `limit` items, highest priority first, then in enqueue order; with `after` (a priority and an item
        id), those that come after that item. Every filter applies to every page."""
        statement = _page(status is not None, destination is not None, after is not None)
        parameters = {
            "queue_id": queue_id,
            "status": status,
            "destination": destination,
            "after_priority": after[0] if after is not None else None,
            "after_item_id": after[1] if after is not None else None,
            "limit": limit,
        }
        return conn.execute(statement, parameters).all()

    @read
    def item_ids(
        self,
        conn: Connection,
        queue_id: str,
        *,
        descending: bool,
        user_id: Optional[str],
        origin_prefix: Optional[str],
    ) -> list[int]:
        """The ids of the queue's items (the account's, with the origin, if given) by creation time."""
        statement = _item_ids(descending, user_id is not None, origin_prefix is not None)
        parameters = {"queue_id": queue_id, "user_id": user_id, "origin_pattern": _origin_pattern(origin_prefix)}
        return list(conn.execute(statement, parameters).scalars().all())

    @mapped(_summaries)
    @read
    def summaries(self, conn: Connection, queue_id: str, item_ids: Sequence[int]) -> Sequence[Row[Any]]:
        """The summary columns of the named items of the queue that exist, in no order."""
        return conn.execute(_SUMMARIES, {"queue_id": queue_id, "item_ids": bound_set(item_ids)}).all()

    @read
    def status_counts(
        self, conn: Connection, queue_id: str, user_id: Optional[str], origin_prefix: Optional[str]
    ) -> StatusCounts:
        by_origin = origin_prefix is not None
        parameters = {"queue_id": queue_id, "user_id": user_id, "origin_pattern": _origin_pattern(origin_prefix)}
        counts = conn.execute(_status_counts(False, by_origin), parameters).all()
        own_counts = conn.execute(_status_counts(True, by_origin), parameters).all() if user_id is not None else []
        current = conn.execute(_current(by_origin), parameters).first()
        return StatusCounts(
            counts=_counts(counts),
            own_counts=_counts(own_counts),
            current=CurrentItem(current[0], current[1], current[2], current[3]) if current is not None else None,
        )

    @read
    def batch_counts(self, conn: Connection, queue_id: str, batch_id: str, user_id: Optional[str]) -> BatchCounts:
        parameters = {"queue_id": queue_id, "batch_id": batch_id, "user_id": user_id}
        rows = conn.execute(_batch_counts(user_id is not None), parameters).all()
        return BatchCounts(
            counts=_counts(rows),
            origin=rows[0][2] if rows else None,
            destination=rows[0][3] if rows else None,
        )

    @mapped(_counts)
    @read
    def destination_counts(
        self, conn: Connection, queue_id: str, destination: str, user_id: Optional[str]
    ) -> Sequence[Row[Any]]:
        parameters = {"queue_id": queue_id, "destination": destination, "user_id": user_id}
        return conn.execute(_destination_counts(user_id is not None), parameters).all()

    @read
    def chain_item_ids(self, conn: Connection, item_id: int) -> Optional[list[int]]:
        """The item's workflow-call chain: its ancestors from its parent up, the item, and every descendant of the
        root, breadth first. None if the item, or an ancestor, does not exist."""
        return _chain(conn, item_id)

    @read
    def descendant_ids(self, conn: Connection, item_id: int) -> list[int]:
        """Every descendant of the item, breadth first."""
        return _descendants(conn, item_id)

    @read
    def current_chain_item_ids(self, conn: Connection, queue_id: str) -> set[int]:
        """The ids in the workflow-call chains of every item in progress, or, with none in progress, in the chain of
        the child that runs next while its parent waits."""
        in_progress: Sequence[int] = conn.execute(_IN_PROGRESS_IDS, {"queue_id": queue_id}).scalars().all()
        if not in_progress:
            waiting_child = conn.execute(_NEXT_WAITING_CHILD, {"queue_id": queue_id}).scalar()
            in_progress = [waiting_child] if waiting_child is not None else []
        chain: set[int] = set()
        for item_id in in_progress:
            # A running item whose parent row is gone still protects what of its chain remains.
            chain.update(_chain(conn, item_id, reachable=True) or [])
        return chain
