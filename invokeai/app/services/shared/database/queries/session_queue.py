"""The session queue: the items users enqueued, their workflow-call chains, and the counts the queue's views show.

An item is read as a dict of its row (every column of `session_queue`) with its owner's display name and email, the
form `SessionQueueItem.queue_item_from_dict()` and `SessionQueueItemSummary.queue_item_summary_from_dict()` take.
An origin prefix matches as SQLite's `LIKE` always has, ignoring case; its `%` and `_` are taken as written.
"""

import functools
import itertools
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    String,
    Update,
    and_,
    bindparam,
    case,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)

from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    ContainsText,
    InBoundSet,
    bound_set,
    fixed_limit,
    like_prefix,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, locking, mapped, read, write
from invokeai.app.services.shared.database.schema.session_queue import session_queue, session_queue_enqueue_receipts
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.types import now_text, timestamp_text

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
    return select(_Q.item_id).where(*conditions).order_by(*ordering).limit(fixed_limit(1))


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


def _row_of(conn: Connection, item_id: Optional[int]) -> Optional[Row[Any]]:
    return conn.execute(_ITEM_BY_ID, {"item_id": item_id}).first() if item_id is not None else None


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


# --- Writes ---------------------------------------------------------------------------------------------------------
#
# A status change is one conditional UPDATE: it applies only to an item that is not finished, and its row count says
# whether this call made it. The triggers of a migrated SQLite database also stamp `updated_at`, `started_at`,
# `completed_at` and `session_revision`; nothing does on a server, so these statements set them themselves.

_TERMINAL = ("completed", "failed", "canceled")
# Statuses of an item that has not finished; the intermediates cleanup protects their media.
ACTIVE_QUEUE_STATUSES = ("pending", "in_progress", "waiting")
_WAITING_WORK = ("pending", "waiting")
_NOT_FINISHED = _Q.status.not_in([literal(status) for status in _TERMINAL])
_NEXT_SEQUENCE = func.coalesce(_Q.status_sequence, 0) + 1
_NEXT_REVISION = _Q.session_revision + 1
_THE_ITEM = _Q.item_id == bindparam("target_item_id")

_STATUS_COLUMNS = (
    _Q.status,
    _Q.status_sequence,
    _Q.error_type,
    _Q.error_message,
    _Q.error_traceback,
    _Q.created_at,
    _Q.updated_at,
    _Q.started_at,
    _Q.completed_at,
    _Q.device,
)
_STATUS_NAMES = tuple(column.name for column in _STATUS_COLUMNS)
_STATUS_ROW = select(*_STATUS_COLUMNS).where(_Q.item_id == bindparam("item_id"))
_ITEM_EXISTS = select(literal(1)).where(_Q.item_id == bindparam("item_id"))
_LOCK_STATUS = select(_Q.status).where(_Q.item_id == bindparam("item_id")).with_for_update()
# Locks the item's own row only: on a server a FOR UPDATE over a join would lock the owner's account row as well.
_LOCK_ITEM = select(*session_queue.c).where(_Q.item_id == bindparam("item_id")).with_for_update()
_LOCK_ITEM_NAMES = tuple(column.name for column in session_queue.c)

_CLAIM = (
    update(session_queue)
    .where(_THE_ITEM, _PENDING)
    .values(
        status=literal("in_progress"),
        status_sequence=_NEXT_SEQUENCE,
        device=func.coalesce(bindparam("new_device", type_=String()), _Q.device),
        started_at=bindparam("now"),
    )
)

_RANKED_PENDING = (
    select(
        _Q.item_id,
        func.row_number().over(partition_by=_Q.user_id, order_by=(_Q.priority.desc(), _Q.item_id)).label("rank"),
    )
    .where(_PENDING)
    .cte("user_next_item")
)
_SERVED = session_queue.alias("served")
# The account served longest ago first: its latest start, a MAX over the account's started items (on SQLite an index
# seek; on a server a scan of the account's history in its index). Accounts never served sort first.
_LAST_SERVED = select(func.max(_SERVED.c.started_at)).where(_SERVED.c.user_id == _Q.user_id).scalar_subquery()
# The next item's id; its row is read by id. Sorting whole rows, sessions included, made MySQL copy every pending row
# into a temporary table on each dequeue.
_ROUND_ROBIN_NEXT = (
    select(_Q.item_id)
    .select_from(
        session_queue.join(_RANKED_PENDING, and_(_RANKED_PENDING.c.item_id == _Q.item_id, _RANKED_PENDING.c.rank == 1))
    )
    .order_by(func.coalesce(_LAST_SERVED, literal("1970-01-01")), _Q.item_id)
    .limit(fixed_limit(1))
)
_FIFO_NEXT = select(_Q.item_id).where(_PENDING).order_by(_Q.priority.desc(), _Q.item_id).limit(fixed_limit(1))

# How many resident model keys the affinity score takes: real caches hold a handful.
MAX_AFFINITY_MODEL_KEYS = 50


@functools.cache
def _affinity(slots: int) -> Select[Any]:
    """The account's pending item of the same priority within the lookahead whose session names the most resident
    models. Keys are bound in `slots` parameters, a power of two, the spare ones NULL, which score nothing."""
    score = sum(
        (case((ContainsText(_Q.session, bindparam(f"key_{i}", type_=String())), 1), else_=0) for i in range(slots)),
        start=literal(0),
    )
    return (
        select(*_ITEM, score.label("affinity"))
        .select_from(_ITEMS)
        .where(
            _PENDING,
            _Q.user_id.is_not_distinct_from(bindparam("user_id", type_=String())),
            _Q.priority == bindparam("priority"),
            _Q.item_id.between(bindparam("first_item_id"), bindparam("last_item_id")),
        )
        .order_by(score.desc(), _Q.item_id)
        .limit(fixed_limit(1))
    )


def _slots(count: int) -> int:
    return 1 << max(count - 1, 0).bit_length()


@functools.cache
def _transition(stamp: Optional[str]) -> Update:
    """Moves an unfinished item to a status, stamping `started_at` or `completed_at` as the status calls for."""
    values: dict[str, Any] = {
        "status": bindparam("new_status"),
        "status_sequence": _NEXT_SEQUENCE,
        "error_type": bindparam("new_error_type", type_=String()),
        "error_message": bindparam("new_error_message", type_=String()),
        "error_traceback": bindparam("new_error_traceback", type_=String()),
        "device": func.coalesce(bindparam("new_device", type_=String()), _Q.device),
    }
    if stamp is not None:
        values[stamp] = bindparam("now")
    return update(session_queue).where(_THE_ITEM, _NOT_FINISHED).values(values)


def _stamp(status: str) -> Optional[str]:
    if status == "in_progress":
        return "started_at"
    return "completed_at" if status in _TERMINAL else None


# By id lists rather than bound sets: MariaDB before 11.1 runs an UPDATE or DELETE with an IN subquery as a scan of
# the table that evaluates the subquery for every row.
_CANCEL_IDS = (
    update(session_queue)
    .where(_Q.item_id.in_(bindparam("item_ids", expanding=True)), _Q.status.in_(bindparam("statuses", expanding=True)))
    .values(status=literal("canceled"), status_sequence=_NEXT_SEQUENCE, completed_at=bindparam("now"))
)
_DELETE_IDS = delete(session_queue).where(_Q.item_id.in_(bindparam("item_ids", expanding=True)))
_INTERRUPTED = select(_Q.item_id).where(_Q.status.in_([literal("in_progress"), literal("waiting")]))

_SET_SESSION = (
    update(session_queue).where(_THE_ITEM).values(session=bindparam("new_session"), session_revision=_NEXT_REVISION)
)
_SET_ACTIVE_SESSION = _SET_SESSION.where(_NOT_FINISHED)
# Suspends the parent for the children it just enqueued, unless its session changed meanwhile or it finished.
_WAIT_FOR_CHILDREN = (
    update(session_queue)
    .where(_THE_ITEM, _NOT_FINISHED, _Q.session == bindparam("expected_session"))
    .values(
        session=bindparam("new_session"),
        session_revision=_NEXT_REVISION,
        status=literal("waiting"),
        status_sequence=_NEXT_SEQUENCE,
    )
)
_WORKFLOW_CALL_ITEMS = (
    select(_Q.item_id).where(_Q.workflow_call_id == bindparam("workflow_call_id")).order_by(_Q.item_id)
)

_INSERT_ITEM = insert(session_queue)
_PENDING_COUNT = select(func.count()).where(_IN_QUEUE, _PENDING)
_TOP_PENDING_PRIORITY = select(func.max(_Q.priority)).where(_IN_QUEUE, _PENDING)
_BATCH_TAKEN = (
    select(literal(1))
    .where(_IN_QUEUE, _Q.user_id == bindparam("user_id"), _Q.batch_id == bindparam("batch_id"))
    .limit(fixed_limit(1))
)
_BATCH_ITEM_IDS = (
    select(_Q.item_id)
    .where(_IN_QUEUE, _Q.user_id == bindparam("user_id"), _Q.batch_id == bindparam("batch_id"))
    .order_by(_Q.item_id)
)

_THE_RECEIPT = and_(
    _R.queue_id == bindparam("queue_id"),
    _R.user_id == bindparam("user_id"),
    _R.idempotency_key == bindparam("idempotency_key"),
)
_SETTLED_RECEIPT = select(_R.payload_hash, _R.batch_id, _R.requested, _R.enqueued, _R.priority, _R.item_ids).where(
    _THE_RECEIPT
)
_ACKNOWLEDGE = (
    update(session_queue_enqueue_receipts)
    .where(
        _R.queue_id == bindparam("target_queue_id"),
        _R.user_id == bindparam("target_user_id"),
        _R.idempotency_key == bindparam("target_idempotency_key"),
    )
    .values(acknowledged_at=bindparam("now"))
)
_DELETE_ACKNOWLEDGED = delete(session_queue_enqueue_receipts).where(
    _R.user_id == bindparam("user_id"), _R.acknowledged_at.is_not(None), _R.acknowledged_at <= bindparam("before")
)
_UNACKNOWLEDGED = _R.acknowledged_at.is_(None)
_RECEIPT_USAGE = select(
    func.count(),
    func.coalesce(func.sum(_R.byte_size), 0),
    func.coalesce(func.sum(case((_UNACKNOWLEDGED, 1), else_=0)), 0),
    func.coalesce(func.sum(case((_UNACKNOWLEDGED, _R.byte_size), else_=0)), 0),
).where(_R.user_id == bindparam("user_id"))
_INSERT_RECEIPT = insert(session_queue_enqueue_receipts)

_ACTIVE_ITEM_IDS = select(_Q.item_id).where(_Q.status.in_([literal(status) for status in ACTIVE_QUEUE_STATUSES]))
# History: finished items that no active workflow call still needs. A completed child is recovery state for its active
# root until the root ends, and the intermediates cleanup protects the root's media through it.
_PRUNABLE = and_(
    _Q.status.in_([literal(status) for status in _TERMINAL]),
    or_(_Q.root_item_id.is_(None), _Q.root_item_id.not_in(_ACTIVE_ITEM_IDS)),
)
_LATEST_FIRST = (func.coalesce(_Q.completed_at, _Q.updated_at, _Q.created_at).desc(), _Q.item_id.desc())


class Scope(NamedTuple):
    """The items a bulk cancel or delete applies to: the queue's, narrowed by every filter given."""

    queue_id: str
    user_id: Optional[str] = None
    origin_prefix: Optional[str] = None
    destination: Optional[str] = None
    batch_ids: Optional[Sequence[str]] = None
    # Leave out the workflow-call chains of the items running when the scope is changed.
    except_current: bool = False

    def parameters(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "user_id": self.user_id,
            "origin_pattern": _origin_pattern(self.origin_prefix),
            "destination": self.destination,
            "batch_ids": sorted(set(self.batch_ids)) if self.batch_ids is not None else None,
        }


@functools.cache
def _scoped(
    by_user: bool,
    by_origin: bool,
    by_destination: bool,
    by_batch: bool,
    statuses: tuple[str, ...],
    locked: bool,
) -> Select[Any]:
    """The (item id, owner) of the scope's items of these statuses (all, for none), locked for a change if `locked`;
    unlocked, only their ids. (`user_id` and `status` follow the session in the row, which SQLite reaches through
    the session's overflow pages: a read of ids alone lets its status index answer the status.)"""
    conditions: list[ColumnElement[bool]] = [_IN_QUEUE]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    if by_origin:
        conditions.append(_origin_matches())
    if by_destination:
        conditions.append(_Q.destination == bindparam("destination"))
    if by_batch:
        # An IN list rather than a bound set: callers name a batch or a few, which the batch index finds.
        conditions.append(_Q.batch_id.in_(bindparam("batch_ids", expanding=True)))
    if statuses:
        conditions.append(_Q.status.in_([literal(status) for status in statuses]))
    if not locked:
        return select(_Q.item_id).where(*conditions).order_by(_Q.item_id)
    return select(_Q.item_id, _Q.user_id).where(*conditions).order_by(_Q.item_id).with_for_update()


def _scope_shape(scope: Scope) -> tuple[bool, bool, bool, bool]:
    return (
        scope.user_id is not None,
        scope.origin_prefix is not None,
        scope.destination is not None,
        scope.batch_ids is not None,
    )


def _locked_scope(conn: Connection, scope: Scope, statuses: tuple[str, ...]) -> Sequence[Sequence[Any]]:
    """The (item id, owner) of the scope's items of these statuses, locked until the transaction ends.

    The running chains are read after the lock, and left out then: a claim of a locked item waits for this
    transaction, and a child enqueue in flight has committed when the lock is granted, so the chains read are those
    the change applies against. (Read before the lock, an item could start running, or a running item enqueue its
    children, between the read and the change.)"""
    rows = conn.execute(_scoped(*_scope_shape(scope), statuses, True), scope.parameters()).all()
    if scope.except_current and rows:
        running = _current_chain(conn, scope.queue_id)
        rows = [row for row in rows if int(row[0]) not in running]
    return rows


def _cancel(conn: Connection, item_ids: Sequence[int], statuses: Sequence[str]) -> int:
    """Cancels the items of these statuses, by id; how many."""
    now = now_text()
    canceled = 0
    for chunk in itertools.batched(item_ids, IN_CHUNK):
        parameters = {"item_ids": list(chunk), "statuses": list(statuses), "now": now}
        canceled += conn.execute(_CANCEL_IDS, parameters).rowcount
    return canceled


def _delete(conn: Connection, item_ids: Iterable[int]) -> int:
    """Deletes the items, by id; how many."""
    return sum(
        conn.execute(_DELETE_IDS, {"item_ids": list(chunk)}).rowcount for chunk in itertools.batched(item_ids, IN_CHUNK)
    )


@functools.cache
def _prunable(by_user: bool, latest: bool) -> Select[Any]:
    """The ids of the queue's history (the account's, if given); with `latest`, its `keep` latest items."""
    conditions = [_IN_QUEUE, _PRUNABLE]
    if by_user:
        conditions.append(_Q.user_id == bindparam("user_id"))
    statement = select(_Q.item_id).where(*conditions)
    return statement.order_by(*_LATEST_FIRST).limit(bindparam("keep")) if latest else statement


def _by_owner(rows: Sequence[Sequence[Any]]) -> dict[str, list[int]]:
    by_owner: dict[str, list[int]] = {}
    for item_id, owner in rows:
        by_owner.setdefault(owner, []).append(int(item_id))
    return by_owner


def _current_chain(conn: Connection, queue_id: str) -> set[int]:
    in_progress: Sequence[int] = conn.execute(_IN_PROGRESS_IDS, {"queue_id": queue_id}).scalars().all()
    if not in_progress:
        waiting_child = conn.execute(_NEXT_WAITING_CHILD, {"queue_id": queue_id}).scalar()
        in_progress = [waiting_child] if waiting_child is not None else []
    chain: set[int] = set()
    for item_id in in_progress:
        # A running item whose parent row is gone still protects what of its chain remains.
        chain.update(_chain(conn, item_id, reachable=True) or [])
    return chain


def _status_row(row: Optional[Sequence[Any]]) -> Optional[dict[str, Any]]:
    return dict(zip(_STATUS_NAMES, row, strict=True)) if row is not None else None


class Transition(NamedTuple):
    """What a status change found: whether the item exists, and its status columns if this call changed them."""

    exists: bool
    changed: Optional[dict[str, Any]]


class SettledReceipt(NamedTuple):
    payload_hash: str
    batch_id: str
    requested: int
    enqueued: int
    priority: int
    # The enqueued items' ids, as a JSON array.
    item_ids: str


class ReceiptUsage(NamedTuple):
    count: int
    bytes: int
    unacknowledged_count: int
    unacknowledged_bytes: int


ITEM_COLUMNS = (
    "queue_id",
    "session",
    "session_id",
    "batch_id",
    "field_values",
    "priority",
    "workflow",
    "origin",
    "destination",
    "retried_from_item_id",
    "user_id",
    "project_id",
)
"""The columns of the tuples `prepare_values_to_insert()` builds, in their order."""


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
        return _row_of(conn, conn.execute(statement, parameters).scalar())

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
        return _current_chain(conn, queue_id)

    # --- Dequeue ----------------------------------------------------------------------------------------------------

    @mapped(_item_or_none)
    @read
    def next_to_run(self, conn: Connection, *, round_robin: bool) -> Optional[Row[Any]]:
        """The pending item to run next: in round robin, the best of the account served longest ago; else the highest
        priority, then the oldest."""
        return _row_of(conn, conn.execute(_ROUND_ROBIN_NEXT if round_robin else _FIFO_NEXT).scalar())

    @read
    def warmest(
        self,
        conn: Connection,
        *,
        user_id: Optional[str],
        priority: int,
        after_item_id: int,
        lookahead: int,
        resident_keys: Sequence[str],
    ) -> Optional[dict[str, Any]]:
        """The account's pending item of this priority, from `after_item_id` to `lookahead` items after it, whose
        session names the most of the resident model keys; None when none names any."""
        keys = list(resident_keys)[:MAX_AFFINITY_MODEL_KEYS]
        if not keys:
            return None
        slots = _slots(len(keys))
        parameters: dict[str, Any] = {f"key_{i}": key for i, key in enumerate(keys)}
        parameters.update({f"key_{i}": None for i in range(len(keys), slots)})
        parameters.update(
            {
                "user_id": user_id,
                "priority": priority,
                "first_item_id": after_item_id,
                "last_item_id": after_item_id + lookahead,
            }
        )
        row = conn.execute(_affinity(slots), parameters).first()
        if row is None or not row[-1]:
            return None
        return _item(row[:-1])

    @mapped(_status_row)
    @write
    def claim(self, conn: Connection, item_id: int, device: Optional[str]) -> Optional[Row[Any]]:
        """Moves the item from pending to in progress for `device`; its status columns, or None if it is no longer
        pending (another worker claimed it, or it was canceled)."""
        parameters = {"target_item_id": item_id, "new_device": device, "now": now_text()}
        if conn.execute(_CLAIM, parameters).rowcount == 0:
            return None
        return conn.execute(_STATUS_ROW, {"item_id": item_id}).first()

    # --- Status changes ---------------------------------------------------------------------------------------------

    @write
    def transition(
        self,
        conn: Connection,
        item_id: int,
        status: str,
        *,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Transition:
        """Moves the item to `status` unless it is finished (completed, failed or canceled)."""
        parameters = {
            "target_item_id": item_id,
            "new_status": status,
            "new_error_type": error_type,
            "new_error_message": error_message,
            "new_error_traceback": error_traceback,
            "new_device": device,
            "now": now_text(),
        }
        if conn.execute(_transition(_stamp(status)), parameters).rowcount == 0:
            return Transition(exists=conn.execute(_ITEM_EXISTS, {"item_id": item_id}).first() is not None, changed=None)
        return Transition(exists=True, changed=_status_row(conn.execute(_STATUS_ROW, {"item_id": item_id}).first()))

    @write
    def cancel_interrupted(self, conn: Connection) -> None:
        """Cancels every item in progress or waiting, and the unfinished items of their workflow-call chains: none can
        resume after a restart."""
        chain: set[int] = set()
        for item_id in conn.execute(_INTERRUPTED).scalars().all():
            chain.update(_chain(conn, item_id, reachable=True) or [item_id])
        _cancel(conn, sorted(chain), ACTIVE_QUEUE_STATUSES)

    @read
    def in_progress(self, conn: Connection, scope: Scope) -> list[int]:
        """The ids of the scope's items in progress."""
        statement = _scoped(*_scope_shape(scope), ("in_progress",), False)
        return [int(row[0]) for row in conn.execute(statement, scope.parameters()).all()]

    @write
    def cancel_waiting_work(self, conn: Connection, scope: Scope) -> dict[str, list[int]]:
        """Cancels the scope's pending and waiting items; the ids it canceled, by owner."""
        rows = _locked_scope(conn, scope, _WAITING_WORK)
        _cancel(conn, [int(row[0]) for row in rows], _WAITING_WORK)
        return _by_owner(rows)

    @write
    def delete_scope(self, conn: Connection, scope: Scope, *, waiting_work_only: bool = False) -> dict[str, list[int]]:
        """Deletes the scope's items, or only its pending and waiting ones; the ids it deleted, by owner."""
        rows = _locked_scope(conn, scope, _WAITING_WORK if waiting_work_only else ())
        _delete(conn, (int(row[0]) for row in rows))
        return _by_owner(rows)

    @write
    def delete_items(self, conn: Connection, item_ids: Sequence[int]) -> None:
        _delete(conn, item_ids)

    @write
    def prune(self, conn: Connection, queue_id: str, *, user_id: Optional[str], keep: Optional[int] = None) -> int:
        """Deletes the queue's history (the account's, if given), all of it or all but its `keep` latest items; how
        many items it deleted."""
        parameters = {"queue_id": queue_id, "user_id": user_id, "keep": keep}
        prunable: set[int] = set(conn.execute(_prunable(user_id is not None, False), parameters).scalars().all())
        if keep is not None:
            prunable.difference_update(conn.execute(_prunable(user_id is not None, True), parameters).scalars().all())
        # Counted as deleted: a clear or prune running meanwhile may have deleted some of them first.
        return _delete(conn, sorted(prunable))

    # --- Sessions and workflow calls --------------------------------------------------------------------------------

    @write
    def set_session(self, conn: Connection, item_id: int, session: str, *, only_unfinished: bool) -> Optional[bool]:
        """Stores the item's session, with `only_unfinished` only while it is not finished. Whether it was stored; None
        if the item does not exist."""
        statement = _SET_ACTIVE_SESSION if only_unfinished else _SET_SESSION
        if conn.execute(statement, {"target_item_id": item_id, "new_session": session}).rowcount != 0:
            return True
        return False if conn.execute(_ITEM_EXISTS, {"item_id": item_id}).first() is not None else None

    @locking
    def lock_status(self, conn: Connection, item_id: int) -> Optional[str]:
        """The item's status, its row locked until the transaction ends; None if it does not exist."""
        return conn.execute(_LOCK_STATUS, {"item_id": item_id}).scalar()

    @locking
    def lock_item(self, conn: Connection, item_id: int) -> Optional[dict[str, Any]]:
        """The item's row (without its owner's name), locked until the transaction ends; None if it does not exist."""
        row = conn.execute(_LOCK_ITEM, {"item_id": item_id}).first()
        return dict(zip(_LOCK_ITEM_NAMES, row, strict=True)) if row is not None else None

    @write
    def insert_item(self, conn: Connection, values: dict[str, Any]) -> int:
        """Adds an item; its id."""
        now = now_text()
        inserted = conn.execute(_INSERT_ITEM, {"created_at": now, "updated_at": now, **values})
        return int(inserted.inserted_primary_key[0])

    @write
    def wait_for_children(self, conn: Connection, item_id: int, *, session: str, expected_session: str) -> bool:
        """Suspends the item with its new session for the children it enqueued; False if its session is no longer
        `expected_session` or it finished meanwhile."""
        parameters = {"target_item_id": item_id, "new_session": session, "expected_session": expected_session}
        return conn.execute(_WAIT_FOR_CHILDREN, parameters).rowcount != 0

    @read
    def workflow_call_item_ids(self, conn: Connection, workflow_call_id: str) -> list[int]:
        return [
            int(item_id)
            for item_id in conn.execute(_WORKFLOW_CALL_ITEMS, {"workflow_call_id": workflow_call_id}).scalars().all()
        ]

    # --- Enqueueing -------------------------------------------------------------------------------------------------

    @read
    def pending_count(self, conn: Connection, queue_id: str) -> int:
        return int(conn.execute(_PENDING_COUNT, {"queue_id": queue_id}).scalar_one())

    @read
    def top_pending_priority(self, conn: Connection, queue_id: str) -> int:
        """The highest priority of the queue's pending items; 0 for none."""
        return int(conn.execute(_TOP_PENDING_PRIORITY, {"queue_id": queue_id}).scalar() or 0)

    @read
    def batch_taken(self, conn: Connection, queue_id: str, user_id: str, batch_id: str) -> bool:
        parameters = {"queue_id": queue_id, "user_id": user_id, "batch_id": batch_id}
        return conn.execute(_BATCH_TAKEN, parameters).first() is not None

    @write
    def insert_items(self, conn: Connection, rows: Sequence[Sequence[Any]]) -> None:
        """Adds items from tuples of `ITEM_COLUMNS`, stamped with one time."""
        if rows:
            now = now_text()
            values = [
                {**dict(zip(ITEM_COLUMNS, row, strict=True)), "created_at": now, "updated_at": now} for row in rows
            ]
            conn.execute(_INSERT_ITEM, values)

    @read
    def batch_item_ids(self, conn: Connection, queue_id: str, user_id: str, batch_id: str) -> list[int]:
        parameters = {"queue_id": queue_id, "user_id": user_id, "batch_id": batch_id}
        return [int(item_id) for item_id in conn.execute(_BATCH_ITEM_IDS, parameters).scalars().all()]

    @read
    def settled_receipt(
        self, conn: Connection, queue_id: str, user_id: str, idempotency_key: str
    ) -> Optional[SettledReceipt]:
        parameters = {"queue_id": queue_id, "user_id": user_id, "idempotency_key": idempotency_key}
        row = conn.execute(_SETTLED_RECEIPT, parameters).first()
        return SettledReceipt(row[0], row[1], row[2], row[3], row[4], row[5]) if row is not None else None

    @write
    def delete_acknowledged_receipts(self, conn: Connection, user_id: str, older_than: timedelta) -> None:
        """Deletes the account's receipts acknowledged longer ago than `older_than`."""
        before = timestamp_text(datetime.now(timezone.utc) - older_than)
        conn.execute(_DELETE_ACKNOWLEDGED, {"user_id": user_id, "before": before})

    @read
    def receipt_usage(self, conn: Connection, user_id: str) -> ReceiptUsage:
        count, size, unacknowledged, unacknowledged_size = conn.execute(_RECEIPT_USAGE, {"user_id": user_id}).one()
        return ReceiptUsage(int(count), int(size), int(unacknowledged), int(unacknowledged_size))

    @write
    def insert_receipt(self, conn: Connection, values: dict[str, Any]) -> None:
        conn.execute(_INSERT_RECEIPT, {"created_at": now_text(), **values})

    @write
    def acknowledge_receipt(self, conn: Connection, queue_id: str, user_id: str, idempotency_key: str) -> None:
        parameters = {
            "target_queue_id": queue_id,
            "target_user_id": user_id,
            "target_idempotency_key": idempotency_key,
            "now": now_text(),
        }
        conn.execute(_ACKNOWLEDGE, parameters)
