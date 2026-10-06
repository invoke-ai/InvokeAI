"""The workflow library: the accounts' own workflows (category `user`) and the bundled ones (`default`)."""

import functools
import itertools
from collections.abc import Collection, Sequence
from typing import Any, Literal, NamedTuple, Optional, Unpack

from sqlalchemy import (
    BindParameter,
    ColumnElement,
    Connection,
    Row,
    Select,
    Update,
    bindparam,
    case,
    delete,
    false,
    func,
    insert,
    literal,
    or_,
    select,
    true,
    update,
)

from invokeai.app.services.shared.database.dialect import CaseInsensitiveLike, like_contains
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, locking, mapped, read, write
from invokeai.app.services.shared.database.schema.workflows import workflow_library
from invokeai.app.services.shared.database.types import now_text
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowCategory,
    WorkflowRecordDTO,
    WorkflowRecordDTOBase,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
)


class LockedWorkflow(NamedTuple):
    """A workflow's row as a transaction locked it: what a write of the workflow decides on."""

    category: str
    user_id: Optional[str]
    revision: int
    # The document, when the lock was taken to rewrite it.
    document: Optional[str] = None


class _Shape(NamedTuple):
    """Which conditions a listing or a count has; their values are bound."""

    categories: tuple[WorkflowCategory, ...]
    # How many tags a workflow is looked for under, any of them.
    tags: int
    has_been_opened: Optional[bool]
    # Whether a text is looked for in the name, description and tags.
    searching: bool
    # Whether it covers only the account's own workflows and the bundled ones.
    scoped: bool
    is_public: Optional[bool]


# A client chooses how many tags it sends, and every number would be a statement of its own that SQLAlchemy keeps
# compiled. Up to this many, a statement has the next power of two of tag slots, the ones left over bound to NULL,
# which matches nothing; a listing for more tags builds and compiles its statement for the call alone, and counts take
# the tags this many at a time.
_TAG_SLOTS = 32


def _slots(count: int) -> int:
    """The tag slots of a statement for `count` tags: the next power of two up to `_TAG_SLOTS`, beyond it `count`."""
    return count if count == 0 or count > _TAG_SLOTS else 1 << (count - 1).bit_length()


def _flag(value: Optional[bool]) -> Optional[bool]:
    # Shapes are cache keys, and 1 == True: normalized, whatever a caller passes for a flag picks the right statement.
    return None if value is None else bool(value)


def _where(
    *,
    categories: Optional[Sequence[WorkflowCategory]] = None,
    tags: Optional[Sequence[str]] = None,
    has_been_opened: Optional[bool] = None,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    is_public: Optional[bool] = None,
) -> tuple[_Shape, dict[str, Any]]:
    """The conditions' shape, and their values: tags and the text looked for match as they are, ignoring case."""
    searched = query.strip() if query else ""
    tags = list(tags or ())
    slots = _slots(len(tags))
    shape = _Shape(
        categories=tuple(sorted({WorkflowCategory(category) for category in categories or ()})),
        tags=slots,
        has_been_opened=_flag(has_been_opened),
        searching=bool(searched),
        scoped=user_id is not None,
        is_public=_flag(is_public),
    )
    parameters: dict[str, Any] = {
        f"tag_{i}": like_contains(tags[i].strip()) if i < len(tags) else None for i in range(slots)
    }
    if searched:
        parameters["query"] = like_contains(searched)
    if user_id is not None:
        parameters["user_id"] = user_id
    return shape, parameters


_W = workflow_library.c
_SUMMARY_COLUMNS = (
    _W.workflow_id,
    _W.name,
    _W.created_at,
    _W.updated_at,
    _W.opened_at,
    _W.last_run_at,
    _W.user_id,
    _W.is_public,
    _W.revision,
)
_LIST_COLUMNS = (*_SUMMARY_COLUMNS, _W.description, _W.category, _W.tags)
_SUMMARY_NAMES = tuple(column.name for column in _SUMMARY_COLUMNS)
_LIST_NAMES = tuple(column.name for column in _LIST_COLUMNS)
_ORDER_COLUMNS = {
    WorkflowRecordOrderBy.CreatedAt: _W.created_at,
    WorkflowRecordOrderBy.UpdatedAt: _W.updated_at,
    WorkflowRecordOrderBy.OpenedAt: _W.opened_at,
    WorkflowRecordOrderBy.Name: _W.name,
    WorkflowRecordOrderBy.IsPublic: _W.is_public,
}

_THE_WORKFLOW = _W.workflow_id == bindparam("workflow_id")
_GET = select(*_SUMMARY_COLUMNS, _W.workflow).where(_THE_WORKFLOW)
_SUMMARY = select(*_SUMMARY_COLUMNS).where(_THE_WORKFLOW)
_DOCUMENTS = select(_W.workflow_id, _W.workflow).where(_W.workflow_id.in_(bindparam("workflow_ids", expanding=True)))
_LOCK = select(_W.category, _W.user_id, _W.revision).where(_THE_WORKFLOW).with_for_update()
_LOCK_WITH_DOCUMENT = select(_W.category, _W.user_id, _W.revision, _W.workflow).where(_THE_WORKFLOW).with_for_update()
_BUNDLED = select(_W.workflow_id, _W.name).where(_W.category == literal(WorkflowCategory.Default.value))
_INSERT = insert(workflow_library)
_SAVE = (
    update(workflow_library)
    .where(_W.workflow_id == bindparam("target_workflow_id"))
    .values(workflow=bindparam("new_workflow"), revision=_W.revision + 1)
)
_SET_PUBLIC = (
    update(workflow_library)
    .where(_W.workflow_id == bindparam("target_workflow_id"))
    .values(workflow=bindparam("new_workflow"), is_public=bindparam("new_is_public"))
)
_DELETE = delete(workflow_library).where(_THE_WORKFLOW)


def _conditions(shape: _Shape) -> list[ColumnElement[bool]]:
    """The conditions of a shape; the statements built from them are cached, not they."""
    conditions: list[ColumnElement[bool]] = []
    if shape.categories:
        conditions.append(_W.category.in_([literal(category.value) for category in shape.categories]))
    if shape.tags:
        conditions.append(or_(*(CaseInsensitiveLike(_W.tags, bindparam(f"tag_{i}")) for i in range(shape.tags))))
    if shape.has_been_opened is not None:
        conditions.append(_W.opened_at.is_not(None) if shape.has_been_opened else _W.opened_at.is_(None))
    if shape.searching:
        query: BindParameter[str] = bindparam("query")
        conditions.append(
            or_(
                CaseInsensitiveLike(_W.name, query),
                CaseInsensitiveLike(_W.description, query),
                CaseInsensitiveLike(_W.tags, query),
            )
        )
    if shape.scoped:
        # The account's own workflows, and the bundled ones every account sees.
        conditions.append(
            or_(_W.user_id == bindparam("user_id"), _W.category == literal(WorkflowCategory.Default.value))
        )
    if shape.is_public is not None:
        conditions.append(_W.is_public == (true() if shape.is_public else false()))
    return conditions


@functools.lru_cache(maxsize=256)
def _list(shape: _Shape, order_by: WorkflowRecordOrderBy, direction: SQLiteDirection, paged: bool) -> Select[Any]:
    # The enums are also strings, which share their cache entries: compare by value, never by identity. The workflow
    # id breaks ties, so that equal keys keep one order from page to page.
    ordering = (_ORDER_COLUMNS[order_by], _W.workflow_id)
    statement = (
        select(*_LIST_COLUMNS)
        .where(*_conditions(shape))
        .order_by(*(column.desc() if direction == SQLiteDirection.Descending else column.asc() for column in ordering))
    )
    if paged:
        statement = statement.limit(bindparam("limit")).offset(bindparam("offset"))
    return statement


@functools.lru_cache(maxsize=256)
def _count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(workflow_library).where(*_conditions(shape))


@functools.lru_cache(maxsize=256)
def _tag_counts(slots: int, shape: _Shape) -> Select[Unpack[tuple[Any, ...]]]:
    """How many of the workflows `shape` covers have each tag, in one pass."""
    counts: list[ColumnElement[int]] = [
        func.count(case((CaseInsensitiveLike(_W.tags, bindparam(f"count_{i}")), 1))) for i in range(slots)
    ]
    return select(*counts).where(*_conditions(shape))


@functools.lru_cache(maxsize=64)
def _category_count(shape: _Shape) -> Select[Any]:
    # One category per count, not one count grouped by category: the generated columns are virtual on SQLite, and
    # grouping reads each workflow's category from its document where a search by category reads it from the index.
    return (
        select(func.count())
        .select_from(workflow_library)
        .where(*_conditions(shape), _W.category == bindparam("category"))
    )


@functools.lru_cache(maxsize=64)
def _tag_lists(shape: _Shape) -> Select[Any]:
    return select(_W.tags).distinct().where(_W.tags.is_not(None), _W.tags != literal(""), *_conditions(shape))


@functools.cache
def _mark(column: str, scoped: bool) -> Update:
    """Sets a timestamp column of the workflow; with `scoped`, only when the account owns it."""
    statement = update(workflow_library).where(_W.workflow_id == bindparam("target_workflow_id"))
    if scoped:
        statement = statement.where(_W.user_id == bindparam("owner_id"))
    return statement.values({column: bindparam("now")})


def _record_or_none(row: Optional[Sequence[Any]]) -> Optional[WorkflowRecordDTO]:
    if row is None:
        return None
    *summary, workflow = row
    return WorkflowRecordDTO.from_dict({**dict(zip(_SUMMARY_NAMES, summary, strict=False)), "workflow": workflow})


def _summary_or_none(row: Optional[Sequence[Any]]) -> Optional[WorkflowRecordDTOBase]:
    return WorkflowRecordDTOBase(**dict(zip(_SUMMARY_NAMES, row, strict=False))) if row is not None else None


def _items_and_total(result: tuple[Sequence[Sequence[Any]], int]) -> tuple[list[WorkflowRecordListItemDTO], int]:
    rows, total = result
    items = [
        WorkflowRecordListItemDTOValidator.validate_python(dict(zip(_LIST_NAMES, row, strict=False))) for row in rows
    ]
    return items, total


class WorkflowQueries(QueryModule):
    @mapped(_record_or_none)
    @read
    def get(self, conn: Connection, workflow_id: str) -> Optional[Row[Any]]:
        """The workflow with its document."""
        return conn.execute(_GET, {"workflow_id": workflow_id}).first()

    @mapped(_summary_or_none)
    @read
    def summary(self, conn: Connection, workflow_id: str) -> Optional[Row[Any]]:
        """The workflow without its document."""
        return conn.execute(_SUMMARY, {"workflow_id": workflow_id}).first()

    @read
    def documents(self, conn: Connection, workflow_ids: Collection[str]) -> dict[str, str]:
        """The documents of those of these workflows that exist."""
        found: dict[str, str] = {}
        for chunk in itertools.batched(workflow_ids, IN_CHUNK):
            found.update((row[0], row[1]) for row in conn.execute(_DOCUMENTS, {"workflow_ids": list(chunk)}).all())
        return found

    @read
    def bundled(self, conn: Connection) -> list[tuple[str, str]]:
        """The id and name of every bundled workflow stored."""
        rows: Sequence[Sequence[Any]] = conn.execute(_BUNDLED).all()
        return [(workflow_id, name) for workflow_id, name in rows]

    @locking
    def lock(self, conn: Connection, workflow_id: str, *, with_document: bool = False) -> Optional[LockedWorkflow]:
        """Locks the workflow's row until the transaction ends, or returns None when there is no such workflow."""
        statement = _LOCK_WITH_DOCUMENT if with_document else _LOCK
        row: Optional[Sequence[Any]] = conn.execute(statement, {"workflow_id": workflow_id}).first()
        return LockedWorkflow(*row) if row is not None else None

    @mapped(_items_and_total)
    @read
    def page(
        self,
        conn: Connection,
        *,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        offset: int,
        limit: Optional[int],
        categories: Optional[Sequence[WorkflowCategory]],
        tags: Optional[Sequence[str]],
        has_been_opened: Optional[bool],
        query: Optional[str],
        user_id: Optional[str],
        is_public: Optional[bool],
    ) -> tuple[Sequence[Row[Any]], int]:
        """The workflows the filters keep, in order, `limit` of them from `offset` (all of them with None), and how
        many the filters keep in all. `user_id` keeps the account's own workflows and the bundled ones."""
        shape, parameters = _where(
            categories=categories,
            tags=tags,
            has_been_opened=has_been_opened,
            query=query,
            user_id=user_id,
            is_public=is_public,
        )
        if shape.tags <= _TAG_SLOTS:
            listing, count, options = _list(shape, order_by, direction, limit is not None), _count(shape), None
        else:
            listing = _list.__wrapped__(shape, order_by, direction, limit is not None)
            count, options = _count.__wrapped__(shape), {"compiled_cache": None}
        rows = conn.execute(listing, {**parameters, "limit": limit, "offset": offset}, execution_options=options).all()
        total: int = conn.execute(count, parameters, execution_options=options).scalar_one()
        return rows, total

    @read
    def tag_counts(
        self,
        conn: Connection,
        tags: Sequence[str],
        *,
        categories: Optional[Sequence[WorkflowCategory]],
        has_been_opened: Optional[bool],
        user_id: Optional[str],
        is_public: Optional[bool],
    ) -> list[int]:
        """How many of the workflows the filters keep have each of `tags`."""
        shape, parameters = _where(
            categories=categories, has_been_opened=has_been_opened, user_id=user_id, is_public=is_public
        )
        counts: list[int] = []
        for chunk in itertools.batched(tags, _TAG_SLOTS):
            slots = _slots(len(chunk))
            patterns = {f"count_{i}": like_contains(chunk[i].strip()) if i < len(chunk) else None for i in range(slots)}
            row: Sequence[Any] = conn.execute(_tag_counts(slots, shape), {**parameters, **patterns}).one()
            counts.extend(row[: len(chunk)])
        return counts

    @read
    def category_counts(
        self,
        conn: Connection,
        categories: Sequence[WorkflowCategory],
        *,
        has_been_opened: Optional[bool],
        user_id: Optional[str],
        is_public: Optional[bool],
    ) -> dict[str, int]:
        """How many of the workflows the filters keep are in each of `categories`."""
        shape, parameters = _where(
            categories=categories, has_been_opened=has_been_opened, user_id=user_id, is_public=is_public
        )
        statement = _category_count(shape)
        return {
            category.value: conn.execute(statement, {**parameters, "category": category.value}).scalar_one()
            for category in dict.fromkeys(WorkflowCategory(category) for category in categories)
        }

    @read
    def tag_lists(
        self,
        conn: Connection,
        *,
        categories: Optional[Sequence[WorkflowCategory]],
        user_id: Optional[str],
        is_public: Optional[bool],
    ) -> list[str]:
        """The distinct comma-separated tags of the workflows the filters keep, of those that have tags."""
        shape, parameters = _where(categories=categories, user_id=user_id, is_public=is_public)
        return list(conn.execute(_tag_lists(shape), parameters).scalars().all())

    @write
    def insert(self, conn: Connection, *, workflow_id: str, workflow: str, user_id: str, is_public: bool) -> None:
        """Adds a workflow at revision 1."""
        conn.execute(
            _INSERT, {"workflow_id": workflow_id, "workflow": workflow, "user_id": user_id, "is_public": is_public}
        )

    @write
    def save(self, conn: Connection, workflow_id: str, workflow: str) -> None:
        """Writes the workflow's document as its next revision."""
        conn.execute(_SAVE, {"target_workflow_id": workflow_id, "new_workflow": workflow})

    @write
    def set_public(self, conn: Connection, workflow_id: str, workflow: str, is_public: bool) -> None:
        """Writes whether the workflow is shared, with its document carrying the matching tag; the revision stays."""
        conn.execute(
            _SET_PUBLIC, {"target_workflow_id": workflow_id, "new_workflow": workflow, "new_is_public": is_public}
        )

    @write
    def mark(
        self, conn: Connection, column: Literal["opened_at", "last_run_at"], workflow_id: str, user_id: Optional[str]
    ) -> None:
        """Sets the workflow's `opened_at` or `last_run_at` to now; with `user_id`, only if the account owns it."""
        parameters = {"target_workflow_id": workflow_id, "now": now_text(), "owner_id": user_id}
        conn.execute(_mark(column, user_id is not None), parameters)

    @write
    def delete(self, conn: Connection, workflow_id: str) -> None:
        conn.execute(_DELETE, {"workflow_id": workflow_id})
