"""Image records: one row per image file, with its generation metadata, listed by the gallery."""

import functools
import itertools
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    Update,
    and_,
    bindparam,
    delete,
    exists,
    false,
    func,
    literal,
    or_,
    select,
    true,
    update,
)

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ResourceOrigin,
    deserialize_image_record,
)
from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    fixed_limit,
    insert_ignore,
    like_contains,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.boards import board_images, boards, shared_boards
from invokeai.app.services.shared.database.schema.images import images

_I = images.c
_BI = board_images.c
_B = boards.c
_S = shared_boards.c

# The columns of an `ImageRecord`, which has no metadata and no owner.
_RECORD = (
    _I.image_name,
    _I.image_origin,
    _I.image_category,
    _I.width,
    _I.height,
    _I.session_id,
    _I.node_id,
    _I.has_workflow,
    _I.is_intermediate,
    _I.created_at,
    _I.updated_at,
    _I.deleted_at,
    _I.starred,
    _I.image_subfolder,
    _I.project_id,
    _I.file_size_bytes,
)
_RECORD_NAMES = tuple(column.name for column in _RECORD)
_SHARED_VISIBILITIES = (literal(BoardVisibility.Shared.value), literal(BoardVisibility.Public.value))

_GET = select(*_RECORD).where(_I.image_name == bindparam("image_name"))
_GET_USER_ID = select(_I.user_id).where(_I.image_name == bindparam("image_name"))
_GET_METADATA = select(_I.metadata).where(_I.image_name == bindparam("image_name"))
_GET_CREATED_AT = select(_I.created_at).where(_I.image_name == bindparam("image_name"))
_EXISTS = select(literal(1)).where(_I.image_name == bindparam("image_name"))
_SUBFOLDERS = select(_I.image_name, _I.image_subfolder).where(
    _I.image_name.in_(bindparam("image_names", expanding=True))
)
_MOST_RECENT_ON_BOARD = (
    select(*_RECORD)
    .join(board_images, _BI.image_name == _I.image_name)
    .where(_BI.board_id == bindparam("board_id"), _I.is_intermediate == false())
    .order_by(_I.starred.desc(), _I.created_at.desc(), _I.image_name.desc())
    .limit(fixed_limit(1))
)
_SET_FILE_SIZE = (
    update(images)
    .where(_I.image_name == bindparam("target_image_name"))
    .values(file_size_bytes=bindparam("new_file_size_bytes"))
)
# A backfill fills gaps only: the writer's own measurement, taken after the file exists, wins over one taken before.
_FILL_FILE_SIZE = (
    update(images)
    .where(_I.image_name == bindparam("target_image_name"), _I.file_size_bytes.is_(None))
    .values(file_size_bytes=bindparam("new_file_size_bytes"))
)
_DELETE = delete(images).where(_I.image_name == bindparam("image_name"))
_DELETE_MANY = delete(images).where(_I.image_name.in_(bindparam("image_names", expanding=True)))
# Locked, so that an image promoted out of the intermediates meanwhile is either seen promoted or deleted before it is.
_LOCK_INTERMEDIATES = (
    select(_I.image_name)
    .where(_I.image_name.in_(bindparam("image_names", expanding=True)), _I.is_intermediate == true())
    .with_for_update()
)
_DELETE_INTERMEDIATES = delete(images).where(
    _I.image_name.in_(bindparam("image_names", expanding=True)), _I.is_intermediate == true()
)


@functools.cache
def _insert(dialect_name: str) -> Any:
    # A name that is taken keeps its row, as SQLite's INSERT OR IGNORE did.
    return insert_ignore(dialect_name, images)


@functools.cache
def _update(fields: tuple[str, ...]) -> Update:
    """Sets these fields of the image; one statement per set of fields, of which there are few."""
    return (
        update(images)
        .where(_I.image_name == bindparam("target_image_name"))
        .values({field: bindparam(f"new_{field}") for field in fields})
    )


class _Shape(NamedTuple):
    """Which filters a listing has: never their values, which are bound, so that the shapes are few."""

    origin: bool
    # The categories' values, sorted; None for no category filter. Each value is bound on its own: SQLAlchemy compiles
    # lists of one length once, whatever they hold.
    categories: Optional[tuple[str, ...]]
    intermediate: bool
    # "any" (no board filter), "none" (on no board), "all" (on no board or a readable one) or "one".
    board: str
    # Whether the listing is limited to what one account may see; else everything, as for an administrator.
    scoped: bool
    search: bool
    created_from: bool
    created_to: bool


def _shape(
    *,
    image_origin: Optional[ResourceOrigin],
    categories: Optional[Sequence[ImageCategory]],
    is_intermediate: Optional[bool],
    board_id: Optional[str],
    search_term: Optional[str],
    created_from: Optional[str],
    created_to: Optional[str],
    user_id: Optional[str],
    is_admin: bool,
) -> _Shape:
    board = "any" if board_id is None else board_id if board_id in ("none", "all") else "one"
    return _Shape(
        origin=image_origin is not None,
        categories=None if categories is None else tuple(sorted({ImageCategory(c).value for c in categories})),
        intermediate=is_intermediate is not None,
        board=board,
        scoped=user_id is not None and not is_admin,
        search=bool(search_term),
        created_from=created_from is not None,
        created_to=created_to is not None,
    )


def _day_after(day: str) -> str:
    """The day after an ISO day (`YYYY-MM-DD`), as text that timestamps of that day sort before. An invalid day, or
    the last one there is, gives the empty text, which no timestamp sorts before: SQLite's DATE() gave NULL for both,
    which matched nothing."""
    try:
        parsed = date.fromisoformat(day)
        return (parsed + timedelta(days=1)).isoformat() if parsed.isoformat() == day else ""
    except (ValueError, OverflowError):
        return ""


def _parameters(
    *,
    image_origin: Optional[ResourceOrigin],
    is_intermediate: Optional[bool],
    board_id: Optional[str],
    search_term: Optional[str],
    created_from: Optional[str],
    created_to: Optional[str],
    user_id: Optional[str],
) -> dict[str, Any]:
    return {
        "image_origin": image_origin.value if image_origin is not None else None,
        "is_intermediate": is_intermediate,
        "board_id": board_id,
        "pattern": like_contains(search_term) if search_term else None,
        "created_from": created_from,
        "created_before": _day_after(created_to) if created_to is not None else None,
        "user_id": user_id,
    }


def _listed_board(scoped: bool) -> list[ColumnElement[bool]]:
    """The conditions on a board whose images the scope lists: active, and with `scoped` one the account may read
    (its own, shared with everyone, or shared with the account)."""
    if not scoped:
        return [_B.archived == false()]
    shared_with_account = exists(
        select(literal(1)).where(_S.board_id == _B.board_id, _S.user_id == bindparam("user_id"))
    )
    readable = or_(
        _B.user_id == bindparam("user_id"), _B.board_visibility.in_(_SHARED_VISIBILITIES), shared_with_account
    )
    return [_B.archived == false(), readable]


def _conditions(shape: _Shape, joined: bool) -> list[ColumnElement[bool]]:
    """The listing's conditions. `joined` lists through a LEFT JOIN of the board memberships, for the board data a
    page of records returns; else the memberships are checked per image, so that SQLite scans the images alone."""
    conditions: list[ColumnElement[bool]] = []
    if shape.origin:
        conditions.append(_I.image_origin == bindparam("image_origin"))
    if shape.categories is not None:
        conditions.append(_I.image_category.in_([literal(category) for category in shape.categories]))
    if shape.intermediate:
        conditions.append(_I.is_intermediate == bindparam("is_intermediate"))

    own = _I.user_id == bindparam("user_id")
    membership = select(literal(1)).where(_BI.image_name == _I.image_name)
    on_no_board = _BI.board_id.is_(None) if joined else ~exists(membership)
    if shape.board == "none":
        conditions.append(on_no_board)
        if shape.scoped:
            conditions.append(own)
    elif shape.board == "all":
        if joined:
            board = exists(select(literal(1)).where(_B.board_id == _BI.board_id, *_listed_board(shape.scoped)))
        else:
            memberships = board_images.join(boards, _B.board_id == _BI.board_id)
            board = exists(
                select(literal(1))
                .select_from(memberships)
                .where(_BI.image_name == _I.image_name, *_listed_board(shape.scoped))
            )
        conditions.append(or_(and_(on_no_board, own) if shape.scoped else on_no_board, board))
    elif shape.board == "one":
        on_the_board = _BI.board_id == bindparam("board_id")
        conditions.append(on_the_board if joined else exists(membership.where(on_the_board)))
    elif shape.scoped:
        conditions.append(own)

    if shape.search:
        pattern: ColumnElement[str] = bindparam("pattern")
        conditions.append(or_(CaseInsensitiveLike(_I.metadata, pattern), CaseInsensitiveLike(_I.created_at, pattern)))
    if shape.created_from:
        conditions.append(_I.created_at >= bindparam("created_from"))
    if shape.created_to:
        conditions.append(_I.created_at < bindparam("created_before"))
    return conditions


def _ordering(starred_first: bool, descending: bool) -> list[ColumnElement[Any]]:
    # The name breaks ties, so that images of the same moment keep one order from page to page.
    ordering: list[ColumnElement[Any]] = (
        [_I.created_at.desc(), _I.image_name.desc()] if descending else [_I.created_at, _I.image_name]
    )
    return [_I.starred.desc(), *ordering] if starred_first else ordering


_JOINED_FROM = images.outerjoin(board_images, _BI.image_name == _I.image_name)


@functools.lru_cache(maxsize=256)
def _page(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    """A page of records; the number of shapes is bounded, the cache keeps the recent ones."""
    return (
        select(*_RECORD)
        .select_from(_JOINED_FROM)
        .where(*_conditions(shape, joined=True))
        .order_by(*_ordering(starred_first, descending))
        .limit(bindparam("limit"))
        .offset(bindparam("offset"))
    )


@functools.lru_cache(maxsize=256)
def _count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(_JOINED_FROM).where(*_conditions(shape, joined=True))


@functools.lru_cache(maxsize=256)
def _names(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    return (
        select(_I.image_name).where(*_conditions(shape, joined=False)).order_by(*_ordering(starred_first, descending))
    )


@functools.lru_cache(maxsize=256)
def _starred_count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(images).where(_I.starred == true(), *_conditions(shape, joined=False))


def _record(row: Sequence[Any]) -> ImageRecord:
    return deserialize_image_record(dict(zip(_RECORD_NAMES, row, strict=True)))


def _record_or_none(row: Optional[Sequence[Any]]) -> Optional[ImageRecord]:
    return _record(row) if row is not None else None


def _records(page: tuple[Sequence[Sequence[Any]], int]) -> tuple[list[ImageRecord], int]:
    rows, total = page
    return [_record(row) for row in rows], total


class ImageQueries(QueryModule):
    @mapped(_record_or_none)
    @read
    def get(self, conn: Connection, image_name: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"image_name": image_name}).first()

    @read
    def user_id(self, conn: Connection, image_name: str) -> Optional[str]:
        """The image's owner; None also for an image that does not exist."""
        return conn.execute(_GET_USER_ID, {"image_name": image_name}).scalar()

    @read
    def metadata(self, conn: Connection, image_name: str) -> tuple[bool, Optional[str]]:
        """Whether the image exists, and its metadata as stored: JSON text, or None."""
        row = conn.execute(_GET_METADATA, {"image_name": image_name}).first()
        return (False, None) if row is None else (True, row[0])

    @read
    def exists(self, conn: Connection, image_name: str) -> bool:
        return conn.execute(_EXISTS, {"image_name": image_name}).first() is not None

    @read
    def subfolders(self, conn: Connection, image_names: Sequence[str]) -> dict[str, str]:
        """The subfolder of each named image that exists."""
        subfolders: dict[str, str] = {}
        for chunk in itertools.batched(image_names, IN_CHUNK):
            subfolders.update((row[0], row[1]) for row in conn.execute(_SUBFOLDERS, {"image_names": list(chunk)}))
        return subfolders

    @mapped(_record_or_none)
    @read
    def most_recent_on_board(self, conn: Connection, board_id: str) -> Optional[Row[Any]]:
        """The board's cover: its newest image that is not an intermediate, a starred one first."""
        return conn.execute(_MOST_RECENT_ON_BOARD, {"board_id": board_id}).first()

    @mapped(_records)
    @read
    def page(
        self,
        conn: Connection,
        *,
        offset: int,
        limit: int,
        starred_first: bool,
        descending: bool,
        image_origin: Optional[ResourceOrigin],
        categories: Optional[Sequence[ImageCategory]],
        is_intermediate: Optional[bool],
        board_id: Optional[str],
        search_term: Optional[str],
        created_from: Optional[str],
        created_to: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
    ) -> tuple[Sequence[Row[Any]], int]:
        """A page of the records that match, and how many match in all."""
        shape = _shape(
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
            is_admin=is_admin,
        )
        parameters = _parameters(
            image_origin=image_origin,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
        )
        rows = conn.execute(
            _page(shape, starred_first, descending), {**parameters, "limit": limit, "offset": offset}
        ).all()
        return rows, conn.execute(_count(shape), parameters).scalar_one()

    @read
    def names(
        self,
        conn: Connection,
        *,
        starred_first: bool,
        descending: bool,
        image_origin: Optional[ResourceOrigin],
        categories: Optional[Sequence[ImageCategory]],
        is_intermediate: Optional[bool],
        board_id: Optional[str],
        search_term: Optional[str],
        created_from: Optional[str],
        created_to: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
    ) -> tuple[list[str], int]:
        """The names of the images that match, in order, and how many of them are starred (with `starred_first`;
        else 0)."""
        shape = _shape(
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
            is_admin=is_admin,
        )
        parameters = _parameters(
            image_origin=image_origin,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
        )
        starred = conn.execute(_starred_count(shape), parameters).scalar_one() if starred_first else 0
        names = list(conn.execute(_names(shape, starred_first, descending), parameters).scalars().all())
        return names, starred

    @write
    def insert(
        self,
        conn: Connection,
        *,
        image_name: str,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        width: int,
        height: int,
        has_workflow: bool,
        is_intermediate: Optional[bool],
        starred: Optional[bool],
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Optional[str],
        user_id: str,
        image_subfolder: str,
        project_id: Optional[str],
    ) -> None:
        """Adds the image, unless an image of that name exists."""
        conn.execute(
            _insert(conn.dialect.name),
            {
                "image_name": image_name,
                "image_origin": image_origin.value,
                "image_category": image_category.value,
                "width": width,
                "height": height,
                "node_id": node_id,
                "session_id": session_id,
                "metadata": metadata,
                "is_intermediate": is_intermediate,
                "starred": starred,
                "has_workflow": has_workflow,
                "user_id": user_id,
                "image_subfolder": image_subfolder,
                "project_id": project_id,
            },
        )

    @read
    def created_at(self, conn: Connection, image_name: str) -> Optional[str]:
        return conn.execute(_GET_CREATED_AT, {"image_name": image_name}).scalar()

    @write
    def update(self, conn: Connection, image_name: str, changes: ImageRecordChanges) -> None:
        """Applies the changes to the image's category, session, intermediate flag and star."""
        fields: dict[str, Any] = {}
        if changes.image_category is not None:
            fields["image_category"] = ImageCategory(changes.image_category).value
        if changes.session_id is not None:
            fields["session_id"] = changes.session_id
        if changes.is_intermediate is not None:
            fields["is_intermediate"] = changes.is_intermediate
        if changes.starred is not None:
            fields["starred"] = changes.starred
        if fields:
            values = {f"new_{field}": value for field, value in fields.items()}
            conn.execute(_update(tuple(sorted(fields))), {"target_image_name": image_name, **values})

    @write
    def set_file_size(self, conn: Connection, image_name: str, file_size_bytes: Optional[int]) -> None:
        conn.execute(_SET_FILE_SIZE, {"target_image_name": image_name, "new_file_size_bytes": file_size_bytes})

    @write
    def fill_file_sizes(self, conn: Connection, sizes: dict[str, int]) -> None:
        """Records the sizes of the images whose size is not known yet."""
        if sizes:
            conn.execute(
                _FILL_FILE_SIZE,
                [{"target_image_name": name, "new_file_size_bytes": size} for name, size in sizes.items()],
            )

    @write
    def delete(self, conn: Connection, image_name: str) -> None:
        conn.execute(_DELETE, {"image_name": image_name})

    @write
    def delete_many(self, conn: Connection, image_names: Sequence[str]) -> None:
        for chunk in itertools.batched(image_names, IN_CHUNK):
            conn.execute(_DELETE_MANY, {"image_names": list(chunk)})

    @write
    def delete_intermediates(self, conn: Connection, image_names: Sequence[str]) -> list[str]:
        """Deletes those of the named images that are intermediates; the names deleted, in the given order."""
        deleted: list[str] = []
        for chunk in itertools.batched(image_names, IN_CHUNK):
            names = list(chunk)
            locked: set[str] = set(conn.execute(_LOCK_INTERMEDIATES, {"image_names": names}).scalars())
            if locked:
                conn.execute(_DELETE_INTERMEDIATES, {"image_names": sorted(locked)})
                deleted.extend(name for name in names if name in locked)
        return deleted
