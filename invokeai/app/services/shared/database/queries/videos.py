"""Video records: one row per video file, with its generation metadata."""

import functools
import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    Update,
    bindparam,
    delete,
    false,
    func,
    literal,
    or_,
    select,
    true,
    update,
)

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    JsonString,
    fixed_limit,
    insert_ignore,
    like_contains,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.boards import board_videos
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.video_records.video_records_common import (
    VideoRecord,
    VideoRecordChanges,
    deserialize_video_record,
)

_V = videos.c
_BV = board_videos.c

# The columns of a `VideoRecord`, and the one member of the metadata the clients need on every row: it marks an
# upload converted from an audio file. Only it is read, so that listings do not carry whole metadata documents.
_RECORD = (
    _V.video_name,
    _V.video_origin,
    _V.video_category,
    _V.width,
    _V.height,
    _V.duration,
    _V.fps,
    _V.session_id,
    _V.node_id,
    _V.has_workflow,
    _V.is_intermediate,
    _V.created_at,
    _V.updated_at,
    _V.deleted_at,
    _V.starred,
    _V.video_subfolder,
    _V.project_id,
    _V.file_size_bytes,
    JsonString(_V.metadata, "$.media_origin").label("media_origin"),
)
_RECORD_NAMES = tuple(column.name for column in _RECORD)

_GET = select(*_RECORD).where(_V.video_name == bindparam("video_name"))
_GET_USER_ID = select(_V.user_id).where(_V.video_name == bindparam("video_name"))
_GET_METADATA = select(_V.metadata).where(_V.video_name == bindparam("video_name"))
_GET_CREATED_AT = select(_V.created_at).where(_V.video_name == bindparam("video_name"))
_EXISTS = select(literal(1)).where(_V.video_name == bindparam("video_name"))
_SUBFOLDERS = select(_V.video_name, _V.video_subfolder).where(
    _V.video_name.in_(bindparam("video_names", expanding=True))
)
_MOST_RECENT_ON_BOARD = (
    select(*_RECORD)
    .join(board_videos, _BV.video_name == _V.video_name)
    .where(_BV.board_id == bindparam("board_id"), _V.is_intermediate == false())
    .order_by(_V.starred.desc(), _V.created_at.desc(), _V.video_name.desc())
    .limit(fixed_limit(1))
)
_SET_FILE_SIZE = (
    update(videos)
    .where(_V.video_name == bindparam("target_video_name"))
    .values(file_size_bytes=bindparam("new_file_size_bytes"))
)
# A backfill fills gaps only: the writer's own measurement, taken after the file exists, wins over one taken before.
_FILL_FILE_SIZE = (
    update(videos)
    .where(_V.video_name == bindparam("target_video_name"), _V.file_size_bytes.is_(None))
    .values(file_size_bytes=bindparam("new_file_size_bytes"))
)
_DELETE = delete(videos).where(_V.video_name == bindparam("video_name"))
_DELETE_MANY = delete(videos).where(_V.video_name.in_(bindparam("video_names", expanding=True)))
# Locked, so that a video promoted out of the intermediates meanwhile is either seen promoted or deleted before it is.
_LOCK_INTERMEDIATES = (
    select(_V.video_name)
    .where(_V.video_name.in_(bindparam("video_names", expanding=True)), _V.is_intermediate == true())
    .with_for_update()
)
_DELETE_INTERMEDIATES = delete(videos).where(
    _V.video_name.in_(bindparam("video_names", expanding=True)), _V.is_intermediate == true()
)


@functools.cache
def _insert(dialect_name: str) -> Any:
    # A name that is taken keeps its row, as SQLite's INSERT OR IGNORE did.
    return insert_ignore(dialect_name, videos)


@functools.cache
def _update(fields: tuple[str, ...]) -> Update:
    """Sets these fields of the video; one statement per set of fields, of which there are few."""
    return (
        update(videos)
        .where(_V.video_name == bindparam("target_video_name"))
        .values({field: bindparam(f"new_{field}") for field in fields})
    )


class _Shape(NamedTuple):
    """Which filters a listing has: never their values, which are bound, so that the shapes are few."""

    origin: bool
    # The categories' values, sorted; None for no category filter. Each value is bound on its own: SQLAlchemy compiles
    # lists of one length once, whatever they hold.
    categories: Optional[tuple[str, ...]]
    intermediate: bool
    # "any" (no board filter), "none" (on no board) or "one".
    board: str
    # Whether the listing is limited to what one account may see; else everything, as for an administrator.
    scoped: bool
    search: bool


def _shape(
    *,
    video_origin: Optional[ResourceOrigin],
    categories: Optional[Sequence[ImageCategory]],
    is_intermediate: Optional[bool],
    board_id: Optional[str],
    search_term: Optional[str],
    user_id: Optional[str],
    is_admin: bool,
) -> _Shape:
    return _Shape(
        origin=video_origin is not None,
        categories=None if categories is None else tuple(sorted({ImageCategory(c).value for c in categories})),
        intermediate=is_intermediate is not None,
        board="any" if board_id is None else "none" if board_id == "none" else "one",
        scoped=user_id is not None and not is_admin,
        search=bool(search_term),
    )


def _parameters(
    *,
    video_origin: Optional[ResourceOrigin],
    is_intermediate: Optional[bool],
    board_id: Optional[str],
    search_term: Optional[str],
    user_id: Optional[str],
) -> dict[str, Any]:
    return {
        "video_origin": video_origin.value if video_origin is not None else None,
        "is_intermediate": is_intermediate,
        "board_id": board_id,
        "pattern": like_contains(search_term) if search_term else None,
        "user_id": user_id,
    }


def _conditions(shape: _Shape) -> list[ColumnElement[bool]]:
    """The listing's conditions, on the videos LEFT JOINed with their board memberships."""
    conditions: list[ColumnElement[bool]] = []
    if shape.origin:
        conditions.append(_V.video_origin == bindparam("video_origin"))
    if shape.categories is not None:
        conditions.append(_V.video_category.in_([literal(category) for category in shape.categories]))
    if shape.intermediate:
        conditions.append(_V.is_intermediate == bindparam("is_intermediate"))
    own = _V.user_id == bindparam("user_id")
    if shape.board == "none":
        conditions.append(_BV.board_id.is_(None))
        if shape.scoped:
            conditions.append(own)
    elif shape.board == "one":
        conditions.append(_BV.board_id == bindparam("board_id"))
    elif shape.scoped:
        # Without a board, still only the account's own videos, so that it cannot list every account's.
        conditions.append(own)
    if shape.search:
        pattern: ColumnElement[str] = bindparam("pattern")
        conditions.append(or_(CaseInsensitiveLike(_V.metadata, pattern), CaseInsensitiveLike(_V.created_at, pattern)))
    return conditions


def _ordering(starred_first: bool, descending: bool) -> list[ColumnElement[Any]]:
    ordering: list[ColumnElement[Any]] = (
        [_V.created_at.desc(), _V.video_name.desc()] if descending else [_V.created_at, _V.video_name]
    )
    return [_V.starred.desc(), *ordering] if starred_first else ordering


_JOINED_FROM = videos.outerjoin(board_videos, _BV.video_name == _V.video_name)


@functools.lru_cache(maxsize=128)
def _page(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    """A page of records; the number of shapes is bounded, the cache keeps the recent ones."""
    return (
        select(*_RECORD)
        .select_from(_JOINED_FROM)
        .where(*_conditions(shape))
        .order_by(*_ordering(starred_first, descending))
        .limit(bindparam("limit"))
        .offset(bindparam("offset"))
    )


@functools.lru_cache(maxsize=128)
def _count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(_JOINED_FROM).where(*_conditions(shape))


@functools.lru_cache(maxsize=128)
def _names(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    return (
        select(_V.video_name)
        .select_from(_JOINED_FROM)
        .where(*_conditions(shape))
        .order_by(*_ordering(starred_first, descending))
    )


@functools.lru_cache(maxsize=128)
def _starred_count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(_JOINED_FROM).where(_V.starred == true(), *_conditions(shape))


def _record(row: Sequence[Any]) -> VideoRecord:
    return deserialize_video_record(dict(zip(_RECORD_NAMES, row, strict=True)))


def _record_or_none(row: Optional[Sequence[Any]]) -> Optional[VideoRecord]:
    return _record(row) if row is not None else None


def _records(page: tuple[Sequence[Sequence[Any]], int]) -> tuple[list[VideoRecord], int]:
    rows, total = page
    return [_record(row) for row in rows], total


class VideoQueries(QueryModule):
    @mapped(_record_or_none)
    @read
    def get(self, conn: Connection, video_name: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"video_name": video_name}).first()

    @read
    def user_id(self, conn: Connection, video_name: str) -> Optional[str]:
        """The video's owner; None also for a video that does not exist."""
        return conn.execute(_GET_USER_ID, {"video_name": video_name}).scalar()

    @read
    def metadata(self, conn: Connection, video_name: str) -> tuple[bool, Optional[str]]:
        """Whether the video exists, and its metadata as stored: JSON text, or None."""
        row = conn.execute(_GET_METADATA, {"video_name": video_name}).first()
        return (False, None) if row is None else (True, row[0])

    @read
    def exists(self, conn: Connection, video_name: str) -> bool:
        return conn.execute(_EXISTS, {"video_name": video_name}).first() is not None

    @read
    def subfolders(self, conn: Connection, video_names: Sequence[str]) -> dict[str, str]:
        """The subfolder of each named video that exists."""
        subfolders: dict[str, str] = {}
        for chunk in itertools.batched(video_names, IN_CHUNK):
            subfolders.update((row[0], row[1]) for row in conn.execute(_SUBFOLDERS, {"video_names": list(chunk)}))
        return subfolders

    @mapped(_record_or_none)
    @read
    def most_recent_on_board(self, conn: Connection, board_id: str) -> Optional[Row[Any]]:
        """The board's cover: its newest video that is not an intermediate, a starred one first."""
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
        video_origin: Optional[ResourceOrigin],
        categories: Optional[Sequence[ImageCategory]],
        is_intermediate: Optional[bool],
        board_id: Optional[str],
        search_term: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
    ) -> tuple[Sequence[Row[Any]], int]:
        """A page of the records that match, and how many match in all."""
        shape = _shape(
            video_origin=video_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        parameters = _parameters(
            video_origin=video_origin,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
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
        video_origin: Optional[ResourceOrigin],
        categories: Optional[Sequence[ImageCategory]],
        is_intermediate: Optional[bool],
        board_id: Optional[str],
        search_term: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
    ) -> tuple[list[str], int]:
        """The names of the videos that match, in order, and how many of them are starred (with `starred_first`;
        else 0)."""
        shape = _shape(
            video_origin=video_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        parameters = _parameters(
            video_origin=video_origin,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
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
        video_name: str,
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        width: int,
        height: int,
        duration: float,
        fps: Optional[float],
        has_workflow: bool,
        is_intermediate: Optional[bool],
        starred: Optional[bool],
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Optional[str],
        user_id: str,
        video_subfolder: str,
        project_id: Optional[str],
    ) -> None:
        """Adds the video, unless a video of that name exists."""
        conn.execute(
            _insert(conn.dialect.name),
            {
                "video_name": video_name,
                "video_origin": video_origin.value,
                "video_category": video_category.value,
                "width": width,
                "height": height,
                "duration": float(duration),
                "fps": float(fps) if fps is not None else None,
                "node_id": node_id,
                "session_id": session_id,
                "metadata": metadata,
                "is_intermediate": is_intermediate,
                "starred": starred,
                "has_workflow": has_workflow,
                "user_id": user_id,
                "video_subfolder": video_subfolder,
                "project_id": project_id,
            },
        )

    @read
    def created_at(self, conn: Connection, video_name: str) -> Optional[str]:
        return conn.execute(_GET_CREATED_AT, {"video_name": video_name}).scalar()

    @write
    def update(self, conn: Connection, video_name: str, changes: VideoRecordChanges) -> None:
        """Applies the changes to the video's category, session, intermediate flag and star."""
        fields: dict[str, Any] = {}
        if changes.video_category is not None:
            fields["video_category"] = ImageCategory(changes.video_category).value
        if changes.session_id is not None:
            fields["session_id"] = changes.session_id
        if changes.is_intermediate is not None:
            fields["is_intermediate"] = changes.is_intermediate
        if changes.starred is not None:
            fields["starred"] = changes.starred
        if fields:
            values = {f"new_{field}": value for field, value in fields.items()}
            conn.execute(_update(tuple(sorted(fields))), {"target_video_name": video_name, **values})

    @write
    def set_file_size(self, conn: Connection, video_name: str, file_size_bytes: Optional[int]) -> None:
        conn.execute(_SET_FILE_SIZE, {"target_video_name": video_name, "new_file_size_bytes": file_size_bytes})

    @write
    def fill_file_sizes(self, conn: Connection, sizes: dict[str, int]) -> None:
        """Records the sizes of the videos whose size is not known yet."""
        if sizes:
            conn.execute(
                _FILL_FILE_SIZE,
                [{"target_video_name": name, "new_file_size_bytes": size} for name, size in sizes.items()],
            )

    @write
    def delete(self, conn: Connection, video_name: str) -> None:
        conn.execute(_DELETE, {"video_name": video_name})

    @write
    def delete_many(self, conn: Connection, video_names: Sequence[str]) -> None:
        for chunk in itertools.batched(video_names, IN_CHUNK):
            conn.execute(_DELETE_MANY, {"video_names": list(chunk)})

    @write
    def delete_intermediates(self, conn: Connection, video_names: Sequence[str]) -> list[str]:
        """Deletes those of the named videos that are intermediates; the names deleted, in the given order."""
        deleted: list[str] = []
        for chunk in itertools.batched(video_names, IN_CHUNK):
            names = list(chunk)
            locked: set[str] = set(conn.execute(_LOCK_INTERMEDIATES, {"video_names": names}).scalars())
            if locked:
                conn.execute(_DELETE_INTERMEDIATES, {"video_names": sorted(locked)})
                deleted.extend(name for name in names if name in locked)
        return deleted
