"""The gallery: images and videos listed as one, through a UNION ALL of two halves with the same columns.

The filters apply alike to each half. A literal `kind` tells the halves apart, and duration and fps are NULL for
images. Lists break ties by kind and name in the listing's direction, so that items of one moment keep one order
from page to page.
"""

import functools
import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    FromClause,
    Row,
    Select,
    Subquery,
    Table,
    and_,
    bindparam,
    case,
    exists,
    false,
    func,
    literal,
    null,
    or_,
    select,
    union_all,
)

from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES,
    IMAGE_CATEGORIES,
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    JsonString,
    OrderedJoin,
    like_contains,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, day_after, read
from invokeai.app.services.shared.database.schema.boards import board_images, board_videos
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.videos import videos


class _Half(NamedTuple):
    media: Table
    membership: Table
    name: str
    category: str
    origin: str


_HALVES = {
    "image": _Half(images, board_images, "image_name", "image_category", "image_origin"),
    "video": _Half(videos, board_videos, "video_name", "video_category", "video_origin"),
}


class _Shape(NamedTuple):
    """Which filters a listing has: never their values, which are bound, so that the shapes are few."""

    origin: bool
    # The categories' values, sorted; None for no category filter.
    categories: Optional[tuple[str, ...]]
    intermediate: bool
    starred: bool
    # "any" (no board filter), "none" (on no board) or "one".
    board: str
    # Whether the listing is limited to one account's items; else everything, as for an administrator.
    scoped: bool
    search: bool
    created_date: bool
    created_from: bool
    created_to: bool


class Filters(NamedTuple):
    """A gallery listing's filters, as the service passes them."""

    origin: Optional[ResourceOrigin] = None
    categories: Optional[Sequence[ImageCategory]] = None
    is_intermediate: Optional[bool] = None
    board_id: Optional[str] = None
    search_term: Optional[str] = None
    user_id: Optional[str] = None
    is_admin: bool = False
    created_date: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    starred: Optional[bool] = None

    def shape(self) -> _Shape:
        return _Shape(
            origin=self.origin is not None,
            categories=None
            if self.categories is None
            else tuple(sorted({ImageCategory(c).value for c in self.categories})),
            intermediate=self.is_intermediate is not None,
            starred=self.starred is not None,
            board="any" if self.board_id is None else "none" if self.board_id == "none" else "one",
            scoped=self.user_id is not None and not self.is_admin,
            search=bool(self.search_term),
            created_date=self.created_date is not None,
            created_from=self.created_from is not None,
            created_to=self.created_to is not None,
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "origin": self.origin.value if self.origin is not None else None,
            "is_intermediate": self.is_intermediate,
            "starred": self.starred,
            "board_id": self.board_id,
            "user_id": self.user_id,
            # Lowered here, as the legacy services did: SQLite's LIKE folds the case of ASCII letters only.
            "pattern": like_contains(self.search_term.lower()) if self.search_term else None,
            "created_date": self.created_date,
            "created_date_before": day_after(self.created_date) if self.created_date is not None else None,
            "created_from": self.created_from,
            "created_before": day_after(self.created_to) if self.created_to is not None else None,
        }


def _from(kind: str, shape: _Shape, with_board: bool) -> tuple[FromClause, ColumnElement[Any]]:
    """The half's FROM and its board id: a board's items come through its memberships, so that the work stays
    proportional to the board; a listing of every item joins the memberships only for the board id it shows."""
    half = _HALVES[kind]
    media, membership = half.media, half.membership
    on = membership.c[half.name] == media.c[half.name]
    if shape.board == "one":
        return OrderedJoin(membership, media, on), membership.c.board_id
    if shape.board == "any" and with_board:
        return media.outerjoin(membership, on), membership.c.board_id
    return media, null()


def _conditions(kind: str, shape: _Shape) -> list[ColumnElement[bool]]:
    half = _HALVES[kind]
    media, membership = half.media, half.membership
    conditions: list[ColumnElement[bool]] = []
    if shape.origin:
        conditions.append(media.c[half.origin] == bindparam("origin"))
    if shape.categories is not None:
        conditions.append(media.c[half.category].in_([literal(category) for category in shape.categories]))
    if shape.intermediate:
        conditions.append(media.c.is_intermediate == bindparam("is_intermediate"))
    if shape.starred:
        conditions.append(media.c.starred == bindparam("starred"))
    if shape.created_date:
        conditions.append(media.c.created_at >= bindparam("created_date"))
        conditions.append(media.c.created_at < bindparam("created_date_before"))
    if shape.created_from:
        conditions.append(media.c.created_at >= bindparam("created_from"))
    if shape.created_to:
        conditions.append(media.c.created_at < bindparam("created_before"))
    own = media.c.user_id == bindparam("user_id")
    if shape.board == "none":
        conditions.append(~exists(select(literal(1)).where(membership.c[half.name] == media.c[half.name])))
        if shape.scoped:
            conditions.append(own)
    elif shape.board == "one":
        conditions.append(membership.c.board_id == bindparam("board_id"))
    elif shape.scoped:
        # Without a board, still only the account's own items, so that it cannot list every account's.
        conditions.append(own)
    if shape.search:
        pattern: ColumnElement[str] = bindparam("pattern")
        conditions.append(
            or_(CaseInsensitiveLike(media.c.metadata, pattern), CaseInsensitiveLike(media.c.created_at, pattern))
        )
    return conditions


def _half(kind: str, shape: _Shape, names_only: bool) -> Select[Any]:
    half = _HALVES[kind]
    media = half.media
    source, board_id = _from(kind, shape, with_board=not names_only)
    if names_only:
        columns: list[ColumnElement[Any]] = [
            literal(kind).label("kind"),
            media.c[half.name].label("name"),
            media.c.starred.label("starred"),
            media.c.created_at.label("created_at"),
        ]
    else:
        columns = [
            literal(kind).label("kind"),
            media.c[half.name].label("name"),
            media.c.width.label("width"),
            media.c.height.label("height"),
            media.c[half.category].label("category"),
            media.c.starred.label("starred"),
            media.c.is_intermediate.label("is_intermediate"),
            board_id.label("board_id"),
            media.c.created_at.label("created_at"),
            (media.c.duration if kind == "video" else null()).label("duration"),
            (media.c.fps if kind == "video" else null()).label("fps"),
        ]
    return select(*columns).select_from(source).where(*_conditions(kind, shape))


def _ordering(items: Subquery, starred_first: bool, descending: bool) -> list[ColumnElement[Any]]:
    keys = [items.c.created_at, items.c.kind, items.c.name]
    ordering = [key.desc() if descending else key.asc() for key in keys]
    return [items.c.starred.desc(), *ordering] if starred_first else ordering


@functools.lru_cache(maxsize=256)
def _page(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    """A page of items. `media_origin` is joined onto the chosen page rather than selected inside the halves: it is
    read out of `videos.metadata`, and selecting it there would read and parse every video before LIMIT."""
    items = union_all(_half("image", shape, False), _half("video", shape, False)).subquery("items")
    page = (
        select(items)
        .order_by(*_ordering(items, starred_first, descending))
        .limit(bindparam("limit"))
        .offset(bindparam("offset"))
        .subquery("page")
    )
    video_of_page = and_(page.c.kind == literal("video"), videos.c.video_name == page.c.name)
    return (
        select(*page.c, JsonString(videos.c.metadata, "$.media_origin").label("media_origin"))
        .select_from(page.outerjoin(videos, video_of_page))
        .order_by(*_ordering(page, starred_first, descending))
    )


@functools.lru_cache(maxsize=256)
def _count(shape: _Shape) -> Select[Any]:
    """The items of both halves that match, in one statement."""
    counts = []
    for kind in _HALVES:
        # Without the board id a listing shows: a membership join cannot change the count, and MySQL does not
        # eliminate it.
        source, _ = _from(kind, shape, with_board=False)
        counts.append(select(func.count()).select_from(source).where(*_conditions(kind, shape)).scalar_subquery())
    return select(counts[0] + counts[1])


@functools.lru_cache(maxsize=256)
def _names(shape: _Shape, starred_first: bool, descending: bool) -> Select[Any]:
    items = union_all(_half("image", shape, True), _half("video", shape, True)).subquery("items")
    return select(items.c.kind, items.c.name, items.c.starred).order_by(*_ordering(items, starred_first, descending))


def _in(column: ColumnElement[Any], categories: Sequence[ImageCategory]) -> ColumnElement[bool]:
    return column.in_([literal(category.value) for category in categories])


def _counted(items: Subquery) -> list[ColumnElement[int]]:
    """Image and asset counts as the gallery's views list them (`IMAGE_CATEGORIES`, `ASSETS_CATEGORIES`), so that a
    count never includes canvas-owned images (OTHER), which neither view shows; and every video, and those of a
    category other than general, as a board counts its videos."""
    is_image, is_video = items.c.kind == literal("image"), items.c.kind == literal("video")
    return [
        func.sum(case((and_(is_image, _in(items.c.category, IMAGE_CATEGORIES)), 1), else_=0)),
        func.sum(case((and_(is_image, _in(items.c.category, ASSETS_CATEGORIES)), 1), else_=0)),
        func.sum(case((is_video, 1), else_=0)),
        func.sum(case((and_(is_video, items.c.category != literal(ImageCategory.GENERAL.value)), 1), else_=0)),
    ]


def _dated(kind: str, scoped: bool) -> Select[Any]:
    half = _HALVES[kind]
    media = half.media
    conditions = [media.c.is_intermediate == false()]
    if kind == "image":
        # Only what the gallery's views show: canvas-owned images (OTHER) neither make a day nor cover one.
        conditions.append(_in(media.c[half.category], [*IMAGE_CATEGORIES, *ASSETS_CATEGORIES]))
    if scoped:
        conditions.append(media.c.user_id == bindparam("user_id"))
    return select(
        # The day of a canonical timestamp: its first ten characters, on every backend.
        func.substr(media.c.created_at, 1, 10).label("day"),
        literal(kind).label("kind"),
        media.c[half.name].label("name"),
        media.c[half.category].label("category"),
        media.c.created_at.label("created_at"),
    ).where(*conditions)


@functools.cache
def _date_counts(scoped: bool) -> Select[Any]:
    items = union_all(_dated("image", scoped), _dated("video", scoped)).subquery("items")
    return select(items.c.day, *_counted(items)).group_by(items.c.day).order_by(items.c.day.desc())


@functools.cache
def _date_covers(scoped: bool) -> Select[Any]:
    """The newest item of each day; kind and name break ties, so that the cover does not flicker between refetches."""
    items = union_all(_dated("image", scoped), _dated("video", scoped)).subquery("items")
    newest_first = func.row_number().over(
        partition_by=items.c.day, order_by=[items.c.created_at.desc(), items.c.kind.desc(), items.c.name.desc()]
    )
    ranked = select(items.c.day, items.c.kind, items.c.name, newest_first.label("rank")).subquery("ranked")
    return select(ranked.c.day, ranked.c.kind, ranked.c.name).where(ranked.c.rank == 1)


def _on_board(kind: str) -> Select[Any]:
    half = _HALVES[kind]
    media, membership = half.media, half.membership
    return (
        select(
            membership.c.board_id.label("board_id"),
            literal(kind).label("kind"),
            media.c[half.name].label("name"),
            media.c[half.category].label("category"),
            media.c.starred.label("starred"),
            media.c.created_at.label("created_at"),
        )
        .select_from(membership.join(media, membership.c[half.name] == media.c[half.name]))
        .where(media.c.is_intermediate == false(), membership.c.board_id.in_(bindparam("board_ids", expanding=True)))
    )


def _board_summaries() -> Select[Any]:
    items = union_all(_on_board("image"), _on_board("video")).subquery("items")
    cover_first = func.row_number().over(
        partition_by=items.c.board_id,
        order_by=[items.c.starred.desc(), items.c.created_at.desc(), items.c.kind.desc(), items.c.name.desc()],
    )
    ranked = select(*items.c, cover_first.label("rank")).subquery("ranked")
    cover = ranked.c.rank == 1
    return select(
        ranked.c.board_id,
        *_counted(ranked),
        func.max(case((and_(cover, ranked.c.kind == literal("image")), ranked.c.name))),
        func.max(case((and_(cover, ranked.c.kind == literal("video")), ranked.c.name))),
    ).group_by(ranked.c.board_id)


_BOARD_SUMMARIES = _board_summaries()


class DateCounts(NamedTuple):
    day: str
    images: int
    assets: int
    videos: int
    asset_videos: int


class BoardSummary(NamedTuple):
    board_id: str
    images: int
    assets: int
    videos: int
    asset_videos: int
    cover_image_name: Optional[str]
    cover_video_name: Optional[str]


class GalleryQueries(QueryModule):
    @read
    def page(
        self, conn: Connection, filters: Filters, *, offset: int, limit: int, starred_first: bool, descending: bool
    ) -> tuple[Sequence[Row[Any]], int]:
        """A page of items, with the columns of `_half` and `media_origin`, and how many match in all."""
        shape = filters.shape()
        parameters = filters.parameters()
        rows = conn.execute(
            _page(shape, starred_first, descending), {**parameters, "limit": limit, "offset": offset}
        ).all()
        return rows, int(conn.execute(_count(shape), parameters).scalar_one())

    @read
    def names(self, conn: Connection, filters: Filters, *, starred_first: bool, descending: bool) -> Sequence[Row[Any]]:
        """(kind, name, starred) of every item that matches, in order."""
        return conn.execute(_names(filters.shape(), starred_first, descending), filters.parameters()).all()

    @read
    def date_counts(
        self, conn: Connection, user_id: Optional[str]
    ) -> tuple[list[DateCounts], dict[str, tuple[str, str]]]:
        """The counts of each day that has items that are not intermediates (only `user_id`'s, if given), newest
        first, and each day's cover as (kind, name)."""
        scoped = user_id is not None
        parameters = {"user_id": user_id}
        counts = [
            DateCounts(str(r[0]), int(r[1] or 0), int(r[2] or 0), int(r[3] or 0), int(r[4] or 0))
            for r in conn.execute(_date_counts(scoped), parameters)
        ]
        covers = {str(r[0]): (str(r[1]), str(r[2])) for r in conn.execute(_date_covers(scoped), parameters)}
        return counts, covers

    @read
    def board_summaries(self, conn: Connection, board_ids: Sequence[str]) -> list[BoardSummary]:
        """Counts and covers of the boards' items that are not intermediates; a board without any has no row."""
        summaries: list[BoardSummary] = []
        for chunk in itertools.batched(board_ids, IN_CHUNK // 2):
            for r in conn.execute(_BOARD_SUMMARIES, {"board_ids": list(chunk)}):
                summaries.append(
                    BoardSummary(str(r[0]), int(r[1] or 0), int(r[2] or 0), int(r[3] or 0), int(r[4] or 0), r[5], r[6])
                )
        return summaries
