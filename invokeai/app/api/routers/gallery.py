from collections.abc import Callable
from datetime import date
from typing import Annotated, Optional, Self

from fastapi import Body, Depends, HTTPException, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._access import assert_board_changes, assert_image_action, assert_video_action
from invokeai.app.api.routers._limits import MAX_IMAGE_BATCH_SIZE, MAX_VIDEO_BATCH_SIZE
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.api.routers.images import _assert_board_read_access
from invokeai.app.services.board_access.board_access_common import BoardAction, ItemAction
from invokeai.app.services.gallery.gallery_base import GalleryMediaKind
from invokeai.app.services.gallery.gallery_common import (
    GalleryItem,
    GalleryItemKind,
    GalleryItemLocation,
    GalleryItemMarksResult,
    GalleryItemNames,
    GalleryItemNamesResult,
    GalleryMediaUsage,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.media_marks import (
    MAX_RATING,
    ColorLabel,
    ColorLabelChoice,
    MarkChanges,
    MarkFilters,
    MarkFlag,
)
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE, OffsetPaginatedResults, SQLiteDirection

gallery_router = APIRouter(prefix="/v1/gallery", tags=["gallery"])

# Scopes rather than boards: the listing itself limits them to what the caller may read, so no board is checked.
_BOARD_SENTINELS = ("none", "all")

BoardIdQuery = Query(
    default=None,
    description="The board id to filter by. Use 'none' for items on no board (the current user's, unless an "
    "administrator), or 'all' for those items plus the items on every non-archived board the current user may read.",
)

MediaName = Annotated[str, Field(min_length=1, max_length=255)]


class GalleryMediaUsageRequest(BaseModel):
    """The media to check: named images and videos, or every item on one board."""

    image_names: list[MediaName] = Field(
        default_factory=list,
        max_length=MAX_IMAGE_BATCH_SIZE,
        description="Images to check. Those the caller may not delete, and unknown ones, are ignored, as "
        "`POST /images/delete` skips them.",
    )
    video_names: list[MediaName] = Field(
        default_factory=list,
        max_length=MAX_VIDEO_BATCH_SIZE,
        description="Videos to check. Those the caller may not delete, and unknown ones, are ignored, as "
        "`POST /videos/delete` skips them.",
    )
    board_id: Optional[str] = Field(
        default=None,
        description="Check the images and videos `DELETE /boards/{board_id}?include_images=true` would delete, under "
        "the same access rules. Not combined with names.",
    )

    @model_validator(mode="after")
    def _one_scope(self) -> Self:
        if self.board_id is not None and (self.image_names or self.video_names):
            raise ValueError("Check either names or a board, not both")
        return self


MediaKindQuery = Query(
    default=None,
    description="List only images ('image') or only videos ('video'). Omit to include both. Totals, positions and "
    "names cover only the listed kind.",
)


def _mark_filters(
    flags: Optional[list[MarkFlag]] = Query(
        default=None,
        max_length=len(MarkFlag),
        description="Keep items with any of these flags: 'pick' (starred), 'reject', or 'none' for neither.",
    ),
    min_rating: Optional[int] = Query(
        default=None, ge=1, le=MAX_RATING, description="Keep items rated at least this (1-5)."
    ),
    max_rating: Optional[int] = Query(
        default=None, ge=0, le=MAX_RATING, description="Keep items rated at most this (0-5); 0 keeps unrated items."
    ),
    color_labels: Optional[list[ColorLabelChoice]] = Query(
        default=None,
        max_length=len(ColorLabel) + 1,
        description="Keep items with any of these color labels; 'none' keeps items without one.",
    ),
) -> MarkFilters:
    """A listing's mark filters, combined with each other and the listing's other filters by AND."""
    if min_rating is not None and max_rating is not None and min_rating > max_rating:
        raise HTTPException(status_code=422, detail="min_rating is above max_rating")
    return MarkFilters(flags=flags, min_rating=min_rating, max_rating=max_rating, color_labels=color_labels)


MarkFiltersQuery = Annotated[MarkFilters, Depends(_mark_filters)]


@gallery_router.get(
    "/items/",
    operation_id="list_gallery_items",
    response_model=OffsetPaginatedResults[GalleryItem],
)
def list_gallery_items(
    current_user: CurrentUserOrDefault,
    marks: MarkFiltersQuery,
    origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of items to list."),
    categories: Optional[list[ImageCategory]] = Query(
        default=None,
        description="The categories to include. Shared between images and videos.",
    ),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate items."),
    board_id: Optional[str] = BoardIdQuery,
    # Bounds matter: these flow verbatim into SQL, and a negative LIMIT means
    # *unlimited* in SQLite — one request would materialize the entire gallery.
    offset: int = Query(default=0, ge=0, description="The page offset"),
    limit: int = Query(default=10, ge=0, le=MAX_PAGE_SIZE, description="The number of items per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred items first"),
    starred: Optional[bool] = Query(
        default=None,
        description="Filter by starred state: true for starred items only, false for unstarred only. Omit to include both.",
    ),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
    media_kind: Optional[GalleryMediaKind] = MediaKindQuery,
    created_from: Optional[date] = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD) to filter by created_at."
    ),
    created_to: Optional[date] = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD) to filter by created_at."
    ),
) -> OffsetPaginatedResults[GalleryItem]:
    """Returns a paginated, time-sorted stream of polymorphic gallery items (images + videos)."""
    if board_id is not None and board_id not in _BOARD_SENTINELS:
        _assert_board_read_access(board_id, current_user)

    return ApiDependencies.invoker.services.gallery.list_items(
        offset=offset,
        limit=limit,
        starred_first=starred_first,
        order_dir=order_dir,
        origin=origin,
        categories=categories,
        is_intermediate=is_intermediate,
        board_id=board_id,
        search_term=search_term,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        created_from=created_from.isoformat() if created_from else None,
        created_to=created_to.isoformat() if created_to else None,
        starred=starred,
        media_kind=media_kind,
        marks=marks,
    )


@gallery_router.get(
    "/items/location",
    operation_id="get_gallery_item_location",
    response_model=GalleryItemLocation,
)
def get_gallery_item_location(
    current_user: CurrentUserOrDefault,
    marks: MarkFiltersQuery,
    kind: GalleryItemKind = Query(description="Whether the target is an image or video."),
    name: str = Query(description="The target image or video name."),
    origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of items to list."),
    categories: Optional[list[ImageCategory]] = Query(
        default=None,
        description="The categories to include. Shared between images and videos.",
    ),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate items."),
    board_id: Optional[str] = BoardIdQuery,
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred: Optional[bool] = Query(
        default=None,
        description="Filter by starred state: true for starred items only, false for unstarred only. Omit to include both.",
    ),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
    media_kind: Optional[GalleryMediaKind] = MediaKindQuery,
    created_from: Optional[date] = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD) to filter by created_at."
    ),
    created_to: Optional[date] = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD) to filter by created_at."
    ),
) -> GalleryItemLocation:
    """Returns exact item's position in an ordinary filtered gallery listing."""
    if board_id is not None and board_id not in _BOARD_SENTINELS:
        _assert_board_read_access(board_id, current_user)

    location = ApiDependencies.invoker.services.gallery.get_item_location(
        kind=kind,
        name=name,
        order_dir=order_dir,
        origin=origin,
        categories=categories,
        is_intermediate=is_intermediate,
        board_id=board_id,
        search_term=search_term,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        created_from=created_from.isoformat() if created_from else None,
        created_to=created_to.isoformat() if created_to else None,
        starred=starred,
        media_kind=media_kind,
        marks=marks,
    )
    if location is None:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return location


@gallery_router.get(
    "/item_names",
    operation_id="list_gallery_item_names",
    response_model=GalleryItemNames,
)
def list_gallery_item_names(
    current_user: CurrentUserOrDefault,
    marks: MarkFiltersQuery,
    origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of items to list."),
    categories: Optional[list[ImageCategory]] = Query(
        default=None,
        description="The categories to include. Shared between images and videos.",
    ),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate items."),
    board_id: Optional[str] = BoardIdQuery,
    created_date: Optional[str] = Query(
        default=None,
        description="Restrict to items created on this ISO date, e.g. '2026-03-18'. Used by date-based virtual boards.",
    ),
    created_from: Optional[date] = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD) to filter by created_at."
    ),
    created_to: Optional[date] = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD) to filter by created_at."
    ),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred items first"),
    starred: Optional[bool] = Query(
        default=None,
        description="Filter by starred state: true for starred items only, false for unstarred only. Omit to include both.",
    ),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
    media_kind: Optional[GalleryMediaKind] = MediaKindQuery,
) -> GalleryItemNames:
    """Returns the ordered flat list of item names — used to drive virtualized gallery selection.

    Names are polymorphic: image and video names are interleaved by `created_at`. A name ending
    in `.mp4` is a video.
    """
    if board_id is not None and board_id not in _BOARD_SENTINELS:
        _assert_board_read_access(board_id, current_user)

    try:
        return ApiDependencies.invoker.services.gallery.get_item_names(
            starred_first=starred_first,
            order_dir=order_dir,
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=current_user.user_id,
            is_admin=current_user.is_admin,
            created_date=created_date,
            created_from=created_from.isoformat() if created_from else None,
            created_to=created_to.isoformat() if created_to else None,
            starred=starred,
            media_kind=media_kind,
            marks=marks,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get gallery item names")


@gallery_router.get(
    "/items/names",
    operation_id="get_gallery_item_names",
    response_model=GalleryItemNamesResult,
    deprecated=True,
)
def get_gallery_item_names(
    current_user: CurrentUserOrDefault,
    marks: MarkFiltersQuery,
    origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of items to list."),
    categories: Optional[list[ImageCategory]] = Query(
        default=None,
        description="The categories to include. Shared between images and videos.",
    ),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate items."),
    board_id: Optional[str] = BoardIdQuery,
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred items first"),
    starred: Optional[bool] = Query(
        default=None,
        description="Filter by starred state: true for starred items only, false for unstarred only. Omit to include both.",
    ),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
    media_kind: Optional[GalleryMediaKind] = MediaKindQuery,
    created_from: Optional[date] = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD) to filter by created_at."
    ),
    created_to: Optional[date] = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD) to filter by created_at."
    ),
) -> GalleryItemNamesResult:
    """Returns an ordered (kind, name) list — used to drive virtualized gallery selection.

    Deprecated: use `GET /v1/gallery/item_names`, which returns the same order as a flat name
    list. The `kind` discriminator here costs a model per row — ~800ms on a 200k-item library —
    for a value callers already derive from the file extension.
    """
    if board_id is not None and board_id not in _BOARD_SENTINELS:
        _assert_board_read_access(board_id, current_user)

    try:
        return ApiDependencies.invoker.services.gallery.list_item_names(
            starred_first=starred_first,
            order_dir=order_dir,
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=current_user.user_id,
            is_admin=current_user.is_admin,
            created_from=created_from.isoformat() if created_from else None,
            created_to=created_to.isoformat() if created_to else None,
            starred=starred,
            media_kind=media_kind,
            marks=marks,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get gallery item names")


@gallery_router.post("/usage", operation_id="get_gallery_media_usage", response_model=GalleryMediaUsage)
def get_gallery_media_usage(
    current_user: CurrentUserOrDefault,
    request: GalleryMediaUsageRequest = Body(description="The media to check"),
) -> GalleryMediaUsage:
    """Reports what still uses media before it is deleted: the caller's saved documents and active queue work by
    name, other accounts' as counts. Deleting stays a separate request; this only reads."""
    gallery = ApiDependencies.invoker.services.gallery
    if request.board_id is None:
        return gallery.get_media_usage(
            request.image_names, request.video_names, current_user.user_id, current_user.is_admin
        )
    access = assert_board_changes(request.board_id, current_user, [BoardAction.Delete])
    if request.board_id in ApiDependencies.invoker.services.board_records.get_inbox_board_ids([request.board_id]):
        raise HTTPException(status_code=409, detail="This board is the project's inbox; delete the project instead")
    return gallery.get_board_media_usage(
        request.board_id, current_user.user_id, access.media_deleted_with_board(current_user.user_id)
    )


class GalleryItemMarksRequest(BaseModel):
    """Marks to set on images and videos. A mark left out stays as it is on each item."""

    model_config = ConfigDict(extra="forbid")

    image_names: list[MediaName] = Field(
        default_factory=list, max_length=MAX_IMAGE_BATCH_SIZE, description="The images to mark."
    )
    video_names: list[MediaName] = Field(
        default_factory=list, max_length=MAX_VIDEO_BATCH_SIZE, description="The videos to mark."
    )
    rating: Optional[StrictInt] = Field(
        default=None, ge=0, le=MAX_RATING, description="The rating to set, from 0 (unrated) to 5."
    )
    flag: Optional[MarkFlag] = Field(
        default=None,
        description="The flag to set: 'pick' stars the item, 'reject' rejects it, 'none' clears both. Either one "
        "clears the other.",
    )
    color_label: Optional[ColorLabelChoice] = Field(
        default=None, description="The color label to set, or 'none' to remove it."
    )

    @model_validator(mode="after")
    def _something_to_mark(self) -> Self:
        if not self.image_names and not self.video_names:
            raise ValueError("Name at least one image or video")
        if self.rating is None and self.flag is None and self.color_label is None:
            raise ValueError("Give at least one mark to set")
        return self


def _markable(names: list[str], may_mark: Callable[[str], None], failed: list[str]) -> list[str]:
    """The names the caller may mark, once each, in order. Names they may not mark, and names of items that do not
    exist, are skipped as the star routes skip them; a name whose check failed otherwise goes to `failed`."""
    markable: list[str] = []
    for name in dict.fromkeys(names):
        try:
            may_mark(name)
        except HTTPException:
            continue
        except Exception:
            failed.append(name)
            continue
        markable.append(name)
    return markable


@gallery_router.post("/items/marks", operation_id="set_gallery_item_marks", response_model=GalleryItemMarksResult)
def set_gallery_item_marks(
    current_user: CurrentUserOrDefault,
    request: GalleryItemMarksRequest = Body(description="The items and the marks to set on them"),
) -> GalleryItemMarksResult:
    """Sets a rating, a flag and a color label on images and videos. Marking someone else's item takes what starring
    it does: being an administrator or one of the managers of the board it is on."""

    def may_mark_image(name: str) -> None:
        assert_image_action(name, current_user, ItemAction.Star)

    def may_mark_video(name: str) -> None:
        assert_video_action(name, current_user, ItemAction.Star)

    if request.image_names:
        try:
            assert_image_move_maintenance_inactive()
        except HTTPException:
            # As `POST /images/star`: a caller that may not mark an image learns that, not about the maintenance.
            for name in request.image_names:
                may_mark_image(name)
            raise

    failed_images: list[str] = []
    failed_videos: list[str] = []
    images = _markable(request.image_names, may_mark_image, failed_images)
    videos = _markable(request.video_names, may_mark_video, failed_videos)
    changes = MarkChanges(rating=request.rating, flag=request.flag, color_label=request.color_label)
    try:
        marked_images, marked_videos = ApiDependencies.invoker.services.gallery.set_marks(images, videos, changes)
    except Exception:
        # One transaction for every item: none of them was marked.
        ApiDependencies.invoker.services.logger.error("Failed to set gallery item marks", exc_info=True)
        return GalleryItemMarksResult(
            updated_image_names=[],
            updated_video_names=[],
            failed_image_names=failed_images + images,
            failed_video_names=failed_videos + videos,
            affected_boards=[],
        )
    boards = {board_id or "none" for board_id in (*marked_images.values(), *marked_videos.values())}
    return GalleryItemMarksResult(
        updated_image_names=[name for name in images if name in marked_images],
        updated_video_names=[name for name in videos if name in marked_videos],
        failed_image_names=failed_images,
        failed_video_names=failed_videos,
        affected_boards=sorted(boards),
    )
