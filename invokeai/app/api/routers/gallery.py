from datetime import date
from typing import Annotated, Optional, Self

from fastapi import Body, HTTPException, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field, model_validator

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._access import assert_board_changes
from invokeai.app.api.routers._limits import MAX_IMAGE_BATCH_SIZE, MAX_VIDEO_BATCH_SIZE
from invokeai.app.api.routers.images import _assert_board_read_access
from invokeai.app.services.board_access.board_access_common import BoardAction
from invokeai.app.services.gallery.gallery_base import GalleryMediaKind
from invokeai.app.services.gallery.gallery_common import (
    GalleryItem,
    GalleryItemKind,
    GalleryItemLocation,
    GalleryItemNames,
    GalleryItemNamesResult,
    GalleryMediaUsage,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE, OffsetPaginatedResults, SQLiteDirection

gallery_router = APIRouter(prefix="/v1/gallery", tags=["gallery"])

# Scopes rather than boards: the listing itself limits them to what the caller may read, so no board is checked.
_BOARD_SENTINELS = ("none", "all")

BoardIdQuery = Query(
    default=None,
    description="The board id to filter by. Use 'none' for items on no board (the current user's, unless an "
    "administrator), or 'all' for those items plus the items on every non-archived board the current user may read.",
)

UsageMediaName = Annotated[str, Field(min_length=1, max_length=255)]


class GalleryMediaUsageRequest(BaseModel):
    """The media to check: named images and videos, or every item on one board."""

    image_names: list[UsageMediaName] = Field(
        default_factory=list,
        max_length=MAX_IMAGE_BATCH_SIZE,
        description="Images to check. Those the caller may not delete, and unknown ones, are ignored, as "
        "`POST /images/delete` skips them.",
    )
    video_names: list[UsageMediaName] = Field(
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


@gallery_router.get(
    "/items/",
    operation_id="list_gallery_items",
    response_model=OffsetPaginatedResults[GalleryItem],
)
def list_gallery_items(
    current_user: CurrentUserOrDefault,
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
    )


@gallery_router.get(
    "/items/location",
    operation_id="get_gallery_item_location",
    response_model=GalleryItemLocation,
)
def get_gallery_item_location(
    current_user: CurrentUserOrDefault,
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
