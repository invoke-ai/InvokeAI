from datetime import date
from typing import Optional

from fastapi import HTTPException, Query
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.images import _assert_board_read_access
from invokeai.app.services.gallery.gallery_common import GalleryItem, GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE, OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

gallery_router = APIRouter(prefix="/v1/gallery", tags=["gallery"])


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
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find items without a board.",
    ),
    # Bounds matter: these flow verbatim into SQL, and a negative LIMIT means
    # *unlimited* in SQLite — one request would materialize the entire gallery.
    offset: int = Query(default=0, ge=0, description="The page offset"),
    limit: int = Query(default=10, ge=0, le=MAX_PAGE_SIZE, description="The number of items per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred items first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
    created_from: Optional[date] = Query(
        default=None, description="Inclusive start date (YYYY-MM-DD) to filter by created_at."
    ),
    created_to: Optional[date] = Query(
        default=None, description="Inclusive end date (YYYY-MM-DD) to filter by created_at."
    ),
) -> OffsetPaginatedResults[GalleryItem]:
    """Returns a paginated, time-sorted stream of polymorphic gallery items (images + videos)."""
    if board_id is not None and board_id != "none":
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
    )


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
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find items without a board.",
    ),
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
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> GalleryItemNames:
    """Returns the ordered flat list of item names — used to drive virtualized gallery selection.

    Names are polymorphic: image and video names are interleaved by `created_at`. A name ending
    in `.mp4` is a video.
    """
    if board_id is not None and board_id != "none":
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
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find items without a board.",
    ),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred items first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
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
    if board_id is not None and board_id != "none":
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
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get gallery item names")
