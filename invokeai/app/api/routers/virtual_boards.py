from fastapi import HTTPException, Path, Query
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageNamesResult
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO

virtual_boards_router = APIRouter(prefix="/v1/virtual_boards", tags=["virtual_boards"])


@virtual_boards_router.get(
    "/by_date",
    operation_id="list_virtual_boards_by_date",
    response_model=list[VirtualSubBoardDTO],
)
async def list_virtual_boards_by_date(
    current_user: CurrentUserOrDefault,
) -> list[VirtualSubBoardDTO]:
    """Gets a list of virtual sub-boards grouped by date."""
    try:
        return ApiDependencies.invoker.services.image_records.get_image_dates(
            user_id=current_user.user_id,
            is_admin=current_user.is_admin,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get virtual boards by date")


@virtual_boards_router.get(
    "/by_date/{date}/image_names",
    operation_id="list_virtual_board_image_names_by_date",
    response_model=ImageNamesResult,
)
async def list_virtual_board_image_names_by_date(
    current_user: CurrentUserOrDefault,
    date: str = Path(description="The ISO date string, e.g. '2026-03-18'"),
    starred_first: bool = Query(default=True, description="Whether to sort starred images first"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The sort direction"),
    categories: list[ImageCategory] | None = Query(default=None, description="The categories of images to include"),
    search_term: str | None = Query(default=None, description="Search term to filter images"),
) -> ImageNamesResult:
    """Gets ordered image names for a specific date."""
    try:
        return ApiDependencies.invoker.services.image_records.get_image_names_by_date(
            date=date,
            starred_first=starred_first,
            order_dir=order_dir,
            categories=categories,
            search_term=search_term,
            user_id=current_user.user_id,
            is_admin=current_user.is_admin,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get image names for date")
