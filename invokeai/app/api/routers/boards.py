from typing import Optional, Union

from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._access import assert_board_read_access as _assert_board_read_access
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecordOrderBy,
    BoardRecordProjectOwnedException,
)
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])


class DeleteBoardResult(BaseModel):
    board_id: str = Field(description="The id of the board that was deleted.")
    deleted_board_images: list[str] = Field(
        description="The image names of the board-images relationships that were deleted."
    )
    deleted_images: list[str] = Field(description="The names of the images that were deleted.")
    deleted_board_videos: list[str] = Field(
        default_factory=list,
        description="The video names of the board-videos relationships that were deleted.",
    )
    deleted_videos: list[str] = Field(
        default_factory=list,
        description="The names of the videos that were deleted.",
    )
    failed_images: list[str] = Field(
        default_factory=list,
        description="The names of images that could not be deleted and became uncategorized.",
    )
    failed_videos: list[str] = Field(
        default_factory=list,
        description="The names of videos that could not be deleted and became uncategorized.",
    )


@boards_router.post(
    "/",
    operation_id="create_board",
    responses={
        201: {"description": "The board was created successfully"},
    },
    status_code=201,
    response_model=BoardDTO,
)
async def create_board(
    current_user: CurrentUserOrDefault,
    board_name: str = Query(description="The name of the board to create", max_length=300),
) -> BoardDTO:
    """Creates a board for the current user"""
    try:
        result = ApiDependencies.invoker.services.boards.create(board_name=board_name, user_id=current_user.user_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create board")


@boards_router.get("/{board_id}", operation_id="get_board", response_model=BoardDTO)
async def get_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to get"),
) -> BoardDTO:
    """Gets a board (user must have access to it)"""

    _assert_board_read_access(board_id, current_user)

    try:
        return ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")


@boards_router.patch(
    "/{board_id}",
    operation_id="update_board",
    responses={
        201: {
            "description": "The board was updated successfully",
        },
    },
    status_code=201,
    response_model=BoardDTO,
)
async def update_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to update"),
    changes: BoardChanges = Body(description="The changes to apply to the board"),
) -> BoardDTO:
    """Updates a board (user must have access to it)"""
    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if not current_user.is_admin and board.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this board")

    # A project's board takes its name, archived state and visibility from the project, so the
    # generic route must not set them — for admins either. The cover is still fair game: it is a
    # display detail with no bearing on the project relationship.
    if board.project_id is not None and (
        changes.board_name is not None or changes.archived is not None or changes.board_visibility is not None
    ):
        raise HTTPException(
            status_code=409,
            detail="This board belongs to a project; rename or archive the project instead",
        )

    try:
        result = ApiDependencies.invoker.services.boards.update(board_id=board_id, changes=changes)
        return result
    except BoardRecordProjectOwnedException:
        raise HTTPException(
            status_code=409,
            detail="This board belongs to a project; rename or archive the project instead",
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update board")


@boards_router.delete("/{board_id}", operation_id="delete_board", response_model=DeleteBoardResult)
def delete_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of board to delete"),
    include_images: Optional[bool] = Query(
        description="Permanently delete all images and videos on the board", default=False
    ),
) -> DeleteBoardResult:
    """Deletes a board (user must have access to it)"""
    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if not current_user.is_admin and board.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this board")

    # Admins delete everything on the board; regular owners only delete their own
    # contributions so that contributions from other users to a public/shared board
    # are preserved (they cascade to "uncategorized" via FK on board_videos / board_images).
    cascade_user_id: Optional[str] = None if current_user.is_admin else current_user.user_id
    deleted_images: list[str] = []
    deleted_videos: list[str] = []
    board_deleted = False

    try:
        if include_images is True:
            assert_image_move_maintenance_inactive()

        # Enumerate first, delete the board second, delete its media third. The order matters:
        # a project's board must be refused *before* anything is destroyed, and the conditional
        # delete is the only check that cannot lose a race against a project claiming the board.
        # Membership rows are gone by the time the media is deleted (the board FK cascades), which
        # is why the names have to be captured up front.
        #
        # Without `include_images` every membership is dropped regardless of who contributed it,
        # so that enumeration is deliberately unfiltered.
        enumerate_user_id = cascade_user_id if include_images is True else None
        board_image_names = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
            board_id=board_id,
            categories=None,
            is_intermediate=None,
            user_id=enumerate_user_id,
        )
        board_video_names = ApiDependencies.invoker.services.board_video_records.get_all_board_video_names_for_board(
            board_id=board_id,
            categories=None,
            is_intermediate=None,
            user_id=enumerate_user_id,
        )

        if not ApiDependencies.invoker.services.boards.delete_if_unclaimed(board_id=board_id):
            # `get_dto` above proved the board existed, so it is either claimed now or it vanished
            # in between. Ask again rather than trusting the DTO, which predates any concurrent claim.
            claimed = ApiDependencies.invoker.services.board_records.get_project_ids_for_boards([board_id])
            if board_id in claimed:
                raise HTTPException(
                    status_code=409,
                    detail="This board belongs to a project; delete the project instead",
                )
            raise HTTPException(status_code=404, detail="Board not found")
        board_deleted = True

        if include_images is True:
            # The services report both outcomes: records whose file delete failed are
            # preserved (they are already uncategorized, the board having gone) and returned
            # as failures. This is the ground truth — reconstructing failures by diffing a
            # router-side board listing against the deleted names would double the DB work and
            # misreport items moved or deleted concurrently between the two queries.
            deleted_images, failed_images = ApiDependencies.invoker.services.images.delete_images_by_names(
                board_image_names
            )
            deleted_videos, failed_videos = ApiDependencies.invoker.services.videos.delete_videos_by_names(
                board_video_names
            )
            return DeleteBoardResult(
                board_id=board_id,
                deleted_board_images=[],
                deleted_images=deleted_images,
                deleted_board_videos=[],
                deleted_videos=deleted_videos,
                failed_images=failed_images,
                failed_videos=failed_videos,
            )

        return DeleteBoardResult(
            board_id=board_id,
            deleted_board_images=board_image_names,
            deleted_images=[],
            deleted_board_videos=board_video_names,
            deleted_videos=[],
        )
    except HTTPException:
        raise
    except Exception:
        if include_images is True:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to delete board media",
                    "deleted_images": deleted_images,
                    "deleted_videos": deleted_videos,
                    "board_deleted": board_deleted,
                },
            )
        raise HTTPException(status_code=500, detail="Failed to delete board")


@boards_router.get(
    "/",
    operation_id="list_boards",
    response_model=Union[OffsetPaginatedResults[BoardDTO], list[BoardDTO]],
)
async def list_boards(
    current_user: CurrentUserOrDefault,
    order_by: BoardRecordOrderBy = Query(default=BoardRecordOrderBy.CreatedAt, description="The attribute to order by"),
    direction: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The direction to order by"),
    all: Optional[bool] = Query(default=None, description="Whether to list all boards"),
    offset: Optional[int] = Query(default=None, description="The page offset"),
    limit: Optional[int] = Query(default=None, description="The number of boards per page"),
    include_archived: bool = Query(default=False, description="Whether or not to include archived boards in list"),
) -> Union[OffsetPaginatedResults[BoardDTO], list[BoardDTO]]:
    """Gets a list of boards for the current user, including shared boards. Admin users see all boards."""
    if all:
        return ApiDependencies.invoker.services.boards.get_all(
            current_user.user_id, current_user.is_admin, order_by, direction, include_archived
        )
    elif offset is not None and limit is not None:
        return ApiDependencies.invoker.services.boards.get_many(
            current_user.user_id, current_user.is_admin, order_by, direction, offset, limit, include_archived
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: Must provide either 'all' or both 'offset' and 'limit'",
        )


@boards_router.get(
    "/{board_id}/image_names",
    operation_id="list_all_board_image_names",
    response_model=list[str],
)
async def list_all_board_image_names(
    current_user: CurrentUserOrDefault,
    board_id: str = Path(description="The id of the board or 'none' for uncategorized images"),
    categories: list[ImageCategory] | None = Query(default=None, description="The categories of image to include."),
    is_intermediate: bool | None = Query(default=None, description="Whether to list intermediate images."),
) -> list[str]:
    """Gets a list of images for a board"""

    if board_id != "none":
        _assert_board_read_access(board_id, current_user)

    image_names = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
        board_id,
        categories,
        is_intermediate,
    )

    # For uncategorized images (board_id="none"), filter to only the caller's
    # images so that one user cannot enumerate another's uncategorized images.
    # Admin users can see all uncategorized images.
    if board_id == "none" and not current_user.is_admin:
        image_names = [
            name
            for name in image_names
            if ApiDependencies.invoker.services.image_records.get_user_id(name) == current_user.user_id
        ]

    return image_names
