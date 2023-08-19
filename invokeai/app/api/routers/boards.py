from typing import Optional, Union

from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.services.board_record_storage import BoardChanges
from invokeai.app.services.image_record_storage import OffsetPaginatedResults
from invokeai.app.services.models.board_record import BoardDTO

from ..dependencies import ApiDependencies

boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])


class DeleteBoardResult(BaseModel):
    board_id: str = Field(description="The id of the board that was deleted.")
    deleted_board_images: list[str] = Field(
        description="The image names of the board-images relationships that were deleted."
    )
    deleted_images: list[str] = Field(description="The names of the images that were deleted.")


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
    board_name: str = Query(description="The name of the board to create"),
) -> BoardDTO:
    """Creates a board"""
    try:
        result = ApiDependencies.invoker.services.boards.create(board_name=board_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create board")


@boards_router.get("/{board_id}", operation_id="get_board", response_model=BoardDTO)
async def get_board(
    board_id: str = Path(description="The id of board to get"),
) -> BoardDTO:
    """Gets a board"""

    try:
        result = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
        return result
    except Exception as e:
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
    board_id: str = Path(description="The id of board to update"),
    changes: BoardChanges = Body(description="The changes to apply to the board"),
) -> BoardDTO:
    """Updates a board"""
    try:
        result = ApiDependencies.invoker.services.boards.update(board_id=board_id, changes=changes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update board")


@boards_router.delete("/{board_id}", operation_id="delete_board", response_model=DeleteBoardResult)
async def delete_board(
    board_id: str = Path(description="The id of board to delete"),
    include_images: Optional[bool] = Query(description="Permanently delete all images on the board", default=False),
) -> DeleteBoardResult:
    """Deletes a board"""
    try:
        if include_images is True:
            deleted_images = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
                board_id=board_id
            )
            ApiDependencies.invoker.services.images.delete_images_on_board(board_id=board_id)
            ApiDependencies.invoker.services.boards.delete(board_id=board_id)
            return DeleteBoardResult(
                board_id=board_id,
                deleted_board_images=[],
                deleted_images=deleted_images,
            )
        else:
            deleted_board_images = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
                board_id=board_id
            )
            ApiDependencies.invoker.services.boards.delete(board_id=board_id)
            return DeleteBoardResult(
                board_id=board_id,
                deleted_board_images=deleted_board_images,
                deleted_images=[],
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete board")


@boards_router.get(
    "/",
    operation_id="list_boards",
    response_model=Union[OffsetPaginatedResults[BoardDTO], list[BoardDTO]],
)
async def list_boards(
    all: Optional[bool] = Query(default=None, description="Whether to list all boards"),
    offset: Optional[int] = Query(default=None, description="The page offset"),
    limit: Optional[int] = Query(default=None, description="The number of boards per page"),
) -> Union[OffsetPaginatedResults[BoardDTO], list[BoardDTO]]:
    """Gets a list of boards"""
    if all:
        return ApiDependencies.invoker.services.boards.get_all()
    elif offset is not None and limit is not None:
        return ApiDependencies.invoker.services.boards.get_many(
            offset,
            limit,
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
    board_id: str = Path(description="The id of the board"),
) -> list[str]:
    """Gets a list of images for a board"""

    image_names = ApiDependencies.invoker.services.board_images.get_all_board_image_names_for_board(
        board_id,
    )
    return image_names
