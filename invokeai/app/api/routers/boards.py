from typing import Optional, Union
from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.services.board_record_storage import BoardChanges
from invokeai.app.services.image_record_storage import OffsetPaginatedResults
from invokeai.app.services.models.board_record import BoardDTO

from ..dependencies import ApiDependencies

boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])


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
        result = ApiDependencies.invoker.services.boards.update(
            board_id=board_id, changes=changes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update board")


@boards_router.delete("/{board_id}", operation_id="delete_board")
async def delete_board(
    board_id: str = Path(description="The id of board to delete"),
) -> None:
    """Deletes a board"""

    try:
        ApiDependencies.invoker.services.boards.delete(board_id=board_id)
    except Exception as e:
        # TODO: Does this need any exception handling at all?
        pass


@boards_router.get(
    "/",
    operation_id="list_boards",
    response_model=Union[OffsetPaginatedResults[BoardDTO], list[BoardDTO]],
)
async def list_boards(
    all: Optional[bool] = Query(default=None, description="Whether to list all boards"),
    offset: Optional[int] = Query(default=None, description="The page offset"),
    limit: Optional[int] = Query(
        default=None, description="The number of boards per page"
    ),
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
