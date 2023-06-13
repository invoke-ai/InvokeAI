from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.services.boards import BoardRecord, BoardRecordChanges
from invokeai.app.services.image_record_storage import OffsetPaginatedResults

from ..dependencies import ApiDependencies

boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])


@boards_router.post(
    "/",
    operation_id="create_board",
    responses={
        201: {"description": "The board was created successfully"},
    },
    status_code=201,
)
async def create_board(
    board_name: str = Body(description="The name of the board to create"),
):
    """Creates a board"""
    try:
        ApiDependencies.invoker.services.boards.save(board_name=board_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create board")


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


@boards_router.patch(
    "/{board_id}",
    operation_id="update_baord"
)
async def update_baord(
    id: str = Path(description="The id of the board to update"),
    board_changes: BoardRecordChanges = Body(
        description="The changes to apply to the board"
    ),
):
    """Updates a board"""

    try:
        return ApiDependencies.invoker.services.boards.update(
            id, board_changes
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to update board")

@boards_router.get(
    "/",
    operation_id="list_boards",
    response_model=OffsetPaginatedResults[BoardRecord],
)
async def list_boards(
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of boards per page"),
) -> OffsetPaginatedResults[BoardRecord]:
    """Gets a list of boards"""

    boards = ApiDependencies.invoker.services.boards.get_many(
        offset,
        limit,
    )

    return boards
