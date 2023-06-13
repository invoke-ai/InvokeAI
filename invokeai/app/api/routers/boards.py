from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter

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
        result = ApiDependencies.invoker.services.boards.save(board_name=board_name)
        return result
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


