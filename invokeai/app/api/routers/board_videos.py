from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.services.videos_common import AddVideosToBoardResult, RemoveVideosFromBoardResult

board_videos_router = APIRouter(prefix="/v1/board_videos", tags=["boards"])

@board_videos_router.post(
    "/batch",
        operation_id="add_videos_to_board",
    responses={
        201: {"description": "Videos were added to board successfully"},
    },
    status_code=201,
    response_model=AddVideosToBoardResult,
)
async def add_videos_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    video_ids: list[str] = Body(description="The ids of the videos to add", embed=True),
) -> AddVideosToBoardResult:
    """Adds a list of videos to a board"""
    raise HTTPException(status_code=501, detail="Not implemented")


@board_videos_router.post(
    "/batch/delete",
    operation_id="remove_videos_from_board",
    responses={
        201: {"description": "Videos were removed from board successfully"},
    },
    status_code=201,
        response_model=RemoveVideosFromBoardResult,
)
async def remove_videos_from_board(
    video_ids: list[str] = Body(description="The ids of the videos to remove", embed=True),
) -> RemoveVideosFromBoardResult:
    """Removes a list of videos from their board, if they had one"""
    raise HTTPException(status_code=501, detail="Not implemented")
