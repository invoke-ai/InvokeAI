from typing import Optional

from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter

from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.videos_common import (
    DeleteVideosResult,
    StarredVideosResult,
    UnstarredVideosResult,
    VideoDTO,
    VideoIdsResult,
    VideoRecordChanges,
)

videos_router = APIRouter(prefix="/v1/videos", tags=["videos"])


@videos_router.patch(
    "/i/{video_id}",
    operation_id="update_video",
    response_model=VideoDTO,
)
async def update_video(
    video_id: str = Path(description="The id of the video to update"),
    video_changes: VideoRecordChanges = Body(description="The changes to apply to the video"),
) -> VideoDTO:
    """Updates a video"""

    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.get(
    "/i/{video_id}",
    operation_id="get_video_dto",
    response_model=VideoDTO,
)
async def get_video_dto(
    video_id: str = Path(description="The id of the video to get"),
) -> VideoDTO:
    """Gets a video's DTO"""

    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.post("/delete", operation_id="delete_videos_from_list", response_model=DeleteVideosResult)
async def delete_videos_from_list(
    video_ids: list[str] = Body(description="The list of ids of videos to delete", embed=True),
) -> DeleteVideosResult:
    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.post("/star", operation_id="star_videos_in_list", response_model=StarredVideosResult)
async def star_videos_in_list(
    video_ids: list[str] = Body(description="The list of ids of videos to star", embed=True),
) -> StarredVideosResult:
    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.post("/unstar", operation_id="unstar_videos_in_list", response_model=UnstarredVideosResult)
async def unstar_videos_in_list(
    video_ids: list[str] = Body(description="The list of ids of videos to unstar", embed=True),
) -> UnstarredVideosResult:
    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.delete("/uncategorized", operation_id="delete_uncategorized_videos", response_model=DeleteVideosResult)
async def delete_uncategorized_videos() -> DeleteVideosResult:
    """Deletes all videos that are uncategorized"""

    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.get("/", operation_id="list_video_dtos", response_model=OffsetPaginatedResults[VideoDTO])
async def list_video_dtos(
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate videos."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of videos per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred videos first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> OffsetPaginatedResults[VideoDTO]:
    """Lists video DTOs"""

    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.get("/ids", operation_id="get_video_ids")
async def get_video_ids(
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate videos."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred videos first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> VideoIdsResult:
    """Gets ordered list of video ids with metadata for optimistic updates"""

    raise HTTPException(status_code=501, detail="Not implemented")


@videos_router.post(
    "/videos_by_ids",
    operation_id="get_videos_by_ids",
    responses={200: {"model": list[VideoDTO]}},
)
async def get_videos_by_ids(
    video_ids: list[str] = Body(embed=True, description="Object containing list of video ids to fetch DTOs for"),
) -> list[VideoDTO]:
    """Gets video DTOs for the specified video ids. Maintains order of input ids."""

    raise HTTPException(status_code=501, detail="Not implemented")
