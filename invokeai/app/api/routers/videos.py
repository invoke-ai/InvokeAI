from typing import Optional

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter

from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecordChanges,
)
from invokeai.app.services.videos.videos_common import (
    VideoDTO,
)

videos_router = APIRouter(prefix="/v1/videos", tags=["videos"])


# videos are immutable; set a high max-age
VIDEO_MAX_AGE = 31536000


@videos_router.get(
    "/i/{video_id}",
    operation_id="get_video_dto",
    response_model=VideoDTO,
)
async def get_video_dto(
    video_id: str = Path(description="The id of the video to get"),
) -> VideoDTO:
    """Gets a video's DTO"""

    raise NotImplementedError("Not implemented")


@videos_router.get(
    "/",
    operation_id="list_video_dtos",
    response_model=OffsetPaginatedResults[VideoDTO],
)
async def list_video_dtos(
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
    """Gets a list of video DTOs"""

    raise NotImplementedError("Not implemented")


@videos_router.get("/ids", operation_id="get_video_ids")
async def get_video_ids(
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred videos first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> VideoNamesResult:
    """Gets ordered list of video names with metadata for optimistic updates"""

    raise NotImplementedError("Not implemented")


@videos_router.post(
    "/videos_by_ids",
    operation_id="get_videos_by_ids",
    responses={200: {"model": list[VideoDTO]}},
)
async def get_videos_by_ids(
    video_ids: list[str] = Body(embed=True, description="Object containing list of video ids to fetch DTOs for"),
) -> list[VideoDTO]:
    """Gets video DTOs for the specified video ids. Maintains order of input ids."""

    raise NotImplementedError("Not implemented")


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

    raise NotImplementedError("Not implemented")


@videos_router.get(
    "/i/{video_id}/thumbnail",
    operation_id="get_video_thumbnail",
    response_class=Response,
    responses={
        200: {
            "description": "Return the video thumbnail",
            "content": {"image/webp": {}},
        },
        404: {"description": "Video not found"},
    },
)
async def get_video_thumbnail(
    video_id: str = Path(description="The id of video to get thumbnail for"),
) -> Response:
    """Gets a video thumbnail file"""

    raise NotImplementedError("Not implemented")
