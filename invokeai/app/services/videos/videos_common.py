from typing import Optional

from pydantic import BaseModel, Field

from invokeai.app.services.video_records.video_records_common import VideoRecord
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class VideoUrlsDTO(BaseModelExcludeNull):
    """The URLs for a video and its thumbnail."""

    video_name: str = Field(description="The unique name of the video.")
    video_url: str = Field(description="The URL of the video file (MP4).")
    thumbnail_url: str = Field(description="The URL of the video's first-frame thumbnail (WebP).")


class VideoDTO(VideoRecord, VideoUrlsDTO):
    """Deserialized video record, enriched for the frontend."""

    board_id: Optional[str] = Field(
        default=None, description="The id of the board the video belongs to, if one exists."
    )


def video_record_to_dto(
    video_record: VideoRecord,
    video_url: str,
    thumbnail_url: str,
    board_id: Optional[str],
) -> VideoDTO:
    """Converts a video record to a video DTO."""
    return VideoDTO(
        **video_record.model_dump(),
        video_url=video_url,
        thumbnail_url=thumbnail_url,
        board_id=board_id,
    )


class VideoResultWithAffectedBoards(BaseModel):
    affected_boards: list[str] = Field(description="The ids of boards affected by the operation")


class DeleteVideosResult(VideoResultWithAffectedBoards):
    deleted_videos: list[str] = Field(description="The names of the videos that were deleted")
    failed_videos: list[str] = Field(description="The names of videos that were not deleted")


class StarredVideosResult(VideoResultWithAffectedBoards):
    starred_videos: list[str] = Field(description="The names of the videos that were starred")


class UnstarredVideosResult(VideoResultWithAffectedBoards):
    unstarred_videos: list[str] = Field(description="The names of the videos that were unstarred")


class AddVideosToBoardResult(VideoResultWithAffectedBoards):
    added_videos: list[str] = Field(description="The video names that were added to the board")


class RemoveVideosFromBoardResult(VideoResultWithAffectedBoards):
    removed_videos: list[str] = Field(description="The video names that were removed from their board")
