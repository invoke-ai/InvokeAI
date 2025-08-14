import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull

VIDEO_DTO_COLS = ", ".join(
    [
        "videos." + c
        for c in [
            "video_id",
            "width",
            "height",
            "session_id",
            "node_id",
            "is_intermediate",
            "created_at",
            "updated_at",
            "deleted_at",
            "starred",
        ]
    ]
)


class VideoRecord(BaseModelExcludeNull):
    """Deserialized video record without metadata."""

    video_id: str = Field(description="The unique id of the video.")
    """The unique id of the video."""
    width: int = Field(description="The width of the video in px.")
    """The actual width of the video in px. This may be different from the width in metadata."""
    height: int = Field(description="The height of the video in px.")
    """The actual height of the video in px. This may be different from the height in metadata."""
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the video.")
    """The created timestamp of the video."""
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the video.")
    """The updated timestamp of the video."""
    deleted_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The deleted timestamp of the video."
    )
    """The deleted timestamp of the video."""
    is_intermediate: bool = Field(description="Whether this is an intermediate video.")
    """Whether this is an intermediate video."""
    session_id: Optional[str] = Field(
        default=None,
        description="The session ID that generated this video, if it is a generated video.",
    )
    """The session ID that generated this video, if it is a generated video."""
    node_id: Optional[str] = Field(
        default=None,
        description="The node ID that generated this video, if it is a generated video.",
    )
    """The node ID that generated this video, if it is a generated video."""
    starred: bool = Field(description="Whether this video is starred.")
    """Whether this video is starred."""


class VideoRecordChanges(BaseModelExcludeNull):
    """A set of changes to apply to a video record.

    Only limited changes are valid:
      - `session_id`: change the session associated with a video
      - `is_intermediate`: change the video's `is_intermediate` flag
      - `starred`: change whether the video is starred
    """

    session_id: Optional[StrictStr] = Field(
        default=None,
        description="The video's new session ID.",
    )
    """The video's new session ID."""
    is_intermediate: Optional[StrictBool] = Field(default=None, description="The video's new `is_intermediate` flag.")
    """The video's new `is_intermediate` flag."""
    starred: Optional[StrictBool] = Field(default=None, description="The video's new `starred` state")
    """The video's new `starred` state."""


def deserialize_video_record(video_dict: dict) -> VideoRecord:
    """Deserializes a video record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.
    video_id = video_dict.get("video_id", "unknown")
    width = video_dict.get("width", 0)
    height = video_dict.get("height", 0)
    session_id = video_dict.get("session_id", None)
    node_id = video_dict.get("node_id", None)
    created_at = video_dict.get("created_at", get_iso_timestamp())
    updated_at = video_dict.get("updated_at", get_iso_timestamp())
    deleted_at = video_dict.get("deleted_at", get_iso_timestamp())
    is_intermediate = video_dict.get("is_intermediate", False)
    starred = video_dict.get("starred", False)

    return VideoRecord(
        video_id=video_id,
        width=width,
        height=height,
        session_id=session_id,
        node_id=node_id,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        is_intermediate=is_intermediate,
        starred=starred,
    )


class VideoCollectionCounts(BaseModel):
    starred_count: int = Field(description="The number of starred videos in the collection.")
    unstarred_count: int = Field(description="The number of unstarred videos in the collection.")


class VideoIdsResult(BaseModel):
    """Response containing ordered video ids with metadata for optimistic updates."""

    video_ids: list[str] = Field(description="Ordered list of video ids")
    starred_count: int = Field(description="Number of starred videos (when starred_first=True)")
    total_count: int = Field(description="Total number of videos matching the query")


class VideoUrlsDTO(BaseModelExcludeNull):
    """The URLs for an image and its thumbnail."""

    video_id: str = Field(description="The unique id of the video.")
    """The unique id of the video."""
    video_url: str = Field(description="The URL of the video.")
    """The URL of the video."""
    thumbnail_url: str = Field(description="The URL of the video's thumbnail.")
    """The URL of the video's thumbnail."""


class VideoDTO(VideoRecord, VideoUrlsDTO):
    """Deserialized video record, enriched for the frontend."""

    board_id: Optional[str] = Field(
        default=None, description="The id of the board the image belongs to, if one exists."
    )
    """The id of the board the image belongs to, if one exists."""


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


class ResultWithAffectedBoards(BaseModel):
    affected_boards: list[str] = Field(description="The ids of boards affected by the delete operation")


class DeleteVideosResult(ResultWithAffectedBoards):
    deleted_videos: list[str] = Field(description="The ids of the videos that were deleted")


class StarredVideosResult(ResultWithAffectedBoards):
    starred_videos: list[str] = Field(description="The ids of the videos that were starred")


class UnstarredVideosResult(ResultWithAffectedBoards):
    unstarred_videos: list[str] = Field(description="The ids of the videos that were unstarred")


class AddVideosToBoardResult(ResultWithAffectedBoards):
    added_videos: list[str] = Field(description="The video ids that were added to the board")


class RemoveVideosFromBoardResult(ResultWithAffectedBoards):
    removed_videos: list[str] = Field(description="The video ids that were removed from their board")
