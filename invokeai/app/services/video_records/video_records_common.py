import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class VideoRecordNotFoundException(Exception):
    """Raised when a video record is not found."""

    def __init__(self, message="Video record not found"):
        super().__init__(message)


class VideoRecordSaveException(Exception):
    """Raised when a video record cannot be saved."""

    def __init__(self, message="Video record not saved"):
        super().__init__(message)


class VideoRecordDeleteException(Exception):
    """Raised when a video record cannot be deleted."""

    def __init__(self, message="Video record not deleted"):
        super().__init__(message)


VIDEO_DTO_COLS = ", ".join(
    [
        "videos." + c
        for c in [
            "video_name",
            "video_origin",
            "video_category",
            "width",
            "height",
            "duration",
            "fps",
            "session_id",
            "node_id",
            "has_workflow",
            "is_intermediate",
            "created_at",
            "updated_at",
            "deleted_at",
            "starred",
            "video_subfolder",
        ]
    ]
)


class VideoRecord(BaseModelExcludeNull):
    """Deserialized video record without metadata."""

    video_name: str = Field(description="The unique name of the video.")
    video_origin: ResourceOrigin = Field(description="The origin of the video.")
    video_category: ImageCategory = Field(description="The category of the video (reuses ImageCategory).")
    width: int = Field(description="The pixel width of the video.")
    height: int = Field(description="The pixel height of the video.")
    duration: float = Field(description="The duration of the video in seconds.")
    fps: Optional[float] = Field(default=None, description="The frames-per-second of the video, if known.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the video.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the video.")
    deleted_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The deleted timestamp of the video."
    )
    is_intermediate: bool = Field(description="Whether this is an intermediate video.")
    session_id: Optional[str] = Field(default=None, description="The session ID that produced this video, if any.")
    node_id: Optional[str] = Field(default=None, description="The node ID that produced this video, if any.")
    starred: bool = Field(description="Whether this video is starred.")
    has_workflow: bool = Field(description="Whether this video has a workflow associated.")
    video_subfolder: str = Field(default="", description="The subfolder where the video is stored on disk.")


class VideoRecordChanges(BaseModelExcludeNull, extra="allow"):
    """Allowed mutations on a video record."""

    video_category: Optional[ImageCategory] = Field(default=None, description="The video's new category.")
    session_id: Optional[StrictStr] = Field(default=None, description="The video's new session ID.")
    is_intermediate: Optional[StrictBool] = Field(default=None, description="The video's new `is_intermediate` flag.")
    starred: Optional[StrictBool] = Field(default=None, description="The video's new `starred` state.")


def deserialize_video_record(video_dict: dict) -> VideoRecord:
    """Deserializes a video record from a sqlite row dict."""
    video_name = video_dict.get("video_name", "unknown")
    video_origin = ResourceOrigin(video_dict.get("video_origin", ResourceOrigin.INTERNAL.value))
    video_category = ImageCategory(video_dict.get("video_category", ImageCategory.GENERAL.value))
    width = video_dict.get("width", 0)
    height = video_dict.get("height", 0)
    duration = video_dict.get("duration", 0.0)
    fps_raw = video_dict.get("fps", None)
    fps = float(fps_raw) if fps_raw is not None else None
    session_id = video_dict.get("session_id", None)
    node_id = video_dict.get("node_id", None)
    created_at = video_dict.get("created_at", get_iso_timestamp())
    updated_at = video_dict.get("updated_at", get_iso_timestamp())
    deleted_at = video_dict.get("deleted_at", None)
    is_intermediate = video_dict.get("is_intermediate", False)
    starred = video_dict.get("starred", False)
    has_workflow = video_dict.get("has_workflow", False)
    video_subfolder = video_dict.get("video_subfolder", "")

    return VideoRecord(
        video_name=video_name,
        video_origin=video_origin,
        video_category=video_category,
        width=width,
        height=height,
        duration=float(duration),
        fps=fps,
        session_id=session_id,
        node_id=node_id,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        is_intermediate=is_intermediate,
        starred=starred,
        has_workflow=has_workflow,
        video_subfolder=video_subfolder,
    )


class VideoNamesResult(BaseModel):
    """Response containing ordered video names with metadata for optimistic updates."""

    video_names: list[str] = Field(description="Ordered list of video names")
    starred_count: int = Field(description="Number of starred videos (when starred_first=True)")
    total_count: int = Field(description="Total number of videos matching the query")
