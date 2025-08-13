import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.util.model_exclude_null import BaseModelExcludeNull

VIDEO_DTO_COLS = ", ".join(
    [
        "videos." + c
        for c in [
            "id",
            "width",
            "height",
            "created_at",
            "updated_at",
        ]
    ]
)


class VideoRecord(BaseModelExcludeNull):
    """Deserialized video record without metadata."""

    id: str = Field(description="The unique id of the video.")
    """The unique id of the video."""
    width: int = Field(description="The width of the video in px.")
    """The actual width of the video in px."""
    height: int = Field(description="The height of the video in px.")
    """The actual height of the video in px."""
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the video.")
    """The created timestamp of the video."""
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the video.")
    """The updated timestamp of the video."""


class VideoRecordChanges(BaseModelExcludeNull):
    """
    A set of changes to apply to a video record.

    Only limited changes are allowed:
    - `starred` - Whether the video is starred.
    """

    starred: Optional[bool] = Field(default=None, description="Whether the video is starred.")
    """The video's new `starred` state."""


class VideoNamesResult(BaseModel):
    """Result of fetching video names."""

    video_ids: list[str] = Field(description="The video IDs")



