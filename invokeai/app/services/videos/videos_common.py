from typing import Optional

from pydantic import BaseModel, Field

from invokeai.app.services.video_records.video_records_common import VideoRecord



class VideoDTO(BaseModel):
    """Deserialized video record, enriched for the frontend."""

    video_id: Optional[str] = Field(
        default=None, description="The id of the board the video belongs to, if one exists."
    )
    """The id of the board the video belongs to, if one exists."""
    width: int = Field(description="The width of the video.")
    height: int = Field(description="The height of the video.")
    board_id: Optional[str] = Field(
        default=None, description="The id of the board the video belongs to, if one exists."
    )


def video_record_to_dto(
    video_record: VideoRecord,
    board_id: Optional[str],
) -> VideoDTO:
    """Converts a VideoRecord to a VideoDTO."""
    return VideoDTO(
        **video_record.model_dump(),
        board_id=board_id,
    )


