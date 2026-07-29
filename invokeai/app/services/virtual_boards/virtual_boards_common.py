from typing import Optional

from pydantic import BaseModel, Field


class VirtualSubBoardDTO(BaseModel):
    """A virtual sub-board computed from image/video metadata, not stored in the database."""

    virtual_board_id: str = Field(description="The virtual board ID, e.g. 'by_date:2026-03-18'.")
    board_name: str = Field(description="The display name of the virtual sub-board, e.g. '2026-03-18'.")
    date: str = Field(description="The ISO date string, e.g. '2026-03-18'.")
    image_count: int = Field(description="The number of general images for this date.")
    asset_count: int = Field(description="The number of asset images for this date.")
    video_count: int = Field(default=0, description="The number of videos for this date.")
    cover_image_name: Optional[str] = Field(default=None, description="The most recent image name for this date.")
    cover_video_name: Optional[str] = Field(
        default=None,
        description="The most recent video name for this date. Set instead of cover_image_name when the newest item for the date is a video.",
    )
