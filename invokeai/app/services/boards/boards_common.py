from typing import Optional

from pydantic import Field

from invokeai.app.services.board_records.board_records_common import BoardRecord


class BoardDTO(BoardRecord):
    """Deserialized board record with cover image URL and image count."""

    cover_image_name: Optional[str] = Field(description="The name of the board's cover image.")
    """The URL of the thumbnail of the most recent image in the board."""
    cover_video_name: Optional[str] = Field(
        default=None, description="The name of the board's cover video, when the most recent item is a video."
    )
    """The name of the cover video, set when the most-recent item on the board is a video rather than an image."""
    image_count: int = Field(description="The number of images in the board.")
    """The number of images in the board."""
    video_count: int = Field(default=0, description="The number of videos in the board.")
    """The number of videos in the board."""
    asset_count: int = Field(description="The number of assets in the board.")
    """The number of assets in the board."""
    asset_video_count: int = Field(
        default=0, description="The number of asset-category (non-'general') videos in the board."
    )
    """Uploaded videos are assets ('user' category) while generated videos are 'general', mirroring
    images. `video_count` stays the total so "delete board with N videos" copy remains honest; this
    field lets clients split the total across the Media/Assets views."""
    owner_username: Optional[str] = Field(default=None, description="The username of the board owner (for admin view).")
    """The username of the board owner (for admin view)."""
    project_id: Optional[str] = Field(default=None, description="The id of the project that owns this board, if any.")
    """Set when a project owns this board. Such a board can only be renamed or deleted through the
    project APIs; the generic board routes refuse it. Ownership is derived by joining `projects`, so
    it never appears on `BoardRecord` and is never a column on `boards`.

    Note that `BoardRecord` extends `BaseModelExcludeNull`: this field is *omitted* from serialized
    output rather than sent as `null`, so clients must treat absent as "not a project board"."""


def board_record_to_dto(
    board_record: BoardRecord,
    cover_image_name: Optional[str],
    image_count: int,
    asset_count: int,
    owner_username: Optional[str] = None,
    cover_video_name: Optional[str] = None,
    video_count: int = 0,
    asset_video_count: int = 0,
    project_id: Optional[str] = None,
) -> BoardDTO:
    """Converts a board record to a board DTO."""
    return BoardDTO(
        **board_record.model_dump(exclude={"cover_image_name"}),
        cover_image_name=cover_image_name,
        cover_video_name=cover_video_name,
        image_count=image_count,
        video_count=video_count,
        asset_count=asset_count,
        asset_video_count=asset_video_count,
        owner_username=owner_username,
        project_id=project_id,
    )
