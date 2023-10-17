from typing import Optional

from pydantic import Field

from invokeai.app.services.image_records.image_records_common import ImageRecord
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class ImageUrlsDTO(BaseModelExcludeNull):
    """The URLs for an image and its thumbnail."""

    image_name: str = Field(description="The unique name of the image.")
    """The unique name of the image."""
    image_url: str = Field(description="The URL of the image.")
    """The URL of the image."""
    thumbnail_url: str = Field(description="The URL of the image's thumbnail.")
    """The URL of the image's thumbnail."""


class ImageDTO(ImageRecord, ImageUrlsDTO):
    """Deserialized image record, enriched for the frontend."""

    board_id: Optional[str] = Field(
        default=None, description="The id of the board the image belongs to, if one exists."
    )
    """The id of the board the image belongs to, if one exists."""


def image_record_to_dto(
    image_record: ImageRecord,
    image_url: str,
    thumbnail_url: str,
    board_id: Optional[str],
) -> ImageDTO:
    """Converts an image record to an image DTO."""
    return ImageDTO(
        **image_record.model_dump(),
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        board_id=board_id,
    )
