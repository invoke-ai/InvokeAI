import datetime
import sqlite3
from typing import Optional, Union
from pydantic import BaseModel, Field
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.util.misc import get_iso_timestamp


class ImageRecord(BaseModel):
    """Deserialized image record."""

    image_name: str = Field(description="The name of the image.")
    image_type: ImageType = Field(description="The type of the image.")
    image_category: ImageCategory = Field(description="The category of the image.")
    created_at: Union[datetime.datetime, str] = Field(
        description="The created timestamp of the image."
    )
    session_id: Optional[str] = Field(default=None, description="The session ID.")
    node_id: Optional[str] = Field(default=None, description="The node ID.")
    metadata: Optional[ImageMetadata] = Field(
        default=None, description="The image's metadata."
    )


class ImageUrlsDTO(BaseModel):
    """The URLs for an image and its thumbnaill"""

    image_name: str = Field(description="The name of the image.")
    image_type: ImageType = Field(description="The type of the image.")
    image_url: str = Field(description="The URL of the image.")
    thumbnail_url: str = Field(description="The thumbnail URL of the image.")


class ImageDTO(ImageRecord, ImageUrlsDTO):
    """Deserialized image record with URLs."""

    pass


def image_record_to_dto(
    image_record: ImageRecord, image_url: str, thumbnail_url: str
) -> ImageDTO:
    """Converts an image record to an image DTO."""
    return ImageDTO(
        image_name=image_record.image_name,
        image_type=image_record.image_type,
        image_category=image_record.image_category,
        created_at=image_record.created_at,
        session_id=image_record.session_id,
        node_id=image_record.node_id,
        metadata=image_record.metadata,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
    )


def deserialize_image_record(image_row: sqlite3.Row) -> ImageRecord:
    """Deserializes an image record."""

    image_dict = dict(image_row)

    image_type = ImageType(image_dict.get("image_type", ImageType.RESULT.value))

    raw_metadata = image_dict.get("metadata", "{}")

    metadata = ImageMetadata.parse_raw(raw_metadata)

    return ImageRecord(
        image_name=image_dict.get("id", "unknown"),
        session_id=image_dict.get("session_id", None),
        node_id=image_dict.get("node_id", None),
        metadata=metadata,
        image_type=image_type,
        image_category=ImageCategory(
            image_dict.get("image_category", ImageCategory.IMAGE.value)
        ),
        created_at=image_dict.get("created_at", get_iso_timestamp()),
    )
