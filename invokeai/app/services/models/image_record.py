import datetime
from typing import Optional, Union
from pydantic import BaseModel, Field
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import ImageCategory, ImageType


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
    metadata: Optional[
        Union[GeneratedImageOrLatentsMetadata, UploadedImageOrLatentsMetadata]
    ] = Field(default=None, description="The image's metadata.")


class ImageDTO(ImageRecord):
    """Deserialized image record with URLs."""

    image_url: str = Field(description="The URL of the image.")
    thumbnail_url: str = Field(description="The thumbnail URL of the image.")


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
