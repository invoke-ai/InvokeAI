import datetime
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.models.resources import ResourceType


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
    image_url: Optional[str] = Field(default=None, description="The URL of the image.")
    thumbnail_url: Optional[str] = Field(
        default=None, description="The thumbnail URL of the image."
    )
