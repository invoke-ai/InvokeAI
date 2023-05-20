from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, Field
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin


class ImageEntity(BaseModel):
    id: str = Field(description="The unique identifier for the image.")
    origin: ResourceOrigin = Field(description="The origin of the image.")
    image_kind: ImageKind = Field(description="The kind of the image.")
    created_at: datetime = Field(description="The created timestamp of the image.")
    session_id: Optional[str] = Field(default=None, description="The session ID.")
    node_id: Optional[str] = Field(default=None, description="The node ID.")
    metadata: Optional[
        Union[GeneratedImageOrLatentsMetadata, UploadedImageOrLatentsMetadata]
    ] = Field(default=None, description="The metadata for the image.")
