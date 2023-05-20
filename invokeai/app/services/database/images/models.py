from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin


class ImageEntity(BaseModel):
    id: str = Field(description="The unique identifier for the image.")
    image_kind: ImageKind = Field(description="The kind of the image.")
    created_at: datetime = Field(description="The created timestamp of the image.")


GENERATED_IMAGE_ORIGIN = Literal[ResourceOrigin.RESULTS, ResourceOrigin.INTERMEDIATES]
UPLOADED_IMAGE_ORIGIN = Literal[ResourceOrigin.UPLOADS]


class GeneratedImageEntity(ImageEntity):
    """Deserialized generated (eg result or intermediate) images DB entity."""

    origin: GENERATED_IMAGE_ORIGIN = Field(description="The origin of the image.")
    session_id: str = Field(description="The session ID.")
    node_id: str = Field(description="The node ID.")
    metadata: GeneratedImageOrLatentsMetadata = Field(
        description="The metadata for the image."
    )


class UploadedImageEntity(ImageEntity):
    """Deserialized uploaded images DB entity."""

    origin: UPLOADED_IMAGE_ORIGIN = Field(description="The origin of the image.")
    metadata: UploadedImageOrLatentsMetadata = Field(
        description="The metadata for the image."
    )
