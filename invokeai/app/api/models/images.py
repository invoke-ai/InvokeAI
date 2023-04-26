from typing import Optional
from pydantic import BaseModel, Field

from invokeai.app.models.image import ImageType
from invokeai.app.services.metadata import InvokeAIMetadata


class ImageResponseMetadata(BaseModel):
    """An image's metadata. Used only in HTTP responses."""

    created: int = Field(description="The creation timestamp of the image")
    width: int = Field(description="The width of the image in pixels")
    height: int = Field(description="The height of the image in pixels")
    invokeai: Optional[InvokeAIMetadata] = Field(
        description="The image's InvokeAI-specific metadata"
    )


class ImageResponse(BaseModel):
    """The response type for images"""

    image_type: ImageType = Field(description="The type of the image")
    image_name: str = Field(description="The name of the image")
    image_url: str = Field(description="The url of the image")
    thumbnail_url: str = Field(description="The url of the image's thumbnail")
    metadata: ImageResponseMetadata = Field(description="The image's metadata")


class ProgressImage(BaseModel):
    """The progress image sent intermittently during processing"""

    width: int = Field(description="The effective width of the image in pixels")
    height: int = Field(description="The effective height of the image in pixels")
    dataURL: str = Field(description="The image data as a b64 data URL")


class SavedImage(BaseModel):
    image_name: str = Field(description="The name of the saved image")
    thumbnail_name: str = Field(description="The name of the saved thumbnail")
    created: int = Field(description="The created timestamp of the saved image")
