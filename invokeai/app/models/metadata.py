from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class InvokeAIMetadata(BaseModel):
    """An image's InvokeAI-specific metadata"""

    session_id: str = Field(description="The session that generated this image")
    invocation: dict = Field(
        default={}, description="The prepared invocation that generated this image"
    )


class ImageMetadata(BaseModel):
    """An image's metadata. Used only in HTTP responses."""

    created: int = Field(description="The creation timestamp of the image")
    width: int = Field(description="The width of the image in pixels")
    height: int = Field(description="The height of the image in pixels")
    mode: str = Field(description="The color mode of the image")
    invokeai: Optional[InvokeAIMetadata] = Field(
        description="The image's InvokeAI-specific metadata"
    )
