from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class InvokeAIMetadata(BaseModel):
    """An image's InvokeAI-specific metadata"""

    session: Optional[str] = Field(description="The session that generated this image")
    source_id: Optional[str] = Field(
        description="The source id of the invocation that generated this image"
    )
    # TODO: figure out metadata
    invocation: Optional[Dict[str, Any]] = Field(
        default={}, description="The prepared invocation that generated this image"
    )


class ImageMetadata(BaseModel):
    """An image's general metadata"""

    created: int = Field(description="The creation timestamp of the image")
    width: int = Field(description="The width of the image in pixels")
    height: int = Field(description="The height of the image in pixels")
    invokeai: Optional[InvokeAIMetadata] = Field(
        default={}, description="The image's InvokeAI-specific metadata"
    )
