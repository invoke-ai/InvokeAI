from typing import Optional
from pydantic import BaseModel, Field

class ImageMetadata(BaseModel):
    """An image's metadata"""

    timestamp: int = Field(description="The creation timestamp of the image")
    width: int = Field(description="The width of the image in pixels")
    height: int = Field(description="The width of the image in pixels")
    # TODO: figure out metadata
    sd_metadata: Optional[dict] = Field(default={}, description="The image's SD-specific metadata")
