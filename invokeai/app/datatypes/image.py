from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from invokeai.app.datatypes.metadata import ImageMetadata


class ImageType(str, Enum):
    RESULT = "results"
    INTERMEDIATE = "intermediates"
    UPLOAD = "uploads"


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""

    image_type: ImageType = Field(
        default=ImageType.RESULT, description="The type of the image"
    )
    image_name: Optional[str] = Field(default=None, description="The name of the image")

    class Config:
        schema_extra = {
            "required": [
                "image_type",
                "image_name",
            ]
        }


class ImageResponse(BaseModel):
    """The response type for images"""

    image_type: ImageType = Field(description="The type of the image")
    image_name: str = Field(description="The name of the image")
    image_url: str = Field(description="The url of the image")
    thumbnail_url: str = Field(description="The url of the image's thumbnail")
    metadata: ImageMetadata = Field(description="The image's metadata")
