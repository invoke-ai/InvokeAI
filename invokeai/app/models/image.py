from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
    width: int = Field(default=None, description="The width of the image in pixels")
    height: int = Field(default=None, description="The height of the image in pixels")
    mode: str = Field(default=None, description="The image mode (ie pixel format)")
    info: dict = Field(default=None, description="The image file's metadata")

    class Config:
        schema_extra = {
            "required": ["image_type", "image_name", "width", "height", "mode", "info"]
        }
