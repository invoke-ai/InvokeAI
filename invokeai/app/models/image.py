from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ImageType(str, Enum):
    RESULT = "results"
    INTERMEDIATE = "intermediates"
    UPLOAD = "uploads"


def is_image_type(obj):
    try:
        ImageType(obj)
    except ValueError:
        return False
    return True


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""

    image_type: ImageType = Field(
        default=ImageType.RESULT, description="The type of the image"
    )
    image_name: Optional[str] = Field(default=None, description="The name of the image")

    class Config:
        schema_extra = {"required": ["image_type", "image_name"]}


class ColorField(BaseModel):
    r: int = Field(ge=0, le=255, description="The red component")
    g: int = Field(ge=0, le=255, description="The green component")
    b: int = Field(ge=0, le=255, description="The blue component")
    a: Optional[int] = Field(default=255, ge=0, le=255, description="The alpha component")
