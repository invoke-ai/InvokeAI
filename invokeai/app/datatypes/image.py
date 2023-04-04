from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ImageType(str, Enum):
    RESULT = "results"
    INTERMEDIATE = "intermediates"
    UPLOAD = "uploads"


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""

    image_type: str = Field(
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
