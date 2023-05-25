from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from invokeai.app.util.metaenum import MetaEnum


class ImageType(str, Enum, metaclass=MetaEnum):
    """The type of an image."""

    RESULT = "results"
    UPLOAD = "uploads"
    INTERMEDIATE = "intermediates"


class InvalidImageTypeException(ValueError):
    """Raised when a provided value is not a valid ImageType.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid image type."):
        super().__init__(message)


class ImageCategory(str, Enum, metaclass=MetaEnum):
    """The category of an image. Use ImageCategory.OTHER for non-default categories."""

    GENERAL = "general"
    CONTROL = "control"
    MASK = "mask"
    OTHER = "other"


class InvalidImageCategoryException(ValueError):
    """Raised when a provided value is not a valid ImageCategory.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid image category."):
        super().__init__(message)


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
    a: int = Field(ge=0, le=255, description="The alpha component")

    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)
