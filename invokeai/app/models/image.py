from enum import Enum
from typing import Optional, Tuple, Literal
from pydantic import BaseModel, Field

from invokeai.app.util.metaenum import MetaEnum
from ..invocations.baseinvocation import (
    BaseInvocationOutput,
    InvocationConfig,
)


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""

    image_name: Optional[str] = Field(default=None, description="The name of the image")

    class Config:
        schema_extra = {"required": ["image_name"]}


class ColorField(BaseModel):
    r: int = Field(ge=0, le=255, description="The red component")
    g: int = Field(ge=0, le=255, description="The green component")
    b: int = Field(ge=0, le=255, description="The blue component")
    a: int = Field(ge=0, le=255, description="The alpha component")

    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)


class ProgressImage(BaseModel):
    """The progress image sent intermittently during processing"""

    width: int = Field(description="The effective width of the image in pixels")
    height: int = Field(description="The effective height of the image in pixels")
    dataURL: str = Field(description="The image data as a b64 data URL")


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["image_output"] = "image_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height"]}


class MaskOutput(BaseInvocationOutput):
    """Base class for invocations that output a mask"""

    # fmt: off
    type: Literal["mask"] = "mask"
    mask:      ImageField = Field(default=None, description="The output mask")
    width:            int = Field(description="The width of the mask in pixels")
    height:           int = Field(description="The height of the mask in pixels")
    # fmt: on

    class Config:
        schema_extra = {
            "required": [
                "type",
                "mask",
            ]
        }


class ResourceOrigin(str, Enum, metaclass=MetaEnum):
    """The origin of a resource (eg image).

    - INTERNAL: The resource was created by the application.
    - EXTERNAL: The resource was not created by the application.
    This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
    """

    INTERNAL = "internal"
    """The resource was created by the application."""
    EXTERNAL = "external"
    """The resource was not created by the application.
    This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
    """


class InvalidOriginException(ValueError):
    """Raised when a provided value is not a valid ResourceOrigin.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid resource origin."):
        super().__init__(message)


class ImageCategory(str, Enum, metaclass=MetaEnum):
    """The category of an image.

    - GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose.
    - MASK: The image is a mask image.
    - CONTROL: The image is a ControlNet control image.
    - USER: The image is a user-provide image.
    - OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes.
    """

    GENERAL = "general"
    """GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose."""
    MASK = "mask"
    """MASK: The image is a mask image."""
    CONTROL = "control"
    """CONTROL: The image is a ControlNet control image."""
    USER = "user"
    """USER: The image is a user-provide image."""
    OTHER = "other"
    """OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes."""


class InvalidImageCategoryException(ValueError):
    """Raised when a provided value is not a valid ImageCategory.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid image category."):
        super().__init__(message)
