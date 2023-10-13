from enum import Enum

from pydantic import BaseModel, Field

from invokeai.app.util.metaenum import MetaEnum


class ProgressImage(BaseModel):
    """The progress image sent intermittently during processing"""

    width: int = Field(description="The effective width of the image in pixels")
    height: int = Field(description="The effective height of the image in pixels")
    dataURL: str = Field(description="The image data as a b64 data URL")


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
