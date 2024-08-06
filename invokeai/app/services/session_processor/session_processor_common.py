from PIL.Image import Image as PILImageType
from pydantic import BaseModel, Field

from invokeai.backend.util.util import image_to_dataURL


class SessionProcessorStatus(BaseModel):
    is_started: bool = Field(description="Whether the session processor is started")
    is_processing: bool = Field(description="Whether a session is being processed")


class CanceledException(Exception):
    """Execution canceled by user."""

    pass


class ProgressImage(BaseModel):
    """The progress image sent intermittently during processing"""

    width: int = Field(ge=1, description="The effective width of the image in pixels")
    height: int = Field(ge=1, description="The effective height of the image in pixels")
    dataURL: str = Field(description="The image data as a b64 data URL")

    @classmethod
    def build(cls, image: PILImageType, size: tuple[int, int] | None = None) -> "ProgressImage":
        """Build a ProgressImage from a PIL image"""

        return cls(
            width=size[0] if size else image.width,
            height=size[1] if size else image.height,
            dataURL=image_to_dataURL(image, image_format="JPEG"),
        )
