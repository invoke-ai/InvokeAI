from pydantic import BaseModel, Field


class SessionProcessorStatus(BaseModel):
    is_started: bool = Field(description="Whether the session processor is started")
    is_processing: bool = Field(description="Whether a session is being processed")


class CanceledException(Exception):
    """Execution canceled by user."""

    pass


class ProgressImage(BaseModel):
    """The progress image sent intermittently during processing"""

    width: int = Field(description="The effective width of the image in pixels")
    height: int = Field(description="The effective height of the image in pixels")
    dataURL: str = Field(description="The image data as a b64 data URL")
