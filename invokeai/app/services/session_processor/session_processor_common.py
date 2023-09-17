from pydantic import BaseModel, Field


class SessionProcessorStatus(BaseModel):
    is_started: bool = Field(description="Whether the session processor is started")
    is_processing: bool = Field(description="Whether a session is being processed")
