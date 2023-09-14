from pydantic import BaseModel, Field


class SessionExecutionStatusResult(BaseModel):
    started: bool = Field(..., description="Whether the session queue is running")
