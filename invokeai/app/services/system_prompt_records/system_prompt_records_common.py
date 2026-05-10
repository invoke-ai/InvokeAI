from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, TypeAdapter


class SystemPromptNotFoundError(Exception):
    """Raised when a system prompt is not found"""


class SystemPromptWithoutId(BaseModel, extra="forbid"):
    name: str = Field(min_length=1, description="The name of the system prompt.")
    content: str = Field(min_length=1, description="The system prompt content.")


class SystemPromptChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, min_length=1, description="The new name.")
    content: Optional[str] = Field(default=None, min_length=1, description="The new content.")


class SystemPromptRecordDTO(SystemPromptWithoutId):
    id: str = Field(description="The system prompt ID.")
    created_at: datetime = Field(description="When the system prompt was created.")
    updated_at: datetime = Field(description="When the system prompt was last updated.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPromptRecordDTO":
        return SystemPromptRecordDTOValidator.validate_python(data)


SystemPromptRecordDTOValidator = TypeAdapter(SystemPromptRecordDTO)
