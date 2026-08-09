from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, TypeAdapter

SYSTEM_PROMPT_DEFAULT_USER_ID = "system"
"""Owner of the seeded default prompts.

Do NOT use this as an "is a built-in default" test. It is also the synthetic user id every
request carries in single-user mode (`auth_dependencies.get_current_user`), so on an install
that later switched to multiuser it owns ordinary user-created prompts too. Visibility is
decided by `is_public` alone -- the seeded defaults are seeded with `is_public=TRUE`.
"""


class SystemPromptNotFoundError(Exception):
    """Raised when a system prompt is not found"""


class SystemPromptNotAuthorizedError(Exception):
    """Raised when the current user is not allowed to access or mutate a prompt."""


class SystemPromptWithoutId(BaseModel, extra="forbid"):
    name: str = Field(min_length=1, description="The name of the system prompt.")
    content: str = Field(min_length=1, description="The system prompt content.")


class SystemPromptChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, min_length=1, description="The new name.")
    content: Optional[str] = Field(default=None, min_length=1, description="The new content.")
    is_public: Optional[bool] = Field(default=None, description="Whether the prompt is shared with all users.")


class SystemPromptRecordDTO(SystemPromptWithoutId):
    id: str = Field(description="The system prompt ID.")
    user_id: str = Field(
        description="The owning user id ('system' for built-in defaults, and for everything created in single-user mode)."
    )
    is_public: bool = Field(description="Whether the prompt is shared with all users.")
    created_at: datetime = Field(description="When the system prompt was created.")
    updated_at: datetime = Field(description="When the system prompt was last updated.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPromptRecordDTO":
        return SystemPromptRecordDTOValidator.validate_python(data)


SystemPromptRecordDTOValidator = TypeAdapter(SystemPromptRecordDTO)
