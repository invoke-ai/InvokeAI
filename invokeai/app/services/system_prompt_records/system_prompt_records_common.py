from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, TypeAdapter

EXPAND_PROMPT_MAX_TOKENS_DEFAULT = 300
"""Output-length cap applied to Expand Prompt when the selected system prompt names none.

Lives here rather than in the router so the storage layer, the request model and the migration
that seeds per-prompt caps all agree on one number without the API module having to be imported
(it pulls in the ML stack).
"""

EXPAND_PROMPT_MAX_TOKENS_MIN = 1
EXPAND_PROMPT_MAX_TOKENS_MAX = 2048
"""Bounds a per-prompt cap must fall inside; also the bounds the expand-prompt request enforces."""

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
    max_tokens: Optional[int] = Field(
        default=None,
        ge=EXPAND_PROMPT_MAX_TOKENS_MIN,
        le=EXPAND_PROMPT_MAX_TOKENS_MAX,
        description=(
            "Cap on the tokens the LLM may emit when expanding with this prompt. "
            f"Null means use the default of {EXPAND_PROMPT_MAX_TOKENS_DEFAULT}."
        ),
    )


class SystemPromptChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, min_length=1, description="The new name.")
    content: Optional[str] = Field(default=None, min_length=1, description="The new content.")
    is_public: Optional[bool] = Field(default=None, description="Whether the prompt is shared with all users.")
    max_tokens: Optional[int] = Field(
        default=None,
        ge=EXPAND_PROMPT_MAX_TOKENS_MIN,
        le=EXPAND_PROMPT_MAX_TOKENS_MAX,
        description=(
            "The new output-token cap. Unlike the other fields, an explicitly supplied null is a "
            "change -- it clears the cap back to the default; omitting the field leaves it alone."
        ),
    )


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
