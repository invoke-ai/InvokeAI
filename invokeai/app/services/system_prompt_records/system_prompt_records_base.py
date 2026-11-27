from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptRecordDTO,
    SystemPromptWithoutId,
)


class SystemPromptRecordsStorageBase(ABC):
    """Base class for system prompt storage services."""

    @abstractmethod
    def get(self, system_prompt_id: str) -> SystemPromptRecordDTO:
        """Get system prompt by id (no permission check; caller must enforce)."""
        pass

    @abstractmethod
    def create(
        self,
        system_prompt: SystemPromptWithoutId,
        user_id: str,
        is_public: bool = False,
    ) -> SystemPromptRecordDTO:
        """Creates a system prompt owned by `user_id`."""
        pass

    @abstractmethod
    def update(
        self,
        system_prompt_id: str,
        changes: SystemPromptChanges,
        user_id: Optional[str] = None,
    ) -> SystemPromptRecordDTO:
        """Updates a system prompt. When `user_id` is provided, only rows owned by that user are touched."""
        pass

    @abstractmethod
    def delete(self, system_prompt_id: str, user_id: Optional[str] = None) -> None:
        """Deletes a system prompt. When `user_id` is provided, only rows owned by that user are deleted."""
        pass

    @abstractmethod
    def get_many(self, user_id: Optional[str] = None) -> list[SystemPromptRecordDTO]:
        """Lists system prompts.

        When `user_id` is given, returns prompts owned by that user OR shared (is_public=TRUE).
        When `user_id` is None, returns all prompts (admin / single-user view).
        """
        pass
