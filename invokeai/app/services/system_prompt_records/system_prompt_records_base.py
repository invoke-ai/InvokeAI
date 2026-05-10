from abc import ABC, abstractmethod

from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptRecordDTO,
    SystemPromptWithoutId,
)


class SystemPromptRecordsStorageBase(ABC):
    """Base class for system prompt storage services."""

    @abstractmethod
    def get(self, system_prompt_id: str) -> SystemPromptRecordDTO:
        """Get system prompt by id."""
        pass

    @abstractmethod
    def create(self, system_prompt: SystemPromptWithoutId) -> SystemPromptRecordDTO:
        """Creates a system prompt."""
        pass

    @abstractmethod
    def update(self, system_prompt_id: str, changes: SystemPromptChanges) -> SystemPromptRecordDTO:
        """Updates a system prompt."""
        pass

    @abstractmethod
    def delete(self, system_prompt_id: str) -> None:
        """Deletes a system prompt."""
        pass

    @abstractmethod
    def get_many(self) -> list[SystemPromptRecordDTO]:
        """Gets all system prompts."""
        pass
