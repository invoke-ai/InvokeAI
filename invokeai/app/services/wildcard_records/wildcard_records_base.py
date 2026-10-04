from abc import ABC, abstractmethod

from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardRecordDTO,
    WildcardWithoutId,
)


class WildcardRecordsStorageBase(ABC):
    """Base class for wildcard storage services."""

    @abstractmethod
    def get(self, wildcard_id: str) -> WildcardRecordDTO:
        """Gets a wildcard by id. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def create(self, wildcard: WildcardWithoutId, user_id: str) -> WildcardRecordDTO:
        """Creates a wildcard owned by user_id.

        Raises WildcardNameConflictError if the user already owns that name.
        """
        pass

    @abstractmethod
    def update(self, wildcard_id: str, changes: WildcardChanges) -> WildcardRecordDTO:
        """Updates a wildcard. Authorization is the caller's responsibility.

        Raises WildcardNameConflictError if the rename collides with another of the owner's names.
        """
        pass

    @abstractmethod
    def delete(self, wildcard_id: str) -> None:
        """Deletes a wildcard. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def get_many(self, user_id: str) -> list[WildcardRecordDTO]:
        """Gets every wildcard owned by user_id, ordered by name."""
        pass
