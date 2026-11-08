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
    def delete_many(self, user_id: str, wildcard_ids: list[str]) -> list[str]:
        """Deletes the wildcards among wildcard_ids that user_id owns, in one transaction, and returns their ids.

        Unknown ids and other accounts' ids are skipped, so the result never confirms that they exist.
        """
        pass

    @abstractmethod
    def get_many(self, user_id: str) -> list[WildcardRecordDTO]:
        """Gets every wildcard owned by user_id, ordered by name."""
        pass
