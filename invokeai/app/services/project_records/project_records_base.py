from abc import ABC, abstractmethod
from typing import Any

from invokeai.app.services.project_records.project_records_common import ProjectRecordDTO, ProjectSummaryDTO


class ProjectRecordsStorageBase(ABC):
    """Storage for per-user workbench project documents.

    All operations are scoped by user_id; a user can never read or write
    another user's projects. Saves use optimistic concurrency via the
    project's monotonic revision.
    """

    @abstractmethod
    def create(self, user_id: str, name: str, data: dict[str, Any], project_id: str | None = None) -> ProjectRecordDTO:
        """Create a project for the user.

        Args:
            user_id: The owning user.
            name: The project's display name.
            data: The opaque client-owned project document.
            project_id: Client-generated id (e.g. for imports); generated when omitted.

        Returns:
            The created project record.

        Raises:
            ProjectRecordExistsError: The user already has a project with this id.
        """
        pass

    @abstractmethod
    def get(self, user_id: str, project_id: str) -> ProjectRecordDTO:
        """Get one of the user's projects, including its document.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
        """
        pass

    @abstractmethod
    def list(self, user_id: str) -> list[ProjectSummaryDTO]:
        """List the user's projects as lightweight summaries, oldest first."""
        pass

    @abstractmethod
    def update(
        self, user_id: str, project_id: str, expected_revision: int, name: str, data: dict[str, Any]
    ) -> ProjectRecordDTO:
        """Save a project if the caller's revision is current.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectRecordConflictError: The stored revision differs from expected_revision.
        """
        pass

    @abstractmethod
    def delete(self, user_id: str, project_id: str) -> None:
        """Delete one of the user's projects. Idempotent: deleting a missing project is a no-op."""
        pass
