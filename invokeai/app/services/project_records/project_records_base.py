from abc import ABC, abstractmethod
from typing import Any

from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardSnapshotDTO,
    ProjectRecordDTO,
    ProjectSummaryDTO,
)


class ProjectRecordsStorageBase(ABC):
    """Storage for per-user workbench project documents.

    All operations are scoped by user_id; a user can never read or write
    another user's projects. Saves use optimistic concurrency via the
    project's monotonic revision.

    Every project owns exactly one private board, and this storage is the only
    thing allowed to create, rename or delete it. The board's name tracks the
    project's name; the two always commit together.
    """

    @abstractmethod
    def create(
        self,
        user_id: str,
        name: str,
        data: dict[str, Any],
        project_id: str | None = None,
        board_id: str | None = None,
    ) -> ProjectRecordDTO:
        """Create a project for the user, with a board.

        Args:
            user_id: The owning user.
            name: The project's display name.
            data: The opaque client-owned project document.
            project_id: Client-generated id (e.g. for imports); generated when omitted.
            board_id: An existing unclaimed private board to adopt, renamed to `name`. Omit to
                create one. Restoration uses this to upload media before the project exists, so
                that creating the project is the single commit point for an import.

        Returns:
            The created project record.

        Raises:
            ProjectRecordExistsError: The user already has a project with this id.
            ProjectBoardNotFoundError: `board_id` is missing or belongs to another user.
            ProjectBoardUnavailableError: The board is public, shared, or already claimed.
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
        """Save a project if the caller's revision is current, renaming its board to match.

        A save that loses the revision race renames nothing.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectRecordConflictError: The stored revision differs from expected_revision.
        """
        pass

    @abstractmethod
    def delete(self, user_id: str, project_id: str) -> None:
        """Delete one of the user's projects and its board.

        Idempotent: deleting a missing project is a no-op. The board's media is not deleted — losing
        the board returns it to Uncategorized.
        """
        pass

    @abstractmethod
    def get_board_snapshot(self, user_id: str, project_id: str) -> ProjectBoardSnapshotDTO:
        """Enumerate everything on the project's board that the gallery would show.

        Excludes intermediates and the canvas's private `other` category, so the result is exactly
        the board as the user sees it. Exports use this to carry a project's whole workspace rather
        than only the media its document references.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
        """
        pass

    @abstractmethod
    def get_board_id(self, user_id: str, project_id: str) -> str:
        """Get the project's board id without loading its document.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
        """
        pass
