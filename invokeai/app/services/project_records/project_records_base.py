from abc import ABC, abstractmethod
from typing import Any, Literal

from invokeai.app.services.project_records.project_records_common import (
    DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ProjectBoardSnapshotDTO,
    ProjectRecordDTO,
    ProjectSummaryDTO,
)


class ProjectRecordsStorageBase(ABC):
    """Storage for per-user workbench project documents.

    All operations are scoped by user_id; a user can never read or write
    another user's projects. Saves use optimistic concurrency via the
    project's monotonic revision.

    Every project has exactly one inbox: a private board that this storage alone
    creates, renames and deletes. Its name tracks the project's name; the two
    always commit together. The project's other boards are ordinary boards whose
    `project_id` names it; they are created and moved through the board service
    and only leave with the project, released to the Library or deleted.
    """

    @abstractmethod
    def create(
        self,
        user_id: str,
        name: str,
        data: dict[str, Any],
        project_id: str | None = None,
        board_id: str | None = None,
        minimum_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectRecordDTO:
        """Create a project for the user, with its inbox.

        Args:
            user_id: The owning user.
            name: The project's display name.
            data: The opaque client-owned project document.
            project_id: Client-generated id (e.g. for imports); generated when omitted.
            board_id: An existing unclaimed private Library board to adopt as the inbox, renamed
                to `name`. Omit to create one. Restoration uses this to upload media before the
                project exists, so that creating the project is the single commit point for an
                import.
            minimum_canvas_schema_version: Compatibility floor stored with the project.
            max_canvas_schema_version: Newest canvas schema understood by the caller.

        Returns:
            The created project record.

        Raises:
            ProjectRecordExistsError: The user already has a project with this id.
            ProjectBoardNotFoundError: `board_id` is missing or belongs to another user.
            ProjectBoardUnavailableError: The board is public, shared, or already claimed.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the requested schema.
            ProjectDocumentInvalidError: The document is not standards-compliant UTF-8 JSON.
            ProjectDocumentTooLargeError: The project document exceeds the storage limit.
        """
        pass

    @abstractmethod
    def get(
        self,
        user_id: str,
        project_id: str,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectRecordDTO:
        """Get one of the user's projects, including its document.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the stored schema.
        """
        pass

    @abstractmethod
    def list(self, user_id: str) -> list[ProjectSummaryDTO]:
        """List the user's projects as lightweight summaries, oldest first."""
        pass

    @abstractmethod
    def update(
        self,
        user_id: str,
        project_id: str,
        expected_revision: int,
        name: str,
        data: dict[str, Any],
        minimum_canvas_schema_version: int | None = None,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectRecordDTO:
        """Save a project if the caller's revision is current, renaming its inbox to match.

        A save that loses the revision race renames nothing.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectRecordConflictError: The stored revision differs from expected_revision.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the stored or requested schema.
            ProjectCanvasSchemaDowngradeError: The save tries to lower the stored compatibility floor.
            ProjectDocumentInvalidError: The document is not standards-compliant UTF-8 JSON.
            ProjectDocumentTooLargeError: The project document exceeds the storage limit.
        """
        pass

    @abstractmethod
    def delete(self, user_id: str, project_id: str, boards: Literal["release", "delete"] = "release") -> None:
        """Delete one of the user's projects and its inbox, in one transaction.

        Idempotent: deleting a missing project is a no-op. The project's other boards are either
        released to the Library, where they keep their media, or deleted with the project. No media
        is ever deleted here — losing a board returns its items to Uncategorized.
        """
        pass

    @abstractmethod
    def get_board_snapshot(self, user_id: str, project_id: str) -> ProjectBoardSnapshotDTO:
        """Enumerate everything on the project's inbox that the gallery would show.

        Excludes intermediates and the canvas's private `other` category, so the result is exactly
        the board as the user sees it. Exports use this to carry a project's whole workspace rather
        than only the media its document references.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
        """
        pass

    @abstractmethod
    def get_board_id(self, user_id: str, project_id: str) -> str:
        """Get the project's inbox board id without loading its document.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
        """
        pass
