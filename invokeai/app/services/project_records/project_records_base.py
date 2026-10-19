from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from invokeai.app.services.project_records.project_records_common import (
    DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
    ProjectBoardSnapshotDTO,
    ProjectSummaryDTO,
    ProjectUpdateWorkspaceRequest,
    ProjectWithWorkspaceDTO,
    ProjectWorkspaceDTO,
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

    From document schema 4 a project is two documents: its content, which the
    project saves, and each member's workspace in it, saved apart with a revision
    of its own. The storage parses neither beyond the media they name.
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
        minimum_document_schema_version: int = DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
        max_document_schema_version: int = DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
        workspace: Optional[dict[str, Any]] = None,
    ) -> ProjectWithWorkspaceDTO:
        """Create a project for the user, with its inbox and, when `workspace` is given, the user's workspace in
        it, in the same transaction.

        The workspace is checked as `save_workspace` checks it, except that `max_canvas_schema_version` covers it,
        and it is always the user's first row in the project. A refused create leaves no project, board, workspace
        or media reference behind.

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
            minimum_document_schema_version: Document schema floor stored with the project.
            max_document_schema_version: Newest document schema understood by the caller.
            workspace: The user's first workspace in the project, stored with the content; omitted to create the
                project without one.

        Returns:
            The created project record, with the workspace this create stored, or None when it carried none.

        Raises:
            ProjectRecordExistsError: The user already has a project with this id.
            ProjectWorkspaceBeforeSplitError: `workspace` is given, but `minimum_document_schema_version` is below
                the split schema, where a project is content and workspaces.
            ProjectBoardNotFoundError: `board_id` is missing or belongs to another user.
            ProjectBoardUnavailableError: The board is public, shared, or already claimed.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the requested schema.
            ProjectDocumentSchemaUnsupportedError: The caller does not understand the requested document schema.
            ProjectDocumentInvalidError: The document or the workspace is not standards-compliant UTF-8 JSON.
            ProjectDocumentTooLargeError: The project document or the workspace exceeds the storage limit.
        """
        pass

    @abstractmethod
    def get(
        self,
        user_id: str,
        project_id: str,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
        max_document_schema_version: int = DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
    ) -> ProjectWithWorkspaceDTO:
        """Get one of the user's projects: its content, and the user's workspace in it if they have saved one.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the stored schema.
            ProjectDocumentSchemaUnsupportedError: The caller does not understand the stored document schema.
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
        minimum_document_schema_version: int | None = None,
        max_document_schema_version: int = DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
        workspace: Optional[ProjectUpdateWorkspaceRequest] = None,
    ) -> ProjectWithWorkspaceDTO:
        """Save a project's content if the caller's revision is current, renaming its inbox to match, and with it,
        when `workspace` is given, the caller's workspace in the project, in the same transaction.

        A save that loses either revision race writes nothing. The workspace is checked as `save_workspace` checks
        it, except that the content's `max_canvas_schema_version` covers it. The result carries the workspace this
        save stored, or None when it carried none. The save that raises the document floor across the split must
        carry the workspace: a split that landed its content alone would leave the caller's settings to a default.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectRecordConflictError: The stored revision differs from expected_revision.
            ProjectWorkspaceConflictError: The stored workspace revision (or its absence) differs from the expected.
            ProjectWorkspaceBeforeSplitError: `workspace` is given, but the save leaves the document floor below the
                split schema.
            ProjectSplitNeedsWorkspaceError: The save raises the document floor across the split without `workspace`.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the stored or requested schema.
            ProjectCanvasSchemaDowngradeError: The save tries to lower the stored compatibility floor.
            ProjectDocumentSchemaUnsupportedError: The caller does not understand the stored or requested document
                schema.
            ProjectDocumentSchemaDowngradeError: The save tries to lower the stored document schema floor.
            ProjectDocumentInvalidError: The document is not standards-compliant UTF-8 JSON.
            ProjectDocumentTooLargeError: The project document exceeds the storage limit.
        """
        pass

    @abstractmethod
    def save_workspace(
        self,
        user_id: str,
        project_id: str,
        expected_revision: Optional[int],
        data: dict[str, Any],
        max_document_schema_version: int,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectWorkspaceDTO:
        """Save the user's workspace in one of their projects if `expected_revision` is its current revision, or
        None and they have none yet. Independent of the content's revision: neither half's save conflicts with the
        other's. The workspace holds canvas state (the staging area, the selection), so the project's canvas floor
        applies to it as to the content.

        Raises:
            ProjectRecordNotFoundError: No such project for this user.
            ProjectWorkspaceConflictError: The stored workspace revision (or its absence) differs from the expected.
            ProjectCanvasSchemaUnsupportedError: The caller cannot safely edit the project's canvas schema.
            ProjectDocumentSchemaUnsupportedError: The caller does not understand the project's document schema.
            ProjectDocumentInvalidError: The workspace is not standards-compliant UTF-8 JSON.
            ProjectDocumentTooLargeError: The workspace exceeds the storage limit.
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
        """Enumerate everything on the project's boards that the gallery would show, inbox first.

        Excludes intermediates and the canvas's private `other` category, so each board is exactly
        what the user sees on it. Exports use this to carry a project's whole workspace rather than
        only the media its document references.

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
