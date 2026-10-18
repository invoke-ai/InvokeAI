import json
import uuid
from typing import Any, Literal, Optional

from invokeai.app.services.board_access.board_access_common import GeneralAccess
from invokeai.app.services.board_records.board_records_common import BOARD_NAME_MAX_LENGTH
from invokeai.app.services.project_records.project_records_base import ProjectRecordsStorageBase
from invokeai.app.services.project_records.project_records_common import (
    DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
    PROJECT_BOARD_SNAPSHOT_MAX_ITEMS,
    PROJECT_DOCUMENT_MAX_BYTES,
    SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION,
    ProjectBoardNotFoundError,
    ProjectBoardSnapshotDTO,
    ProjectBoardTooLargeError,
    ProjectBoardUnavailableError,
    ProjectCanvasSchemaDowngradeError,
    ProjectCanvasSchemaUnsupportedError,
    ProjectDocumentInvalidError,
    ProjectDocumentSchemaDowngradeError,
    ProjectDocumentSchemaUnsupportedError,
    ProjectDocumentTooLargeError,
    ProjectRecordConflictError,
    ProjectRecordDTO,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSplitNeedsWorkspaceError,
    ProjectSummaryDTO,
    ProjectUpdateWorkspaceRequest,
    ProjectWithWorkspaceDTO,
    ProjectWorkspaceBeforeSplitError,
    ProjectWorkspaceConflictError,
    ProjectWorkspaceDTO,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.media_references import MediaReferences, extract_media_references
from invokeai.app.util.misc import uuid_string


def _serialize_project_document(data: dict[str, Any]) -> tuple[str, int]:
    try:
        document_json = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return document_json, len(document_json.encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ProjectDocumentInvalidError from error


def _require_project_document_size(actual_bytes: int) -> None:
    if actual_bytes > PROJECT_DOCUMENT_MAX_BYTES:
        raise ProjectDocumentTooLargeError(actual_bytes=actual_bytes, max_bytes=PROJECT_DOCUMENT_MAX_BYTES)


def _require_supported_schema(*, project_id: str, minimum_version: int, client_maximum_version: int) -> None:
    if client_maximum_version < minimum_version:
        raise ProjectCanvasSchemaUnsupportedError(
            project_id=project_id,
            minimum_version=minimum_version,
            client_maximum_version=client_maximum_version,
        )


def _require_supported_document_schema(*, project_id: str, minimum_version: int, client_maximum_version: int) -> None:
    if client_maximum_version < minimum_version:
        raise ProjectDocumentSchemaUnsupportedError(
            project_id=project_id,
            minimum_version=minimum_version,
            client_maximum_version=client_maximum_version,
        )


def _next_document_floor(project_id: str, current: int, requested: Optional[int]) -> int:
    """The document schema floor a save leaves: the requested one, which may only raise it, or the current one."""
    if requested is None:
        return current
    if requested < current:
        raise ProjectDocumentSchemaDowngradeError(
            project_id=project_id, current_version=current, requested_version=requested
        )
    return requested


def _write_workspace(
    q: Queries,
    *,
    user_id: str,
    project_id: str,
    expected_revision: Optional[int],
    document_json: str,
    references: MediaReferences,
) -> tuple[int, str]:
    """Writes the account's workspace in a project whose row the caller has locked: its first row when
    `expected_revision` is None, else its next revision. Returns the stamp (revision, saved at) it leaves.

    The project's row is the lock for its workspaces too: every workspace write takes it first, so the revision
    read here is the one this write replaces, and a deletion of the project waits for it.
    """
    current = q.projects.workspace_stamp(user_id, project_id)
    current_revision = current[0] if current is not None else None
    if current_revision != expected_revision:
        raise ProjectWorkspaceConflictError(project_id, expected_revision, current_revision)
    if current_revision is None:
        q.projects.insert_workspace(user_id=user_id, project_id=project_id, data=document_json)
    else:
        q.projects.save_workspace(user_id=user_id, project_id=project_id, data=document_json)
    q.media_references.replace(
        owner_kind="project_workspace", user_id=user_id, owner_id=project_id, references=references
    )
    stamp = q.projects.workspace_stamp(user_id, project_id)
    assert stamp is not None, "the workspace was written in this transaction"
    return stamp


def _record(summary: Optional[ProjectSummaryDTO], project_id: str, data: dict[str, Any]) -> ProjectRecordDTO:
    """The record a write returns: its row as the write left it, read in the write's transaction, with the document
    the write stored. Not read back: the document is the largest part of the row, and decoding it again would take
    longer than the whole transaction."""
    if summary is None:
        raise ProjectRecordNotFoundError(project_id)
    return ProjectRecordDTO(**summary.model_dump(), data=data)


def _claim_board(q: Queries, *, user_id: str, board_id: str, board_name: str, project_id: str) -> None:
    """Takes ownership of an existing private board for the project being created, renaming it after the project,
    or explains why it cannot be taken.

    The board's row is locked before the claim looks at the board: a change to the board, or another project
    claiming it, either committed before and is seen here, or waits until this transaction has ended and then finds
    the board claimed. The project's row, if the account already has one, is locked before the board's, the order
    every write of a project takes them in.

    The project being created is not counted as a claim on the board, so a client that re-sends a create it never
    got an answer for is refused as `ProjectRecordExistsError` -- the truth. Otherwise it would be told its board
    was unavailable, which reads as "somebody else took it" and is the opposite of what happened. A repeated create
    is not an edge case: it is how a client establishes whether a request whose response was lost actually landed,
    and the answer decides whether it deletes the media it uploaded.
    """
    project_exists = q.projects.lock(user_id, project_id) is not None
    board = q.boards.lock(board_id)
    if board is None:
        raise ProjectBoardNotFoundError(board_id)
    owner, general_access = board
    if owner != user_id:
        raise ProjectBoardNotFoundError(board_id)
    claimant = q.projects.claimant(board_id)
    record = q.boards.get(board_id)
    assert record is not None
    if (
        record.project_id not in (None, project_id)
        or general_access != GeneralAccess.Restricted.value
        or claimant not in (None, (user_id, project_id))
        or q.boards.is_shared(board_id)
    ):
        raise ProjectBoardUnavailableError(board_id)
    # The board's claimant can be the project being created although its lock found nothing: the same create sent
    # twice, both under way at once. Inserting it again would lock its row after the board's.
    if record.project_id is not None and claimant != (user_id, project_id):
        raise ProjectBoardUnavailableError(board_id)
    if project_exists or claimant is not None:
        raise ProjectRecordExistsError(project_id)
    # Un-archived as well as renamed. A project's board takes its archived state from the project, and
    # `PATCH /boards/{id}` refuses to set it on a claimed board -- so a board that arrived archived would be
    # invisible in every listing with no API left to fix it.
    q.boards.rename(board_id, board_name, unarchive=True)
    q.boards.set_project(board_id, project_id)


class ProjectRecordsStorage(ProjectRecordsStorageBase):
    """Per-user project documents. A project and its board are one unit: every write touching both runs in one
    transaction, so the board and the project commit or roll back together."""

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

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
        new_project_id = project_id or uuid.uuid4().hex
        _require_supported_schema(
            project_id=new_project_id,
            minimum_version=minimum_canvas_schema_version,
            client_maximum_version=max_canvas_schema_version,
        )
        _require_supported_document_schema(
            project_id=new_project_id,
            minimum_version=minimum_document_schema_version,
            client_maximum_version=max_document_schema_version,
        )
        # The workspace is the split's other half: beside a whole document nothing reads it, as in `update`.
        if workspace is not None and minimum_document_schema_version < SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION:
            raise ProjectWorkspaceBeforeSplitError(new_project_id, minimum_document_schema_version)
        document_json, document_bytes = _serialize_project_document(data)
        _require_project_document_size(document_bytes)
        references = extract_media_references(data)
        workspace_json, workspace_bytes = ("", 0) if workspace is None else _serialize_project_document(workspace)
        _require_project_document_size(workspace_bytes)
        workspace_references = MediaReferences() if workspace is None else extract_media_references(workspace)
        board_name = name[:BOARD_NAME_MAX_LENGTH]

        def insert(q: Queries) -> tuple[Optional[ProjectSummaryDTO], Optional[tuple[int, str]]]:
            # Shared with every write that makes media protected, exclusive for the intermediates cleanup's check and
            # delete: the media this names cannot be deleted between that check and this commit.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            if board_id is None:
                project_board_id = uuid_string()
                q.boards.insert(
                    board_id=project_board_id, board_name=board_name, user_id=user_id, project_id=new_project_id
                )
            else:
                _claim_board(q, user_id=user_id, board_id=board_id, board_name=board_name, project_id=new_project_id)
                project_board_id = board_id
            try:
                q.projects.insert(
                    project_id=new_project_id,
                    user_id=user_id,
                    name=name,
                    data=document_json,
                    board_id=project_board_id,
                    minimum_canvas_schema_version=minimum_canvas_schema_version,
                    minimum_document_schema_version=minimum_document_schema_version,
                )
            except UniqueViolation as error:
                # A project has this id: the account's own, created meanwhile or, when the create brings no board,
                # before; or another account's, since ids are global. It cannot be the board: a new board is the
                # project's alone, and a claimed one is locked and claimed by no other project. The board insert or
                # rename rolls back with the project insert, so a refused create changes no board.
                raise ProjectRecordExistsError(new_project_id) from error
            # Indexed in the document's own transaction: a reader never sees a saved project whose media the
            # intermediates cleanup does not know about.
            q.media_references.replace(
                owner_kind="project", user_id=user_id, owner_id=new_project_id, references=references
            )
            stamp = None
            if workspace is not None:
                # In the project's transaction: the creator's workspace is never a second request that a sign-out
                # can lose, and a refused create leaves no workspace row. Always the first row: the project is new.
                stamp = _write_workspace(
                    q,
                    user_id=user_id,
                    project_id=new_project_id,
                    expected_revision=None,
                    document_json=workspace_json,
                    references=workspace_references,
                )
            return q.projects.summary(user_id, new_project_id), stamp

        summary, stamp = self._queries.run(insert)
        record = _record(summary, new_project_id, data)
        saved_workspace = None
        if workspace is not None and stamp is not None:
            saved_workspace = ProjectWorkspaceDTO(data=workspace, revision=stamp[0], updated_at=stamp[1])
        # Shallow, like `get`: `model_dump` would copy the whole content.
        return ProjectWithWorkspaceDTO(**dict(record), workspace=saved_workspace)

    def get(
        self,
        user_id: str,
        project_id: str,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
        max_document_schema_version: int = DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION,
    ) -> ProjectWithWorkspaceDTO:
        def read(q: Queries) -> tuple[Optional[ProjectRecordDTO], Optional[ProjectWorkspaceDTO]]:
            # One snapshot: both rows as they stood at one instant, though each half is saved on its own.
            return q.projects.get(user_id, project_id), q.projects.workspace(user_id, project_id)

        record, workspace = self._queries.run(read, read_only=True)
        if record is None:
            raise ProjectRecordNotFoundError(project_id)
        _require_supported_schema(
            project_id=project_id,
            minimum_version=record.minimum_canvas_schema_version,
            client_maximum_version=max_canvas_schema_version,
        )
        _require_supported_document_schema(
            project_id=project_id,
            minimum_version=record.minimum_document_schema_version,
            client_maximum_version=max_document_schema_version,
        )
        # Shallow: `model_dump` would copy the whole decoded content, which can run to tens of megabytes.
        return ProjectWithWorkspaceDTO(**dict(record), workspace=workspace)

    def list(self, user_id: str) -> list[ProjectSummaryDTO]:
        return self._queries.projects.summaries(user_id)

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
        document_json, document_bytes = _serialize_project_document(data)
        references = extract_media_references(data)
        workspace_json, workspace_bytes = ("", 0) if workspace is None else _serialize_project_document(workspace.data)
        workspace_references = MediaReferences() if workspace is None else extract_media_references(workspace.data)

        def save(q: Queries) -> tuple[Optional[ProjectSummaryDTO], Optional[tuple[int, str]]]:
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            # Locked before anything is checked, so the checks and the write apply to the same row, whatever other
            # transactions save, delete or create under this id meanwhile.
            project = q.projects.lock(user_id, project_id)
            if project is None:
                raise ProjectRecordNotFoundError(project_id)
            current_minimum_version = project.minimum_canvas_schema_version
            _require_supported_schema(
                project_id=project_id,
                minimum_version=current_minimum_version,
                client_maximum_version=max_canvas_schema_version,
            )
            _require_supported_document_schema(
                project_id=project_id,
                minimum_version=project.minimum_document_schema_version,
                client_maximum_version=max_document_schema_version,
            )
            if project.revision != expected_revision:
                raise ProjectRecordConflictError(project_id, expected_revision, project.revision)
            next_minimum_version = (
                current_minimum_version if minimum_canvas_schema_version is None else minimum_canvas_schema_version
            )
            if next_minimum_version < current_minimum_version:
                raise ProjectCanvasSchemaDowngradeError(
                    project_id=project_id,
                    current_version=current_minimum_version,
                    requested_version=next_minimum_version,
                )
            _require_supported_schema(
                project_id=project_id,
                minimum_version=next_minimum_version,
                client_maximum_version=max_canvas_schema_version,
            )
            next_document_floor = _next_document_floor(
                project_id, project.minimum_document_schema_version, minimum_document_schema_version
            )
            _require_supported_document_schema(
                project_id=project_id,
                minimum_version=next_document_floor,
                client_maximum_version=max_document_schema_version,
            )
            if workspace is not None and next_document_floor < SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION:
                raise ProjectWorkspaceBeforeSplitError(project_id, next_document_floor)
            # The split must land whole: stored content without the caller's workspace row would join every later
            # load with a default workspace, whatever client sent it.
            if (
                workspace is None
                and project.minimum_document_schema_version
                < SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION
                <= next_document_floor
            ):
                raise ProjectSplitNeedsWorkspaceError(project_id, project.minimum_document_schema_version)
            _require_project_document_size(document_bytes)
            _require_project_document_size(workspace_bytes)

            q.projects.save(
                user_id=user_id,
                project_id=project_id,
                name=name,
                data=document_json,
                minimum_canvas_schema_version=next_minimum_version,
                minimum_document_schema_version=next_document_floor,
            )
            q.boards.rename(project.board_id, name[:BOARD_NAME_MAX_LENGTH])
            q.media_references.replace(
                owner_kind="project", user_id=user_id, owner_id=project_id, references=references
            )
            stamp = None
            if workspace is not None:
                # In the content's transaction: a workspace conflict rolls the content back with it, so a split
                # never lands by half.
                stamp = _write_workspace(
                    q,
                    user_id=user_id,
                    project_id=project_id,
                    expected_revision=workspace.expected_revision,
                    document_json=workspace_json,
                    references=workspace_references,
                )
            return q.projects.summary(user_id, project_id), stamp

        summary, stamp = self._queries.run(save)
        record = _record(summary, project_id, data)
        saved_workspace = None
        if workspace is not None and stamp is not None:
            saved_workspace = ProjectWorkspaceDTO(data=workspace.data, revision=stamp[0], updated_at=stamp[1])
        # Shallow, like `get`: `model_dump` would copy the whole content.
        return ProjectWithWorkspaceDTO(**dict(record), workspace=saved_workspace)

    def save_workspace(
        self,
        user_id: str,
        project_id: str,
        expected_revision: Optional[int],
        data: dict[str, Any],
        max_document_schema_version: int,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectWorkspaceDTO:
        document_json, document_bytes = _serialize_project_document(data)
        _require_project_document_size(document_bytes)
        references = extract_media_references(data)

        def save(q: Queries) -> tuple[int, str]:
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            project = q.projects.lock(user_id, project_id)
            if project is None:
                raise ProjectRecordNotFoundError(project_id)
            _require_supported_schema(
                project_id=project_id,
                minimum_version=project.minimum_canvas_schema_version,
                client_maximum_version=max_canvas_schema_version,
            )
            _require_supported_document_schema(
                project_id=project_id,
                minimum_version=project.minimum_document_schema_version,
                client_maximum_version=max_document_schema_version,
            )
            # The workspace is the split's other half: beside a whole document nothing reads it, as in `update`.
            if project.minimum_document_schema_version < SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION:
                raise ProjectWorkspaceBeforeSplitError(project_id, project.minimum_document_schema_version)
            return _write_workspace(
                q,
                user_id=user_id,
                project_id=project_id,
                expected_revision=expected_revision,
                document_json=document_json,
                references=references,
            )

        stamp = self._queries.run(save)
        # The document the client sent, not a decode of the stored one: the same JSON, and the larger part.
        return ProjectWorkspaceDTO(data=data, revision=stamp[0], updated_at=stamp[1])

    def delete(self, user_id: str, project_id: str, boards: Literal["release", "delete"] = "release") -> None:
        def delete(q: Queries) -> None:
            # The project's row, then its board's, as every write of a project locks them: the board deleted below
            # is the one of the project deleted here.
            project = q.projects.lock(user_id, project_id)
            if project is None:
                return
            q.boards.lock(project.board_id)
            # Read before the project's row goes, and its workspaces with it; its lock keeps new ones from being saved.
            members = q.projects.workspace_members(project_id)
            q.boards.remove_project_members(user_id, project_id, project.board_id, delete_members=boards == "delete")
            q.projects.delete(user_id, project_id)
            q.media_references.delete(owner_kind="project", user_id=user_id, owner_id=project_id)
            q.media_references.delete_many_owners(owner_kind="project_workspace", user_ids=members, owner_id=project_id)
            # Only after the project is gone: a claimed board cannot be deleted. Deleting the board deletes its
            # memberships, returning the media to Uncategorized; the image and video records and their files are
            # untouched.
            q.boards.delete_if_unclaimed(project.board_id)

        self._queries.run(delete)

    def get_board_snapshot(self, user_id: str, project_id: str) -> ProjectBoardSnapshotDTO:
        # Bounded, unlike the board's counts: the caller holds the whole answer in memory and so does this, and
        # the route is reachable by anyone with a project id.
        snapshot = self._queries.projects.board_snapshot(user_id, project_id, PROJECT_BOARD_SNAPSHOT_MAX_ITEMS + 1)
        if snapshot is None:
            raise ProjectRecordNotFoundError(project_id)
        if sum(len(board.items) for board in snapshot.boards) > PROJECT_BOARD_SNAPSHOT_MAX_ITEMS:
            raise ProjectBoardTooLargeError(project_id, PROJECT_BOARD_SNAPSHOT_MAX_ITEMS)
        return snapshot

    def get_board_id(self, user_id: str, project_id: str) -> str:
        board_id = self._queries.projects.board_id(user_id, project_id)
        if board_id is None:
            raise ProjectRecordNotFoundError(project_id)
        return board_id
