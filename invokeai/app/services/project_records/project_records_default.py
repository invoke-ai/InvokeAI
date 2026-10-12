import json
import uuid
from typing import Any, Literal, Optional

from invokeai.app.services.board_records.board_records_common import BOARD_NAME_MAX_LENGTH, BoardVisibility
from invokeai.app.services.project_records.project_records_base import ProjectRecordsStorageBase
from invokeai.app.services.project_records.project_records_common import (
    DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    PROJECT_BOARD_SNAPSHOT_MAX_ITEMS,
    PROJECT_DOCUMENT_MAX_BYTES,
    ProjectBoardNotFoundError,
    ProjectBoardSnapshotDTO,
    ProjectBoardTooLargeError,
    ProjectBoardUnavailableError,
    ProjectCanvasSchemaDowngradeError,
    ProjectCanvasSchemaUnsupportedError,
    ProjectDocumentInvalidError,
    ProjectDocumentTooLargeError,
    ProjectRecordConflictError,
    ProjectRecordDTO,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSummaryDTO,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.media_references import extract_media_references
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
    owner, visibility = board
    if owner != user_id:
        raise ProjectBoardNotFoundError(board_id)
    claimant = q.projects.claimant(board_id)
    record = q.boards.get(board_id)
    assert record is not None
    if (
        record.project_id not in (None, project_id)
        or visibility != BoardVisibility.Private.value
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
    ) -> ProjectRecordDTO:
        new_project_id = project_id or uuid.uuid4().hex
        _require_supported_schema(
            project_id=new_project_id,
            minimum_version=minimum_canvas_schema_version,
            client_maximum_version=max_canvas_schema_version,
        )
        document_json, document_bytes = _serialize_project_document(data)
        _require_project_document_size(document_bytes)
        references = extract_media_references(data)
        board_name = name[:BOARD_NAME_MAX_LENGTH]

        def insert(q: Queries) -> Optional[ProjectSummaryDTO]:
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
                )
            except UniqueViolation as error:
                # The account has a project with this id, created meanwhile or, when the create brings no board,
                # before. It cannot be the board: a new board is the project's alone, and a claimed one is locked and
                # claimed by no other project. The board insert or rename rolls back with the project insert, so a
                # refused create changes no board.
                raise ProjectRecordExistsError(new_project_id) from error
            # Indexed in the document's own transaction: a reader never sees a saved project whose media the
            # intermediates cleanup does not know about.
            q.media_references.replace(
                owner_kind="project", user_id=user_id, owner_id=new_project_id, references=references
            )
            return q.projects.summary(user_id, new_project_id)

        return _record(self._queries.run(insert), new_project_id, data)

    def get(
        self,
        user_id: str,
        project_id: str,
        max_canvas_schema_version: int = DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    ) -> ProjectRecordDTO:
        record = self._queries.projects.get(user_id, project_id)
        if record is None:
            raise ProjectRecordNotFoundError(project_id)
        _require_supported_schema(
            project_id=project_id,
            minimum_version=record.minimum_canvas_schema_version,
            client_maximum_version=max_canvas_schema_version,
        )
        return record

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
    ) -> ProjectRecordDTO:
        document_json, document_bytes = _serialize_project_document(data)
        references = extract_media_references(data)

        def save(q: Queries) -> Optional[ProjectSummaryDTO]:
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
            _require_project_document_size(document_bytes)

            q.projects.save(
                user_id=user_id,
                project_id=project_id,
                name=name,
                data=document_json,
                minimum_canvas_schema_version=next_minimum_version,
            )
            q.boards.rename(project.board_id, name[:BOARD_NAME_MAX_LENGTH])
            q.media_references.replace(
                owner_kind="project", user_id=user_id, owner_id=project_id, references=references
            )
            return q.projects.summary(user_id, project_id)

        return _record(self._queries.run(save), project_id, data)

    def delete(self, user_id: str, project_id: str, boards: Literal["release", "delete"] = "release") -> None:
        def delete(q: Queries) -> None:
            # The project's row, then its board's, as every write of a project locks them: the board deleted below
            # is the one of the project deleted here.
            project = q.projects.lock(user_id, project_id)
            if project is None:
                return
            q.boards.lock(project.board_id)
            q.boards.remove_project_members(user_id, project_id, project.board_id, delete_members=boards == "delete")
            q.projects.delete(user_id, project_id)
            q.media_references.delete(owner_kind="project", user_id=user_id, owner_id=project_id)
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
