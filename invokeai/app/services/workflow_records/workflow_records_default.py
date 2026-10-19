import uuid
from pathlib import Path
from typing import Optional, Union

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.media_references import extract_media_references_from_json
from invokeai.app.services.shared.pagination import PaginatedResults, SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    WORKFLOW_LIBRARY_DEFAULT_USER_ID,
    Workflow,
    WorkflowAccessDeniedError,
    WorkflowCategory,
    WorkflowIdConflictError,
    WorkflowImmutableError,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordDTOBase,
    WorkflowRecordListItemDTO,
    WorkflowRecordOrderBy,
    WorkflowRevisionConflictError,
    WorkflowValidator,
    WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string

_BUNDLED_WORKFLOWS = Path(__file__).parent / "default_workflows"


class _IdTaken(Exception):
    """A creation's id belongs to a stored workflow."""


def _not_found(workflow_id: str) -> WorkflowNotFoundError:
    return WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")


def _owner(user_id: Optional[str]) -> str:
    return user_id if user_id is not None else WORKFLOW_LIBRARY_DEFAULT_USER_ID


def _record(summary: Optional[WorkflowRecordDTOBase], workflow: Workflow) -> WorkflowRecordDTO:
    """The record a write returns: its row as the write left it, read in the write's transaction, with the workflow
    the write stored."""
    if summary is None:
        raise _not_found(workflow.id)
    return WorkflowRecordDTO(**summary.model_dump(), workflow=workflow)


def _same_submission(existing: WorkflowRecordDTO, workflow: Workflow, user_id: str) -> WorkflowRecordDTO:
    """A retried creation is accepted only when the record it finds is this owner's identical submission.

    Any other record under the id is a conflict; its content and owner are not revealed to the caller.
    """
    if existing.user_id != user_id or existing.workflow.model_dump() != workflow.model_dump():
        raise WorkflowIdConflictError(workflow.id)
    return existing


class WorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        """Gets a workflow by ID."""
        record = self._queries.workflows.get(workflow_id)
        if record is None:
            raise _not_found(workflow_id)
        return record

    def create(
        self,
        workflow: WorkflowWithoutID,
        user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
        is_public: bool = False,
        workflow_id: Optional[str] = None,
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be created via this method")
        if workflow_id is not None:
            try:
                uuid.UUID(workflow_id)
            except ValueError as e:
                raise ValueError("A reserved workflow id must be a UUID") from e

        workflow_with_id = Workflow(**workflow.model_dump(), id=workflow_id or uuid_string())
        document_json = workflow_with_id.model_dump_json()
        references = extract_media_references_from_json(document_json)

        def insert(q: Queries) -> Optional[WorkflowRecordDTOBase]:
            # Shared with every write that makes media protected, exclusive for the intermediates cleanup's check and
            # delete: the media this names cannot be deleted between that check and this commit.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            try:
                q.workflows.insert(
                    workflow_id=workflow_with_id.id, workflow=document_json, user_id=user_id, is_public=is_public
                )
            except UniqueViolation as error:
                # Only a reserved id can collide; a generated one is unique for practical purposes.
                raise _IdTaken from error
            q.media_references.replace(
                owner_kind="workflow", user_id=user_id, owner_id=workflow_with_id.id, references=references
            )
            return q.workflows.summary(workflow_with_id.id)

        try:
            return _record(self._queries.run(insert), workflow_with_id)
        except _IdTaken as error:
            if workflow_id is None:
                raise WorkflowIdConflictError(workflow_with_id.id) from error
            # A creation sent again, to learn whether it landed -- possibly while the first one was still under way --
            # finds the record it made; anything else stored under the id is a conflict.
            existing = self._queries.workflows.get(workflow_with_id.id)
            if existing is None:
                raise WorkflowIdConflictError(workflow_with_id.id) from error
            return _same_submission(existing, workflow_with_id, user_id)

    def update(
        self, workflow: Workflow, user_id: Optional[str] = None, expected_revision: Optional[int] = None
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("A workflow cannot be updated into the default category")

        document_json = workflow.model_dump_json()
        references = extract_media_references_from_json(document_json)

        def save(q: Queries) -> Optional[WorkflowRecordDTOBase]:
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            # Locked before anything is checked: the checks and the write apply to the same row. `category` is
            # generated from the stored JSON, so it still describes the record as it is, not as the request would
            # rewrite it.
            stored = q.workflows.lock(workflow.id)
            if stored is None:
                raise _not_found(workflow.id)
            if stored.category == WorkflowCategory.Default.value:
                raise WorkflowImmutableError(workflow.id)
            owner = _owner(stored.user_id)
            if user_id is not None and owner != user_id:
                raise WorkflowAccessDeniedError(workflow.id)
            if expected_revision is not None and stored.revision != expected_revision:
                raise WorkflowRevisionConflictError(workflow.id, expected_revision, stored.revision)
            q.workflows.save(workflow.id, document_json)
            q.media_references.replace(
                owner_kind="workflow", user_id=owner, owner_id=workflow.id, references=references
            )
            return q.workflows.summary(workflow.id)

        return _record(self._queries.run(save), workflow)

    def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        def delete(q: Queries) -> None:
            stored = q.workflows.lock(workflow_id)
            if stored is None:
                raise _not_found(workflow_id)
            if stored.category == WorkflowCategory.Default.value:
                raise WorkflowImmutableError(workflow_id)
            owner = _owner(stored.user_id)
            if user_id is not None and owner != user_id:
                raise WorkflowAccessDeniedError(workflow_id)
            q.workflows.delete(workflow_id)
            # Every write of a workflow indexes its references under its owner.
            q.media_references.delete(owner_kind="workflow", user_id=owner, owner_id=workflow_id)

        self._queries.run(delete)

    def update_is_public(self, workflow_id: str, is_public: bool, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        """Updates the is_public field of a workflow and manages the 'shared' tag automatically."""

        def change(q: Queries) -> tuple[Optional[WorkflowRecordDTOBase], Union[Workflow, str]]:
            # The document is read and rewritten under the row's lock. A concurrent workflow edit must not land
            # between the read and this full-document write: its reference index would then describe the edit while
            # the stored workflow names the previous assets.
            stored = q.workflows.lock(workflow_id, with_document=True)
            if stored is None or stored.document is None:
                raise _not_found(workflow_id)
            if stored.category != WorkflowCategory.User.value or (user_id is not None and stored.user_id != user_id):
                return q.workflows.summary(workflow_id), stored.document
            workflow = Workflow.model_validate_json(stored.document)
            tags_list = [t.strip() for t in workflow.tags.split(",") if t.strip()] if workflow.tags else []
            if is_public and "shared" not in tags_list:
                tags_list.append("shared")
            elif not is_public and "shared" in tags_list:
                tags_list.remove("shared")
            workflow = workflow.model_copy(update={"tags": ", ".join(tags_list)})
            # Visibility is bookkeeping: the `shared` tag rewrite does not advance the content revision, so an editor
            # holding the previous revision is not asked to resolve a conflict it cannot see.
            q.workflows.set_public(workflow_id, workflow.model_dump_json(), is_public)
            return q.workflows.summary(workflow_id), workflow

        summary, workflow = self._queries.run(change)
        return _record(
            summary, workflow if isinstance(workflow, Workflow) else WorkflowValidator.validate_json(workflow)
        )

    def get_many(
        self,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        categories: Optional[list[WorkflowCategory]],
        page: int = 0,
        per_page: Optional[int] = None,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        workflows, total = self._queries.workflows.page(
            order_by=order_by,
            direction=direction,
            offset=page * per_page if per_page else 0,
            limit=per_page if per_page else None,
            categories=categories,
            tags=tags,
            has_been_opened=has_been_opened,
            query=query,
            user_id=user_id,
            is_public=is_public,
        )
        pages = total // per_page + (total % per_page > 0) if per_page else 1  # Without pagination, one page.
        return PaginatedResults(
            items=workflows,
            page=page,
            per_page=per_page if per_page else total,
            pages=pages,
            total=total,
        )

    def counts_by_tag(
        self,
        tags: list[str],
        categories: Optional[list[WorkflowCategory]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        if not tags:
            return {}
        counts = self._queries.workflows.tag_counts(
            tags, categories=categories, has_been_opened=has_been_opened, user_id=user_id, is_public=is_public
        )
        return dict(zip(tags, counts, strict=True))

    def counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        if not categories:
            return {}
        return self._queries.workflows.category_counts(
            categories, has_been_opened=has_been_opened, user_id=user_id, is_public=is_public
        )

    def update_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        self._queries.workflows.mark("opened_at", workflow_id, user_id)

    def update_last_run_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        self._queries.workflows.mark("last_run_at", workflow_id, user_id)

    def get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        tag_lists = self._queries.workflows.tag_lists(categories=categories, user_id=user_id, is_public=is_public)
        # Each workflow's tags are one comma-separated string.
        return sorted({tag.strip() for tag_list in tag_lists for tag in tag_list.split(",") if tag.strip()})

    def _sync_default_workflows(self) -> None:
        """Syncs default workflows to the database. Internal use only.

        An enhancement might be to only update workflows that have changed. This would require stable default
        workflow IDs, and properly incrementing the workflow version.

        It's much simpler to just replace them all with whichever workflows are in the directory.

        The downside is that the `updated_at` and `opened_at` timestamps for default workflows are meaningless, as
        they are overwritten every time the server starts.
        """
        bundled: list[Workflow] = []
        for path in _BUNDLED_WORKFLOWS.glob("*.json"):
            workflow = WorkflowValidator.validate_json(path.read_bytes())
            assert workflow.id.startswith("default_"), (
                f'Invalid default workflow ID (must start with "default_"): {workflow.id}'
            )
            assert workflow.meta.category is WorkflowCategory.Default, (
                f"Invalid default workflow category: {workflow.meta.category}"
            )
            bundled.append(workflow)
        bundled_ids = {workflow.id for workflow in bundled}

        def sync(q: Queries) -> tuple[list[tuple[str, str]], list[Workflow], list[Workflow]]:
            stored = q.workflows.documents(bundled_ids)
            obsolete = [
                (workflow_id, name) for workflow_id, name in q.workflows.bundled() if workflow_id not in bundled_ids
            ]
            missing = [workflow for workflow in bundled if workflow.id not in stored]
            changed = [
                workflow
                for workflow in bundled
                if workflow.id in stored and workflow != WorkflowValidator.validate_json(stored[workflow.id])
            ]
            for workflow_id, _name in obsolete:
                # Not `delete`, which refuses bundled workflows.
                q.workflows.delete(workflow_id)
            for workflow in missing:
                q.workflows.insert(
                    workflow_id=workflow.id,
                    workflow=workflow.model_dump_json(),
                    user_id=WORKFLOW_LIBRARY_DEFAULT_USER_ID,
                    is_public=False,
                )
            for workflow in changed:
                # Not `update`, which refuses bundled workflows. A changed bundle is new content, so its revision
                # advances like any other write.
                q.workflows.save(workflow.id, workflow.model_dump_json())
            return obsolete, missing, changed

        obsolete, missing, changed = self._queries.run(sync)
        logger = self._invoker.services.logger
        for workflow_id, name in obsolete:
            logger.debug(f"Deleting obsolete default workflow {name} ({workflow_id})")
        for workflow in missing:
            logger.debug(f"Adding missing default workflow {workflow.name} ({workflow.id})")
        for workflow in changed:
            logger.debug(f"Updating library workflow {workflow.name} ({workflow.id})")
