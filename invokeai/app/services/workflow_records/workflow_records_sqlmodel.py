from pathlib import Path
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    WORKFLOW_LIBRARY_DEFAULT_USER_ID,
    Workflow,
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordOrderBy,
    WorkflowValidator,
    WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string


class SqlModelWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        return self._q.workflows_get(workflow_id)

    def create(
        self,
        workflow: WorkflowWithoutID,
        user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
        is_public: bool = False,
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be created via this method")

        workflow_with_id = Workflow(**workflow.model_dump(), id=uuid_string())
        self._q.workflows_insert(
            workflow_id=workflow_with_id.id,
            workflow_json=workflow_with_id.model_dump_json(),
            user_id=user_id,
            is_public=is_public,
        )
        return self.get(workflow_with_id.id)

    def update(self, workflow: Workflow, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be updated")

        self._q.workflows_update_json(workflow.id, workflow.model_dump_json(), user_id)
        return self.get(workflow.id)

    def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        if self.get(workflow_id).workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be deleted")

        self._q.workflows_delete_user_workflow(workflow_id, user_id)

    def update_is_public(self, workflow_id: str, is_public: bool, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        record = self.get(workflow_id)
        workflow = record.workflow

        tags_list = [t.strip() for t in workflow.tags.split(",") if t.strip()] if workflow.tags else []
        if is_public and "shared" not in tags_list:
            tags_list.append("shared")
        elif not is_public and "shared" in tags_list:
            tags_list.remove("shared")
        updated_tags = ", ".join(tags_list)
        updated_workflow = workflow.model_copy(update={"tags": updated_tags})

        self._q.workflows_set_public(workflow_id, updated_workflow.model_dump_json(), is_public, user_id)
        return self.get(workflow_id)

    def get_many(
        self,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        categories: Optional[list[WorkflowCategory]] = None,
        page: int = 0,
        per_page: Optional[int] = None,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        return self._q.workflows_get_many(
            order_by=order_by,
            direction=direction,
            categories=categories,
            page=page,
            per_page=per_page,
            query=query,
            tags=tags,
            has_been_opened=has_been_opened,
            user_id=user_id,
            is_public=is_public,
        )

    def counts_by_tag(
        self,
        tags: list[str],
        categories: Optional[list[WorkflowCategory]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        return self._q.workflows_counts_by_tag(
            tags=tags,
            categories=categories,
            has_been_opened=has_been_opened,
            user_id=user_id,
            is_public=is_public,
        )

    def counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        return self._q.workflows_counts_by_category(
            categories=categories,
            has_been_opened=has_been_opened,
            user_id=user_id,
            is_public=is_public,
        )

    def update_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        self._q.workflows_touch_opened_at(workflow_id, user_id)

    def get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        return self._q.workflows_get_all_tags(categories=categories, user_id=user_id, is_public=is_public)

    def _sync_default_workflows(self) -> None:
        """Syncs default workflows to the database."""
        workflows_from_file: list[Workflow] = []
        workflows_to_update: list[Workflow] = []
        workflows_to_add: list[Workflow] = []
        workflows_dir = Path(__file__).parent / Path("default_workflows")
        workflow_paths = workflows_dir.glob("*.json")

        for path in workflow_paths:
            bytes_ = path.read_bytes()
            workflow_from_file = WorkflowValidator.validate_json(bytes_)

            assert workflow_from_file.id.startswith("default_"), (
                f'Invalid default workflow ID (must start with "default_"): {workflow_from_file.id}'
            )
            assert workflow_from_file.meta.category is WorkflowCategory.Default, (
                f"Invalid default workflow category: {workflow_from_file.meta.category}"
            )

            workflows_from_file.append(workflow_from_file)

            try:
                workflow_from_db = self.get(workflow_from_file.id).workflow
                if workflow_from_file != workflow_from_db:
                    self._invoker.services.logger.debug(
                        f"Updating library workflow {workflow_from_file.name} ({workflow_from_file.id})"
                    )
                    workflows_to_update.append(workflow_from_file)
            except WorkflowNotFoundError:
                self._invoker.services.logger.debug(
                    f"Adding missing default workflow {workflow_from_file.name} ({workflow_from_file.id})"
                )
                workflows_to_add.append(workflow_from_file)

        # Find obsolete defaults (in the DB but no longer shipped as a file)
        library_workflows_from_db = self.get_many(
            order_by=WorkflowRecordOrderBy.Name,
            direction=SQLiteDirection.Ascending,
            categories=[WorkflowCategory.Default],
        ).items
        workflows_from_file_ids = [w.id for w in workflows_from_file]

        delete_ids: list[str] = []
        for w in library_workflows_from_db:
            if w.workflow_id not in workflows_from_file_ids:
                self._invoker.services.logger.debug(f"Deleting obsolete default workflow {w.name} ({w.workflow_id})")
                delete_ids.append(w.workflow_id)

        # Apply all changes in one transaction
        self._q.workflows_apply_default_sync(
            delete_ids=delete_ids,
            add_workflows=[(w.id, w.model_dump_json()) for w in workflows_to_add],
            update_workflows=[(w.id, w.model_dump_json()) for w in workflows_to_update],
        )
