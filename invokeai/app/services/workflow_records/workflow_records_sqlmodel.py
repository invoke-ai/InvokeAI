from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func
from sqlmodel import col, select

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.models import WorkflowLibraryTable
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
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
    WorkflowValidator,
    WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string


def _row_to_dto(row: WorkflowLibraryTable) -> WorkflowRecordDTO:
    return WorkflowRecordDTO.from_dict(
        {
            "workflow_id": row.workflow_id,
            "workflow": row.workflow,
            "name": row.name,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
            "opened_at": str(row.opened_at) if row.opened_at else None,
            "user_id": row.user_id,
            "is_public": row.is_public,
        }
    )


def _row_to_list_item(row: WorkflowLibraryTable) -> WorkflowRecordListItemDTO:
    return WorkflowRecordListItemDTOValidator.validate_python(
        {
            "workflow_id": row.workflow_id,
            "category": row.category,
            "name": row.name,
            "description": row.description,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
            "opened_at": str(row.opened_at) if row.opened_at else None,
            "tags": row.tags,
            "user_id": row.user_id,
            "is_public": row.is_public,
        }
    )


class SqlModelWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        with self._db.get_readonly_session() as session:
            row = session.get(WorkflowLibraryTable, workflow_id)
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
            return _row_to_dto(row)

    def create(
        self,
        workflow: WorkflowWithoutID,
        user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
        is_public: bool = False,
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be created via this method")

        workflow_with_id = Workflow(**workflow.model_dump(), id=uuid_string())
        row = WorkflowLibraryTable(
            workflow_id=workflow_with_id.id,
            workflow=workflow_with_id.model_dump_json(),
            user_id=user_id,
            is_public=is_public,
        )
        with self._db.get_session() as session:
            session.add(row)
        return self.get(workflow_with_id.id)

    def update(self, workflow: Workflow, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be updated")

        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow.id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                row.workflow = workflow.model_dump_json()
                session.add(row)

        return self.get(workflow.id)

    def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        if self.get(workflow_id).workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be deleted")

        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow_id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                session.delete(row)

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

        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow_id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                row.workflow = updated_workflow.model_dump_json()
                row.is_public = is_public
                session.add(row)

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
        with self._db.get_readonly_session() as session:
            stmt = select(WorkflowLibraryTable)
            count_stmt = select(func.count()).select_from(WorkflowLibraryTable)

            # Apply filters to both
            stmt, count_stmt = self._apply_filters(
                stmt,
                count_stmt,
                categories,
                query,
                tags,
                has_been_opened,
                user_id,
                is_public,
            )

            # Count
            total = session.exec(count_stmt).one()

            # Ordering
            order_col = self._get_order_col(order_by)
            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())

            # Pagination
            if per_page:
                stmt = stmt.limit(per_page).offset(page * per_page)

            rows = session.exec(stmt).all()
            workflows = [_row_to_list_item(r) for r in rows]

        if per_page:
            pages = total // per_page + (total % per_page > 0)
        else:
            pages = 1

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

        result: dict[str, int] = {}
        with self._db.get_readonly_session() as session:
            for tag in tags:
                stmt = select(func.count()).select_from(WorkflowLibraryTable)
                stmt, _ = self._apply_filters(stmt, stmt, categories, None, None, has_been_opened, user_id, is_public)
                stmt = stmt.where(col(WorkflowLibraryTable.tags).like(f"%{tag.strip()}%"))
                count = session.exec(stmt).one()
                result[tag] = count
        return result

    def counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._db.get_readonly_session() as session:
            for category in categories:
                stmt = select(func.count()).select_from(WorkflowLibraryTable)
                stmt, _ = self._apply_filters(stmt, stmt, categories, None, None, has_been_opened, user_id, is_public)
                stmt = stmt.where(col(WorkflowLibraryTable.category) == category.value)
                count = session.exec(stmt).one()
                result[category.value] = count
        return result

    def update_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(col(WorkflowLibraryTable.workflow_id) == workflow_id)
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)
            row = session.exec(stmt).first()
            if row is not None:
                row.opened_at = datetime.utcnow()
                session.add(row)

    def get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt = select(WorkflowLibraryTable.tags).where(
                col(WorkflowLibraryTable.tags).is_not(None),
                col(WorkflowLibraryTable.tags) != "",
            )

            if categories:
                category_strings = [c.value for c in categories]
                stmt = stmt.where(col(WorkflowLibraryTable.category).in_(category_strings))
            if user_id is not None:
                stmt = stmt.where(
                    (col(WorkflowLibraryTable.user_id) == user_id) | (col(WorkflowLibraryTable.category) == "default")
                )
            if is_public is True:
                stmt = stmt.where(col(WorkflowLibraryTable.is_public) == True)  # noqa: E712
            elif is_public is False:
                stmt = stmt.where(col(WorkflowLibraryTable.is_public) == False)  # noqa: E712

            rows = session.exec(stmt).all()

        all_tags: set[str] = set()
        for tags_value in rows:
            if tags_value and isinstance(tags_value, str):
                for tag in tags_value.split(","):
                    tag_stripped = tag.strip()
                    if tag_stripped:
                        all_tags.add(tag_stripped)
        return sorted(all_tags)

    def _sync_default_workflows(self) -> None:
        """Syncs default workflows to the database."""
        with self._db.get_session() as session:
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

            # Delete obsolete defaults
            library_workflows_from_db = self.get_many(
                order_by=WorkflowRecordOrderBy.Name,
                direction=SQLiteDirection.Ascending,
                categories=[WorkflowCategory.Default],
            ).items
            workflows_from_file_ids = [w.id for w in workflows_from_file]

            for w in library_workflows_from_db:
                if w.workflow_id not in workflows_from_file_ids:
                    self._invoker.services.logger.debug(
                        f"Deleting obsolete default workflow {w.name} ({w.workflow_id})"
                    )
                    row = session.get(WorkflowLibraryTable, w.workflow_id)
                    if row is not None:
                        session.delete(row)

            # Add new defaults
            for w in workflows_to_add:
                session.add(
                    WorkflowLibraryTable(
                        workflow_id=w.id,
                        workflow=w.model_dump_json(),
                    )
                )

            # Update changed defaults
            for w in workflows_to_update:
                row = session.get(WorkflowLibraryTable, w.id)
                if row is not None:
                    row.workflow = w.model_dump_json()
                    session.add(row)

    @staticmethod
    def _apply_filters(stmt, count_stmt, categories, query, tags, has_been_opened, user_id, is_public):
        """Apply common filters to both data and count queries."""
        if categories:
            category_strings = [c.value for c in categories]
            cond = col(WorkflowLibraryTable.category).in_(category_strings)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if tags:
            for tag in tags:
                cond = col(WorkflowLibraryTable.tags).like(f"%{tag.strip()}%")
                stmt = stmt.where(cond)
                count_stmt = count_stmt.where(cond)

        if has_been_opened is True:
            cond = col(WorkflowLibraryTable.opened_at).is_not(None)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)
        elif has_been_opened is False:
            cond = col(WorkflowLibraryTable.opened_at).is_(None)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        stripped_query = query.strip() if query else None
        if stripped_query:
            wildcard = f"%{stripped_query}%"
            cond = (
                col(WorkflowLibraryTable.name).like(wildcard)
                | col(WorkflowLibraryTable.description).like(wildcard)
                | col(WorkflowLibraryTable.tags).like(wildcard)
            )
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if user_id is not None:
            cond = (col(WorkflowLibraryTable.user_id) == user_id) | (col(WorkflowLibraryTable.category) == "default")
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if is_public is True:
            cond = col(WorkflowLibraryTable.is_public) == True  # noqa: E712
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)
        elif is_public is False:
            cond = col(WorkflowLibraryTable.is_public) == False  # noqa: E712
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        return stmt, count_stmt

    @staticmethod
    def _get_order_col(order_by: WorkflowRecordOrderBy):
        if order_by == WorkflowRecordOrderBy.Name:
            return col(WorkflowLibraryTable.name)
        elif order_by == WorkflowRecordOrderBy.Description:
            return col(WorkflowLibraryTable.description)
        elif order_by == WorkflowRecordOrderBy.CreatedAt:
            return col(WorkflowLibraryTable.created_at)
        elif order_by == WorkflowRecordOrderBy.UpdatedAt:
            return col(WorkflowLibraryTable.updated_at)
        elif order_by == WorkflowRecordOrderBy.OpenedAt:
            return col(WorkflowLibraryTable.opened_at)
        else:
            return col(WorkflowLibraryTable.created_at)
