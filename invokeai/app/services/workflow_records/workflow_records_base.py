from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.shared.pagination import PaginatedResults, SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    WORKFLOW_LIBRARY_DEFAULT_USER_ID,
    Workflow,
    WorkflowCategory,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordOrderBy,
    WorkflowWithoutID,
)


class WorkflowRecordsStorageBase(ABC):
    """Base class for workflow storage services."""

    @abstractmethod
    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        """Get workflow by id."""
        pass

    @abstractmethod
    def create(
        self,
        workflow: WorkflowWithoutID,
        user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
        is_public: bool = False,
        workflow_id: Optional[str] = None,
    ) -> WorkflowRecordDTO:
        """Creates a workflow at revision 1.

        `workflow_id` is an optional client-reserved UUID that makes creation retry-safe: a record that already
        carries it is returned as-is when the same owner submitted the same content, and any other collision raises
        `WorkflowIdConflictError`.
        """
        pass

    @abstractmethod
    def update(
        self, workflow: Workflow, user_id: Optional[str] = None, expected_revision: Optional[int] = None
    ) -> WorkflowRecordDTO:
        """Replaces a workflow's content and increments its revision, atomically with authorization and reference
        indexing.

        When `user_id` is provided the write is refused (`WorkflowAccessDeniedError`) unless that user owns the
        record. When `expected_revision` is provided the write is refused (`WorkflowRevisionConflictError`) unless
        the stored revision still matches. Bundled workflows raise `WorkflowImmutableError` regardless of the
        submitted category.
        """
        pass

    @abstractmethod
    def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        """Deletes a workflow. When user_id is provided, the DELETE is scoped to that user."""
        pass

    @abstractmethod
    def get_many(
        self,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        categories: Optional[list[WorkflowCategory]],
        page: int,
        per_page: Optional[int],
        query: Optional[str],
        tags: Optional[list[str]],
        has_been_opened: Optional[bool],
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        """Gets many workflows."""
        pass

    @abstractmethod
    def counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        """Gets a dictionary of counts for each of the provided categories."""
        pass

    @abstractmethod
    def counts_by_tag(
        self,
        tags: list[str],
        categories: Optional[list[WorkflowCategory]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        """Gets a dictionary of counts for each of the provided tags."""
        pass

    @abstractmethod
    def update_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        """Open a workflow. When user_id is provided, the UPDATE is scoped to that user."""
        pass

    @abstractmethod
    def update_last_run_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        """Records that a workflow was just run. When user_id is provided, the UPDATE is scoped to that user."""
        pass

    @abstractmethod
    def get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        """Gets all unique tags from workflows."""
        pass

    @abstractmethod
    def update_is_public(self, workflow_id: str, is_public: bool, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        """Updates the is_public field of a workflow. When user_id is provided, the UPDATE is scoped to that user."""
        pass
