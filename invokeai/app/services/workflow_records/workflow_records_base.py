from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
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
    def create(self, workflow: WorkflowWithoutID) -> WorkflowRecordDTO:
        """Creates a workflow."""
        pass

    @abstractmethod
    def update(self, workflow: Workflow) -> WorkflowRecordDTO:
        """Updates a workflow."""
        pass

    @abstractmethod
    def delete(self, workflow_id: str) -> None:
        """Deletes a workflow."""
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
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        """Gets many workflows."""
        pass

    @abstractmethod
    def get_tag_counts_with_filter(
        self,
        tags_to_count: list[str],
        selected_tags: Optional[list[str]] = None,
        categories: Optional[list[WorkflowCategory]] = None,
    ) -> dict[str, int]:
        """
        For each tag in tags_to_count, count workflows matching:
        - All selected_tags (AND logic filter)
        - AND the specific tag being counted
        - Filtered by categories if provided

        Returns a dictionary of tag -> count.
        """
        pass

    @abstractmethod
    def update_opened_at(self, workflow_id: str) -> None:
        """Open a workflow."""
        pass
