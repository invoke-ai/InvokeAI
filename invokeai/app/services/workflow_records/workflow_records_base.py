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
        page: int,
        per_page: int,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        category: WorkflowCategory,
        filter_text: Optional[str],
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        """Gets many workflows."""
        pass

    @abstractmethod
    def _create_system_workflow(self, workflow: Workflow) -> None:
        """Creates a system workflow. Internal use only."""
        pass

    @abstractmethod
    def _update_system_workflow(self, workflow: Workflow) -> None:
        """Updates a system workflow. Internal use only."""
        pass

    @abstractmethod
    def _delete_system_workflow(self, workflow_id: str) -> None:
        """Deletes a system workflow. Internal use only."""
        pass

    @abstractmethod
    def _get_all_system_workflows(self) -> list[Workflow]:
        """Gets all system workflows. Internal use only."""
        pass
