from abc import ABC, abstractmethod

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
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
    def get_many(self, page: int, per_page: int) -> PaginatedResults[WorkflowRecordListItemDTO]:
        """Gets many workflows."""
        pass
