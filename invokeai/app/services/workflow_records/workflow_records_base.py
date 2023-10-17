from abc import ABC, abstractmethod

from invokeai.app.invocations.baseinvocation import WorkflowField


class WorkflowRecordsStorageBase(ABC):
    """Base class for workflow storage services."""

    @abstractmethod
    def get(self, workflow_id: str) -> WorkflowField:
        """Get workflow by id."""
        pass

    @abstractmethod
    def create(self, workflow: WorkflowField) -> WorkflowField:
        """Creates a workflow."""
        pass
