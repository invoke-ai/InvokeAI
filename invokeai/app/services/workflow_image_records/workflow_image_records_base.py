from abc import ABC, abstractmethod
from typing import Optional


class WorkflowImageRecordsStorageBase(ABC):
    """Abstract base class for the one-to-many workflow-image relationship record storage."""

    @abstractmethod
    def create(
        self,
        workflow_id: str,
        image_name: str,
    ) -> None:
        """Creates a workflow-image record."""
        pass

    @abstractmethod
    def get_workflow_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        """Gets an image's workflow id, if it has one."""
        pass
