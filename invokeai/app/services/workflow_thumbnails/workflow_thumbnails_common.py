from pathlib import Path

from PIL import Image


class WorkflowThumbnailFileNotFoundException(Exception):
    """Raised when a workflow thumbnail file is not found"""

    def __init__(self, message: str = "Workflow thumbnail file not found"):
        self.message = message
        super().__init__(self.message)


class WorkflowThumbnailFileSaveException(Exception):
    """Raised when a workflow thumbnail file cannot be saved"""

    def __init__(self, message: str = "Workflow thumbnail file cannot be saved"):
        self.message = message
        super().__init__(self.message)


class WorkflowThumbnailFileDeleteException(Exception):
    """Raised when a workflow thumbnail file cannot be deleted"""

    def __init__(self, message: str = "Workflow thumbnail file cannot be deleted"):
        self.message = message
        super().__init__(self.message)


class WorkflowThumbnailServiceBase:
    """Base class for workflow thumbnail services"""

    def get_path(self, workflow_id: str) -> Path:
        """Gets the path to a workflow thumbnail"""
        raise NotImplementedError

    def get_url(self, workflow_id: str) -> str | None:
        """Gets the URL of a workflow thumbnail"""
        raise NotImplementedError

    def save(self, workflow_id: str, image: Image.Image) -> None:
        """Saves a workflow thumbnail"""
        raise NotImplementedError

    def delete(self, workflow_id: str) -> None:
        """Deletes a workflow thumbnail"""
        raise NotImplementedError
