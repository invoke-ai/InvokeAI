from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class WorkflowThumbnailServiceBase(ABC):
    """Base class for workflow thumbnail services"""

    @abstractmethod
    def get_path(self, workflow_id: str, with_hash: bool = True) -> Path:
        """Gets the path to a workflow thumbnail"""
        pass

    @abstractmethod
    def get_url(self, workflow_id: str, with_hash: bool = True) -> str | None:
        """Gets the URL of a workflow thumbnail"""
        pass

    @abstractmethod
    def save(self, workflow_id: str, image: Image.Image) -> None:
        """Saves a workflow thumbnail"""
        pass

    @abstractmethod
    def delete(self, workflow_id: str) -> None:
        """Deletes a workflow thumbnail"""
        pass
