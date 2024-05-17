from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL.Image import Image as PILImageType


class ImageFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, image_name: str) -> PILImageType:
        """Retrieves an image as PIL Image."""
        pass

    @abstractmethod
    def get_path(self, image_name: str, thumbnail: bool = False) -> Path:
        """Gets the internal path to an image or thumbnail."""
        pass

    # TODO: We need to validate paths before starlette makes the FileResponse, else we get a
    # 500 internal server error. I don't like having this method on the service.
    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for an image or thumbnail."""
        pass

    @abstractmethod
    def save(
        self,
        image: PILImageType,
        image_name: str,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        thumbnail_size: int = 256,
    ) -> None:
        """Saves an image and a 256x256 WEBP thumbnail. Returns a tuple of the image name, thumbnail name, and created timestamp."""
        pass

    @abstractmethod
    def delete(self, image_name: str) -> None:
        """Deletes an image and its thumbnail (if one exists)."""
        pass

    @abstractmethod
    def get_workflow(self, image_name: str) -> Optional[str]:
        """Gets the workflow of an image."""
        pass

    @abstractmethod
    def get_graph(self, image_name: str) -> Optional[str]:
        """Gets the graph of an image."""
        pass
