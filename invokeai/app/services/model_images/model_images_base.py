from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image as PILImageType


class ModelImagesBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, image_name: str) -> PILImageType:
        """Retrieves an image as PIL Image."""
        pass

    @abstractmethod
    def get_path(self, image_name: str) -> Path:
        """Gets the internal path to an image."""
        pass

    @abstractmethod
    def save(
        self,
        image: PILImageType,
        image_name: str,
    ) -> None:
        """Saves an image. Returns a tuple of the image name and created timestamp."""
        pass

    @abstractmethod
    def delete(self, image_name: str) -> None:
        """Deletes an image."""
        pass
