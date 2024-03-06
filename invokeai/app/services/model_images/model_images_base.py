from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image as PILImageType

class ModelImagesBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, model_key: str) -> PILImageType:
        """Retrieves a model image as PIL Image."""
        pass

    @abstractmethod
    def get_path(self, model_key: str) -> Path:
        """Gets the internal path to a model image."""
        pass

    @abstractmethod
    def get_url(self, model_key: str) -> str:
        """Gets the url for a model image."""
        pass

    @abstractmethod
    def save(
        self,
        image: PILImageType,
        model_key: str,
    ) -> None:
        """Saves a model image. Returns a tuple of the image name and created timestamp."""
        pass

    @abstractmethod
    def delete(self, model_key: str) -> None:
        """Deletes a model image."""
        pass
