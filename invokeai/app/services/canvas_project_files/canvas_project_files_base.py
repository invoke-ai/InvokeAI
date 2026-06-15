from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class CanvasProjectFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving canvas project (.invk) files."""

    @abstractmethod
    def get_path(self, project_name: str, thumbnail: bool = False, project_subfolder: str = "") -> Path:
        """Gets the internal path to a canvas project ZIP or its thumbnail WebP."""
        pass

    @abstractmethod
    def save(
        self,
        zip_bytes: bytes,
        project_name: str,
        thumbnail_bytes: Optional[bytes] = None,
        project_subfolder: str = "",
    ) -> None:
        """Saves a canvas project ZIP and optional WebP thumbnail to disk."""
        pass

    @abstractmethod
    def delete(self, project_name: str, project_subfolder: str = "") -> None:
        """Deletes a canvas project file and its thumbnail (if one exists)."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for a project or thumbnail."""
        pass
