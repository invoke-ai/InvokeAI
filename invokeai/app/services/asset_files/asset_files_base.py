from abc import ABC, abstractmethod
from pathlib import Path


class AssetFilesServiceBase(ABC):
    """Stores and serves binary 3D asset files (e.g. Gaussian-splat .ply / .splat)."""

    @abstractmethod
    def save(self, asset_name: str, data: bytes) -> None:
        """Saves a 3D asset file."""
        pass

    @abstractmethod
    def get_path(self, asset_name: str) -> Path:
        """Gets the path to a 3D asset file."""
        pass

    @abstractmethod
    def get_url(self, asset_name: str) -> str:
        """Gets the URL of a 3D asset file."""
        pass

    @abstractmethod
    def delete(self, asset_name: str) -> None:
        """Deletes a 3D asset file."""
        pass
