"""
Disk-based converted model cache.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ModelConvertCacheBase(ABC):
    @property
    @abstractmethod
    def max_size(self) -> float:
        """Return the maximum size of this cache directory."""
        pass

    @abstractmethod
    def make_room(self, size: float) -> None:
        """
        Make sufficient room in the cache directory for a model of max_size.

        :param size: Size required (GB)
        """
        pass

    @abstractmethod
    def cache_path(self, key: str) -> Path:
        """Return the path for a model with the indicated key."""
        pass
