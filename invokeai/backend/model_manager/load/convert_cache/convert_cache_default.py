"""
Placeholder for convert cache implementation.
"""

import shutil
from pathlib import Path

from invokeai.backend.util import GIG, directory_size
from invokeai.backend.util.logging import InvokeAILogger

from .convert_cache_base import ModelConvertCacheBase


class ModelConvertCache(ModelConvertCacheBase):
    def __init__(self, cache_path: Path, max_size: float = 10.0):
        """Initialize the convert cache with the base directory and a limit on its maximum size (in GBs)."""
        if not cache_path.exists():
            cache_path.mkdir(parents=True)
        self._cache_path = cache_path
        self._max_size = max_size

    @property
    def max_size(self) -> float:
        """Return the maximum size of this cache directory (GB)."""
        return self._max_size

    def cache_path(self, key: str) -> Path:
        """Return the path for a model with the indicated key."""
        return self._cache_path / key

    def make_room(self, size: float) -> None:
        """
        Make sufficient room in the cache directory for a model of max_size.

        :param size: Size required (GB)
        """
        size_needed = directory_size(self._cache_path) + size
        max_size = int(self.max_size) * GIG
        logger = InvokeAILogger.get_logger()

        if size_needed <= max_size:
            return

        logger.debug(
            f"Convert cache has gotten too large {(size_needed / GIG):4.2f} > {(max_size / GIG):4.2f}G.. Trimming."
        )

        # For this to work, we make the assumption that the directory contains
        # a 'model_index.json', 'unet/config.json' file, or a 'config.json' file at top level.
        # This should be true for any diffusers model.
        def by_atime(path: Path) -> float:
            for config in ["model_index.json", "unet/config.json", "config.json"]:
                sentinel = path / config
                if sentinel.exists():
                    return sentinel.stat().st_atime
            return 0.0

        # sort by last access time - least accessed files will be at the end
        lru_models = sorted(self._cache_path.iterdir(), key=by_atime, reverse=True)
        logger.debug(f"cached models in descending atime order: {lru_models}")
        while size_needed > max_size and len(lru_models) > 0:
            next_victim = lru_models.pop()
            victim_size = directory_size(next_victim)
            logger.debug(f"Removing cached converted model {next_victim} to free {victim_size / GIG} GB")
            shutil.rmtree(next_victim)
            size_needed -= victim_size
