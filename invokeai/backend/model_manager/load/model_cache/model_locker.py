"""
Base class and implementation of a class that moves models in and out of VRAM.
"""

from typing import Dict, Optional

import torch

from invokeai.backend.model_manager import AnyModel
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


class ModelLocker:
    def __init__(self, cache: ModelCache, cache_entry: CacheRecord):
        self._cache = cache
        self._cache_entry = cache_entry

    @property
    def model(self) -> AnyModel:
        """Return the model without moving it around."""
        return self._cache_entry.model

    def get_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the state dict (if any) for the cached model."""
        return self._cache_entry.state_dict

    def lock(self) -> AnyModel:
        """Move the model into the execution device (GPU) and lock it."""
        self._cache.lock(self._cache_entry.key)
        return self.model

    def unlock(self) -> None:
        """Unlock a model."""
        self._cache.unlock(self._cache_entry.key)
