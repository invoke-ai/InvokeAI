"""
Base class and implementation of a class that moves models in and out of VRAM.
"""

import copy
from typing import Optional

import torch

from invokeai.backend.model_manager import AnyModel

from .model_cache_base import CacheRecord, ModelCacheBase, ModelLockerBase

MAX_GPU_WAIT = 600  # wait up to 10 minutes for a GPU to become free


class ModelLocker(ModelLockerBase):
    """Internal class that mediates movement in and out of GPU."""

    def __init__(self, cache: ModelCacheBase[AnyModel], cache_entry: CacheRecord[AnyModel]):
        """
        Initialize the model locker.

        :param cache: The ModelCache object
        :param cache_entry: The entry in the model cache
        """
        self._cache = cache
        self._cache_entry = cache_entry
        self._execution_device: Optional[torch.device] = None

    @property
    def model(self) -> AnyModel:
        """Return the model without moving it around."""
        return self._cache_entry.model

    def lock(self) -> AnyModel:
        """Move the model into the execution device (GPU) and lock it."""
        if not hasattr(self.model, "to"):
            return self.model

        # NOTE that the model has to have the to() method in order for this code to move it into GPU!
        self._cache_entry.lock()

        try:
            # We wait for a gpu to be free - may raise a TimeoutError
            self._execution_device = self._cache.acquire_execution_device(MAX_GPU_WAIT)
            self._cache.logger.debug(f"Locking {self._cache_entry.key} in {self._execution_device}")
            model_in_gpu = copy.deepcopy(self._cache_entry.model)
            if hasattr(model_in_gpu, "to"):
                model_in_gpu.to(self._execution_device)
            self._cache_entry.loaded = True
            self._cache.print_cuda_stats()
        except torch.cuda.OutOfMemoryError:
            self._cache.logger.warning("Insufficient GPU memory to load model. Aborting")
            self._cache_entry.unlock()
            raise
        except Exception:
            self._cache_entry.unlock()
            raise
        return model_in_gpu

    def unlock(self) -> None:
        """Call upon exit from context."""
        if not hasattr(self.model, "to"):
            return

        self._cache_entry.unlock()
        if self._execution_device:
            self._cache.release_execution_device(self._execution_device)

        try:
            torch.cuda.empty_cache()
            torch.mps.empty_cache()
        except Exception:
            pass
        self._cache.print_cuda_stats()
