"""
Base class and implementation of a class that moves models in and out of VRAM.
"""

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

    # ---------------------------- NOTE -----------------
    # Ryan suggests keeping a copy of the model's state dict in CPU and copying it
    # into the GPU with code like this:
    #
    # def state_dict_to(state_dict: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    #    new_state_dict: dict[str, torch.Tensor] = {}
    #    for k, v in state_dict.items():
    #       new_state_dict[k] = v.to(device=device, copy=True, non_blocking=True)
    #    return new_state_dict
    #
    # I believe we'd then use load_state_dict() to inject the state dict into the model.
    # See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # ---------------------------- NOTE -----------------

    def lock(self) -> AnyModel:
        """Move the model into the execution device (GPU) and lock it."""
        if not hasattr(self.model, "to"):
            return self.model

        # NOTE that the model has to have the to() method in order for this code to move it into GPU!
        self._cache_entry.lock()
        try:
            if self._cache.lazy_offloading:
                self._cache.offload_unlocked_models(self._cache_entry.size)

            execution_device = self._cache.get_execution_device()
            self._cache.move_model_to_device(self._cache_entry, execution_device)
            self._cache_entry.loaded = True

            self._cache.logger.debug(f"Locking {self._cache_entry.key} in {execution_device}")
            self._cache.print_cuda_stats()
        except torch.cuda.OutOfMemoryError:
            self._cache.logger.warning("Insufficient GPU memory to load model. Aborting")
            self._cache_entry.unlock()
            raise
        except Exception:
            self._cache_entry.unlock()
            raise

        return self.model

    def unlock(self) -> None:
        """Call upon exit from context."""
        if not hasattr(self.model, "to"):
            return

        self._cache_entry.unlock()
        if not self._cache.lazy_offloading:
            self._cache.offload_unlocked_models(self._cache_entry.size)
            self._cache.print_cuda_stats()
