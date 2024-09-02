# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development team
# TODO: Add Stalker's proper name to copyright
"""
Manage a RAM cache of diffusion/transformer models for fast switching.
They are moved between GPU VRAM and CPU RAM as necessary. If the cache
grows larger than a preset maximum, then the least recently used
model will be cleared and (re)loaded from disk when next needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, Generic, Optional, TypeVar

import torch

from invokeai.backend.model_manager.config import AnyModel, SubModelType


class ModelLockerBase(ABC):
    """Base class for the model locker used by the loader."""

    @abstractmethod
    def lock(self) -> AnyModel:
        """Lock the contained model and move it into VRAM."""
        pass

    @abstractmethod
    def unlock(self) -> None:
        """Unlock the contained model, and remove it from VRAM."""
        pass

    @abstractmethod
    def get_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the state dict (if any) for the cached model."""
        pass

    @property
    @abstractmethod
    def model(self) -> AnyModel:
        """Return the model."""
        pass


T = TypeVar("T")


@dataclass
class CacheRecord(Generic[T]):
    """
    Elements of the cache:

    key: Unique key for each model, same as used in the models database.
    model: Model in memory.
    state_dict: A read-only copy of the model's state dict in RAM. It will be
                used as a template for creating a copy in the VRAM.
    size: Size of the model
    loaded: True if the model's state dict is currently in VRAM

    Before a model is executed, the state_dict template is copied into VRAM,
    and then injected into the model. When the model is finished, the VRAM
    copy of the state dict is deleted, and the RAM version is reinjected
    into the model.

    The state_dict should be treated as a read-only attribute. Do not attempt
    to patch or otherwise modify it. Instead, patch the copy of the state_dict
    after it is loaded into the execution device (e.g. CUDA) using the `LoadedModel`
    context manager call `model_on_device()`.
    """

    key: str
    model: T
    device: torch.device
    state_dict: Optional[Dict[str, torch.Tensor]]
    size: int
    loaded: bool = False
    _locks: int = 0

    def lock(self) -> None:
        """Lock this record."""
        self._locks += 1

    def unlock(self) -> None:
        """Unlock this record."""
        self._locks -= 1
        assert self._locks >= 0

    @property
    def locked(self) -> bool:
        """Return true if record is locked."""
        return self._locks > 0


@dataclass
class CacheStats(object):
    """Collect statistics on cache performance."""

    hits: int = 0  # cache hits
    misses: int = 0  # cache misses
    high_watermark: int = 0  # amount of cache used
    in_cache: int = 0  # number of models in cache
    cleared: int = 0  # number of models cleared to make space
    cache_size: int = 0  # total size of cache
    loaded_model_sizes: Dict[str, int] = field(default_factory=dict)


class ModelCacheBase(ABC, Generic[T]):
    """Virtual base class for RAM model cache."""

    @property
    @abstractmethod
    def storage_device(self) -> torch.device:
        """Return the storage device (e.g. "CPU" for RAM)."""
        pass

    @property
    @abstractmethod
    def execution_device(self) -> torch.device:
        """Return the exection device (e.g. "cuda" for VRAM)."""
        pass

    @property
    @abstractmethod
    def lazy_offloading(self) -> bool:
        """Return true if the cache is configured to lazily offload models in VRAM."""
        pass

    @property
    @abstractmethod
    def max_cache_size(self) -> float:
        """Return the maximum size the RAM cache can grow to."""
        pass

    @max_cache_size.setter
    @abstractmethod
    def max_cache_size(self, value: float) -> None:
        """Set the cap on vram cache size."""

    @property
    @abstractmethod
    def max_vram_cache_size(self) -> float:
        """Return the maximum size the VRAM cache can grow to."""
        pass

    @max_vram_cache_size.setter
    @abstractmethod
    def max_vram_cache_size(self, value: float) -> float:
        """Set the maximum size the VRAM cache can grow to."""
        pass

    @abstractmethod
    def offload_unlocked_models(self, size_required: int) -> None:
        """Offload from VRAM any models not actively in use."""
        pass

    @abstractmethod
    def move_model_to_device(self, cache_entry: CacheRecord[AnyModel], target_device: torch.device) -> None:
        """Move model into the indicated device."""
        pass

    @property
    @abstractmethod
    def stats(self) -> Optional[CacheStats]:
        """Return collected CacheStats object."""
        pass

    @stats.setter
    @abstractmethod
    def stats(self, stats: CacheStats) -> None:
        """Set the CacheStats object for collectin cache statistics."""
        pass

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """Return the logger used by the cache."""
        pass

    @abstractmethod
    def make_room(self, size: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size."""
        pass

    @abstractmethod
    def put(
        self,
        key: str,
        model: T,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        """Store model under key and optional submodel_type."""
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        stats_name: Optional[str] = None,
    ) -> ModelLockerBase:
        """
        Retrieve model using key and optional submodel_type.

        :param key: Opaque model key
        :param submodel_type: Type of the submodel to fetch
        :param stats_name: A human-readable id for the model for the purposes of
        stats reporting.

        This may raise an IndexError if the model is not in the cache.
        """
        pass

    @abstractmethod
    def cache_size(self) -> int:
        """Get the total size of the models currently cached."""
        pass

    @abstractmethod
    def print_cuda_stats(self) -> None:
        """Log debugging information on CUDA usage."""
        pass
