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
from typing import Dict, Optional

import torch

from invokeai.backend.model_manager import SubModelType
from invokeai.backend.model_manager.load.load_base import AnyModel, ModelLockerBase


@dataclass
class CacheStats(object):
    """Data object to record statistics on cache hits/misses."""

    hits: int = 0  # cache hits
    misses: int = 0  # cache misses
    high_watermark: int = 0  # amount of cache used
    in_cache: int = 0  # number of models in cache
    cleared: int = 0  # number of models cleared to make space
    cache_size: int = 0  # total size of cache
    loaded_model_sizes: Dict[str, int] = field(default_factory=dict)


@dataclass
class CacheRecord:
    """Elements of the cache."""

    key: str
    model: AnyModel
    size: int
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


class ModelCacheBase(ABC):
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

    @abstractmethod
    def offload_unlocked_models(self) -> None:
        """Offload from VRAM any models not actively in use."""
        pass

    @abstractmethod
    def move_model_to_device(self, cache_entry: CacheRecord, device: torch.device) -> None:
        """Move model into the indicated device."""
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
        model: AnyModel,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        """Store model under key and optional submodel_type."""
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
    ) -> ModelLockerBase:
        """
        Retrieve model locker object using key and optional submodel_type.

        This may return an UnknownModelException if the model is not in the cache.
        """
        pass

    @abstractmethod
    def exists(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
    ) -> bool:
        """Return true if the model identified by key and submodel_type is in the cache."""
        pass

    @abstractmethod
    def cache_size(self) -> int:
        """Get the total size of the models currently cached."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Return cache hit/miss/size statistics."""
        pass

    @abstractmethod
    def print_cuda_stats(self) -> None:
        """Log debugging information on CUDA usage."""
        pass
