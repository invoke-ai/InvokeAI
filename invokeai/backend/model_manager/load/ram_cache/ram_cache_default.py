# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development team
# TODO: Add Stalker's proper name to copyright
"""
Manage a RAM cache of diffusion/transformer models for fast switching.
They are moved between GPU VRAM and CPU RAM as necessary. If the cache
grows larger than a preset maximum, then the least recently used
model will be cleared and (re)loaded from disk when next needed.

The cache returns context manager generators designed to load the
model into the GPU within the context, and unload outside the
context. Use like this:

   cache = ModelCache(max_cache_size=7.5)
   with cache.get_model('runwayml/stable-diffusion-1-5') as SD1,
          cache.get_model('stabilityai/stable-diffusion-2') as SD2:
       do_something_in_GPU(SD1,SD2)


"""

import math
import time
from contextlib import suppress
from logging import Logger
from typing import Any, Dict, List, Optional

import torch

from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager import SubModelType
from invokeai.backend.model_manager.load.load_base import AnyModel, ModelLockerBase
from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot, get_pretty_snapshot_diff
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data
from invokeai.backend.model_manager.load.ram_cache.ram_cache_base import CacheRecord, CacheStats, ModelCacheBase
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.logging import InvokeAILogger

if choose_torch_device() == torch.device("mps"):
    from torch import mps

# Maximum size of the cache, in gigs
# Default is roughly enough to hold three fp16 diffusers models in RAM simultaneously
DEFAULT_MAX_CACHE_SIZE = 6.0

# amount of GPU memory to hold in reserve for use by generations (GB)
DEFAULT_MAX_VRAM_CACHE_SIZE = 2.75

# actual size of a gig
GIG = 1073741824

# Size of a MB in bytes.
MB = 2**20


class ModelCache(ModelCacheBase):
    """Implementation of ModelCacheBase."""

    def __init__(
        self,
        max_cache_size: float = DEFAULT_MAX_CACHE_SIZE,
        max_vram_cache_size: float = DEFAULT_MAX_VRAM_CACHE_SIZE,
        execution_device: torch.device = torch.device("cuda"),
        storage_device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float16,
        sequential_offload: bool = False,
        lazy_offloading: bool = True,
        sha_chunksize: int = 16777216,
        log_memory_usage: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the model RAM cache.

        :param max_cache_size: Maximum size of the RAM cache [6.0 GB]
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param precision: Precision for loaded models [torch.float16]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        """
        # allow lazy offloading only when vram cache enabled
        self._lazy_offloading = lazy_offloading and max_vram_cache_size > 0
        self._precision: torch.dtype = precision
        self._max_cache_size: float = max_cache_size
        self._max_vram_cache_size: float = max_vram_cache_size
        self._execution_device: torch.device = execution_device
        self._storage_device: torch.device = storage_device
        self._logger = logger or InvokeAILogger.get_logger(self.__class__.__name__)
        self._log_memory_usage = log_memory_usage

        # used for stats collection
        self.stats = None

        self._cached_models: Dict[str, CacheRecord] = {}
        self._cache_stack: List[str] = []

    class ModelLocker(ModelLockerBase):
        """Internal class that mediates movement in and out of GPU."""

        def __init__(self, cache: ModelCacheBase, cache_entry: CacheRecord):
            """
            Initialize the model locker.

            :param cache: The ModelCache object
            :param cache_entry: The entry in the model cache
            """
            self._cache = cache
            self._cache_entry = cache_entry

        @property
        def model(self) -> AnyModel:
            """Return the model without moving it around."""
            return self._cache_entry.model

        def lock(self) -> Any:
            """Move the model into the execution device (GPU) and lock it."""
            if not hasattr(self.model, "to"):
                return self.model

            # NOTE that the model has to have the to() method in order for this code to move it into GPU!
            self._cache_entry.lock()

            try:
                if self._cache.lazy_offloading:
                    self._cache.offload_unlocked_models()

                self._cache.move_model_to_device(self._cache_entry, self._cache.execution_device)

                self._cache.logger.debug(f"Locking {self._cache_entry.key} in {self._cache.execution_device}")
                self._cache.print_cuda_stats()

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
                self._cache.offload_unlocked_models()
                self._cache.print_cuda_stats()

    @property
    def logger(self) -> Logger:
        """Return the logger used by the cache."""
        return self._logger

    @property
    def lazy_offloading(self) -> bool:
        """Return true if the cache is configured to lazily offload models in VRAM."""
        return self._lazy_offloading

    @property
    def storage_device(self) -> torch.device:
        """Return the storage device (e.g. "CPU" for RAM)."""
        return self._storage_device

    @property
    def execution_device(self) -> torch.device:
        """Return the exection device (e.g. "cuda" for VRAM)."""
        return self._execution_device

    def cache_size(self) -> int:
        """Get the total size of the models currently cached."""
        total = 0
        for cache_record in self._cached_models.values():
            total += cache_record.size
        return total

    def exists(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
    ) -> bool:
        """Return true if the model identified by key and submodel_type is in the cache."""
        key = self._make_cache_key(key, submodel_type)
        return key in self._cached_models

    def put(
        self,
        key: str,
        model: AnyModel,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        """Store model under key and optional submodel_type."""
        key = self._make_cache_key(key, submodel_type)
        assert key not in self._cached_models

        loaded_model_size = calc_model_size_by_data(model)
        cache_record = CacheRecord(key, model, loaded_model_size)
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)

    def get(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
    ) -> ModelLockerBase:
        """
        Retrieve model using key and optional submodel_type.

        This may return an UnknownModelException if the model is not in the cache.
        """
        key = self._make_cache_key(key, submodel_type)
        if key not in self._cached_models:
            raise UnknownModelException

        # this moves the entry to the top (right end) of the stack
        with suppress(Exception):
            self._cache_stack.remove(key)
        self._cache_stack.append(key)
        cache_entry = self._cached_models[key]
        return self.ModelLocker(
            cache=self,
            cache_entry=cache_entry,
        )

    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self._log_memory_usage:
            return MemorySnapshot.capture()
        return None

    def _make_cache_key(self, model_key: str, submodel_type: Optional[SubModelType] = None) -> str:
        if submodel_type:
            return f"{model_key}:{submodel_type.value}"
        else:
            return model_key

    def offload_unlocked_models(self) -> None:
        """Move any unused models from VRAM."""
        reserved = self._max_vram_cache_size * GIG
        vram_in_use = torch.cuda.memory_allocated()
        self.logger.debug(f"{(vram_in_use/GIG):.2f}GB VRAM used for models; max allowed={(reserved/GIG):.2f}GB")
        for _, cache_entry in sorted(self._cached_models.items(), key=lambda x: x[1].size):
            if vram_in_use <= reserved:
                break
            if not cache_entry.locked:
                self.move_model_to_device(cache_entry, self.storage_device)

                vram_in_use = torch.cuda.memory_allocated()
                self.logger.debug(f"{(vram_in_use/GIG):.2f}GB VRAM used for models; max allowed={(reserved/GIG):.2f}GB")

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

    # TO DO: Only reason to pass the CacheRecord rather than the model is to get the key and size
    # for printing debugging messages. Revisit whether this is necessary
    def move_model_to_device(self, cache_entry: CacheRecord, target_device: torch.device) -> None:
        """Move model into the indicated device."""
        # These attributes are not in the base class but in derived classes
        assert hasattr(cache_entry.model, "device")
        assert hasattr(cache_entry.model, "to")

        source_device = cache_entry.model.device

        # Note: We compare device types only so that 'cuda' == 'cuda:0'. This would need to be revised to support
        # multi-GPU.
        if torch.device(source_device).type == torch.device(target_device).type:
            return

        start_model_to_time = time.time()
        snapshot_before = self._capture_memory_snapshot()
        cache_entry.model.to(target_device)
        snapshot_after = self._capture_memory_snapshot()
        end_model_to_time = time.time()
        self.logger.debug(
            f"Moved model '{cache_entry.key}' from {source_device} to"
            f" {target_device} in {(end_model_to_time-start_model_to_time):.2f}s.\n"
            f"Estimated model size: {(cache_entry.size/GIG):.3f} GB.\n"
            f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
        )

        if (
            snapshot_before is not None
            and snapshot_after is not None
            and snapshot_before.vram is not None
            and snapshot_after.vram is not None
        ):
            vram_change = abs(snapshot_before.vram - snapshot_after.vram)

            # If the estimated model size does not match the change in VRAM, log a warning.
            if not math.isclose(
                vram_change,
                cache_entry.size,
                rel_tol=0.1,
                abs_tol=10 * MB,
            ):
                self.logger.debug(
                    f"Moving model '{cache_entry.key}' from {source_device} to"
                    f" {target_device} caused an unexpected change in VRAM usage. The model's"
                    " estimated size may be incorrect. Estimated model size:"
                    f" {(cache_entry.size/GIG):.3f} GB.\n"
                    f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
                )

    def print_cuda_stats(self) -> None:
        """Log CUDA diagnostics."""
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GIG)
        ram = "%4.2fG" % self.cache_size()

        cached_models = 0
        loaded_models = 0
        locked_models = 0
        for cache_record in self._cached_models.values():
            cached_models += 1
            assert hasattr(cache_record.model, "device")
            if cache_record.model.device is self.storage_device:
                loaded_models += 1
            if cache_record.locked:
                locked_models += 1

        self.logger.debug(
            f"Current VRAM/RAM usage: {vram}/{ram}; cached_models/loaded_models/locked_models/ ="
            f" {cached_models}/{loaded_models}/{locked_models}"
        )

    def get_stats(self) -> CacheStats:
        """Return cache hit/miss/size statistics."""
        raise NotImplementedError

    def make_room(self, size: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size."""
        raise NotImplementedError
