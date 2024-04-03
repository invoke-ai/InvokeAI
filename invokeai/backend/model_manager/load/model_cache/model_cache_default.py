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

import gc
import math
import sys
import time
from contextlib import suppress
from logging import Logger
from typing import Dict, List, Optional

import torch

from invokeai.backend.model_manager import AnyModel, SubModelType
from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot, get_pretty_snapshot_diff
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.logging import InvokeAILogger

from .model_cache_base import CacheRecord, CacheStats, ModelCacheBase, ModelLockerBase
from .model_locker import ModelLocker

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


class ModelCache(ModelCacheBase[AnyModel]):
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
        self._stats: Optional[CacheStats] = None

        self._cached_models: Dict[str, CacheRecord[AnyModel]] = {}
        self._cache_stack: List[str] = []

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

    @property
    def max_cache_size(self) -> float:
        """Return the cap on cache size."""
        return self._max_cache_size

    @max_cache_size.setter
    def max_cache_size(self, value: float) -> None:
        """Set the cap on cache size."""
        self._max_cache_size = value

    @property
    def stats(self) -> Optional[CacheStats]:
        """Return collected CacheStats object."""
        return self._stats

    @stats.setter
    def stats(self, stats: CacheStats) -> None:
        """Set the CacheStats object for collectin cache statistics."""
        self._stats = stats

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
        size: int,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        """Store model under key and optional submodel_type."""
        key = self._make_cache_key(key, submodel_type)
        if key in self._cached_models:
            return
        self.make_room(size)
        cache_record = CacheRecord(key, model, size)
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)

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
        key = self._make_cache_key(key, submodel_type)
        if key in self._cached_models:
            if self.stats:
                self.stats.hits += 1
        else:
            if self.stats:
                self.stats.misses += 1
            raise IndexError(f"The model with key {key} is not in the cache.")

        cache_entry = self._cached_models[key]

        # more stats
        if self.stats:
            stats_name = stats_name or key
            self.stats.cache_size = int(self._max_cache_size * GIG)
            self.stats.high_watermark = max(self.stats.high_watermark, self.cache_size())
            self.stats.in_cache = len(self._cached_models)
            self.stats.loaded_model_sizes[stats_name] = max(
                self.stats.loaded_model_sizes.get(stats_name, 0), cache_entry.size
            )

        # this moves the entry to the top (right end) of the stack
        with suppress(Exception):
            self._cache_stack.remove(key)
        self._cache_stack.append(key)
        return ModelLocker(
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

    def offload_unlocked_models(self, size_required: int) -> None:
        """Move any unused models from VRAM."""
        reserved = self._max_vram_cache_size * GIG
        vram_in_use = torch.cuda.memory_allocated() + size_required
        self.logger.debug(f"{(vram_in_use/GIG):.2f}GB VRAM needed for models; max allowed={(reserved/GIG):.2f}GB")
        for _, cache_entry in sorted(self._cached_models.items(), key=lambda x: x[1].size):
            if vram_in_use <= reserved:
                break
            if not cache_entry.loaded:
                continue
            if not cache_entry.locked:
                self.move_model_to_device(cache_entry, self.storage_device)
                cache_entry.loaded = False
                vram_in_use = torch.cuda.memory_allocated() + size_required
                self.logger.debug(
                    f"Removing {cache_entry.key} from VRAM to free {(cache_entry.size/GIG):.2f}GB; vram free = {(torch.cuda.memory_allocated()/GIG):.2f}GB"
                )

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

    def move_model_to_device(self, cache_entry: CacheRecord[AnyModel], target_device: torch.device) -> None:
        """Move model into the indicated device.

        :param cache_entry: The CacheRecord for the model
        :param target_device: The torch.device to move the model into

        May raise a torch.cuda.OutOfMemoryError
        """
        # These attributes are not in the base ModelMixin class but in various derived classes.
        # Some models don't have these attributes, in which case they run in RAM/CPU.
        self.logger.debug(f"Called to move {cache_entry.key} to {target_device}")
        if not (hasattr(cache_entry.model, "device") and hasattr(cache_entry.model, "to")):
            return

        source_device = cache_entry.model.device

        # Note: We compare device types only so that 'cuda' == 'cuda:0'.
        # This would need to be revised to support multi-GPU.
        if torch.device(source_device).type == torch.device(target_device).type:
            return

        # may raise an exception here if insufficient GPU VRAM
        self._check_free_vram(target_device, cache_entry.size)

        start_model_to_time = time.time()
        snapshot_before = self._capture_memory_snapshot()
        cache_entry.model.to(target_device)
        snapshot_after = self._capture_memory_snapshot()
        end_model_to_time = time.time()
        self.logger.debug(
            f"Moved model '{cache_entry.key}' from {source_device} to"
            f" {target_device} in {(end_model_to_time-start_model_to_time):.2f}s."
            f"Estimated model size: {(cache_entry.size/GIG):.3f} GB."
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
        ram = "%4.2fG" % (self.cache_size() / GIG)

        in_ram_models = 0
        in_vram_models = 0
        locked_in_vram_models = 0
        for cache_record in self._cached_models.values():
            if hasattr(cache_record.model, "device"):
                if cache_record.model.device == self.storage_device:
                    in_ram_models += 1
                else:
                    in_vram_models += 1
                if cache_record.locked:
                    locked_in_vram_models += 1

                self.logger.debug(
                    f"Current VRAM/RAM usage: {vram}/{ram}; models_in_ram/models_in_vram(locked) ="
                    f" {in_ram_models}/{in_vram_models}({locked_in_vram_models})"
                )

    def make_room(self, model_size: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size."""
        # calculate how much memory this model will require
        # multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = model_size
        maximum_size = self.max_cache_size * GIG  # stored in GB, convert to bytes
        current_size = self.cache_size()

        if current_size + bytes_needed > maximum_size:
            self.logger.debug(
                f"Max cache size exceeded: {(current_size/GIG):.2f}/{self.max_cache_size:.2f} GB, need an additional"
                f" {(bytes_needed/GIG):.2f} GB"
            )

        self.logger.debug(f"Before making_room: cached_models={len(self._cached_models)}")

        pos = 0
        models_cleared = 0
        while current_size + bytes_needed > maximum_size and pos < len(self._cache_stack):
            model_key = self._cache_stack[pos]
            cache_entry = self._cached_models[model_key]

            refs = sys.getrefcount(cache_entry.model)

            # HACK: This is a workaround for a memory-management issue that we haven't tracked down yet. We are directly
            # going against the advice in the Python docs by using `gc.get_referrers(...)` in this way:
            # https://docs.python.org/3/library/gc.html#gc.get_referrers

            # manualy clear local variable references of just finished function calls
            # for some reason python don't want to collect it even by gc.collect() immidiately
            if refs > 2:
                while True:
                    cleared = False
                    for referrer in gc.get_referrers(cache_entry.model):
                        if type(referrer).__name__ == "frame":
                            # RuntimeError: cannot clear an executing frame
                            with suppress(RuntimeError):
                                referrer.clear()
                                cleared = True
                                # break

                    # repeat if referrers changes(due to frame clear), else exit loop
                    if cleared:
                        gc.collect()
                    else:
                        break

            device = cache_entry.model.device if hasattr(cache_entry.model, "device") else None
            self.logger.debug(
                f"Model: {model_key}, locks: {cache_entry._locks}, device: {device}, loaded: {cache_entry.loaded},"
                f" refs: {refs}"
            )

            # Expected refs:
            # 1 from cache_entry
            # 1 from getrefcount function
            # 1 from onnx runtime object
            if not cache_entry.locked and refs <= (3 if "onnx" in model_key else 2):
                self.logger.debug(
                    f"Removing {model_key} from RAM cache to free at least {(model_size/GIG):.2f} GB (-{(cache_entry.size/GIG):.2f} GB)"
                )
                current_size -= cache_entry.size
                models_cleared += 1
                del self._cache_stack[pos]
                del self._cached_models[model_key]
                del cache_entry

            else:
                pos += 1

        if models_cleared > 0:
            # There would likely be some 'garbage' to be collected regardless of whether a model was cleared or not, but
            # there is a significant time cost to calling `gc.collect()`, so we want to use it sparingly. (The time cost
            # is high even if no garbage gets collected.)
            #
            # Calling gc.collect(...) when a model is cleared seems like a good middle-ground:
            # - If models had to be cleared, it's a signal that we are close to our memory limit.
            # - If models were cleared, there's a good chance that there's a significant amount of garbage to be
            #   collected.
            #
            # Keep in mind that gc is only responsible for handling reference cycles. Most objects should be cleaned up
            # immediately when their reference count hits 0.
            if self.stats:
                self.stats.cleared = models_cleared
            gc.collect()

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

        self.logger.debug(f"After making room: cached_models={len(self._cached_models)}")

    def _check_free_vram(self, target_device: torch.device, needed_size: int) -> None:
        if target_device.type != "cuda":
            return
        vram_device = (  # mem_get_info() needs an indexed device
            target_device if target_device.index is not None else torch.device(str(target_device), index=0)
        )
        free_mem, _ = torch.cuda.mem_get_info(torch.device(vram_device))
        if needed_size > free_mem:
            raise torch.cuda.OutOfMemoryError
