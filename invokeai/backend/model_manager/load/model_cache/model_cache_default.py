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
import sys
import threading
from contextlib import contextmanager, suppress
from logging import Logger
from threading import BoundedSemaphore
from typing import Dict, Generator, List, Optional, Set

import torch

from invokeai.backend.model_manager import AnyModel, SubModelType
from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

from .model_cache_base import CacheRecord, CacheStats, ModelCacheBase, ModelLockerBase
from .model_locker import ModelLocker

# Maximum size of the cache, in gigs
# Default is roughly enough to hold three fp16 diffusers models in RAM simultaneously
DEFAULT_MAX_CACHE_SIZE = 6.0

# actual size of a gig
GIG = 1073741824

# Size of a MB in bytes.
MB = 2**20


class ModelCache(ModelCacheBase[AnyModel]):
    """Implementation of ModelCacheBase."""

    def __init__(
        self,
        max_cache_size: float = DEFAULT_MAX_CACHE_SIZE,
        storage_device: torch.device = torch.device("cpu"),
        execution_devices: Optional[Set[torch.device]] = None,
        precision: torch.dtype = torch.float16,
        sequential_offload: bool = False,
        sha_chunksize: int = 16777216,
        log_memory_usage: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the model RAM cache.

        :param max_cache_size: Maximum size of the RAM cache [6.0 GB]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param precision: Precision for loaded models [torch.float16]
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        """
        self._precision: torch.dtype = precision
        self._max_cache_size: float = max_cache_size
        self._storage_device: torch.device = storage_device
        self._ram_lock = threading.Lock()
        self._logger = logger or InvokeAILogger.get_logger(self.__class__.__name__)
        self._log_memory_usage = log_memory_usage
        self._stats: Optional[CacheStats] = None

        self._cached_models: Dict[str, CacheRecord[AnyModel]] = {}
        self._cache_stack: List[str] = []

        # device to thread id
        self._device_lock = threading.Lock()
        self._execution_devices: Dict[torch.device, int] = {
            x: 0 for x in TorchDevice.execution_devices()
        }
        self._free_execution_device = BoundedSemaphore(len(self._execution_devices))

        self.logger.info(
            f"Using rendering device(s): {', '.join(sorted([str(x) for x in self._execution_devices.keys()]))}"
        )

    @property
    def logger(self) -> Logger:
        """Return the logger used by the cache."""
        return self._logger

    @property
    def storage_device(self) -> torch.device:
        """Return the storage device (e.g. "CPU" for RAM)."""
        return self._storage_device

    @property
    def execution_devices(self) -> Set[torch.device]:
        """Return the set of available execution devices."""
        devices = self._execution_devices.keys()
        return set(devices)

    def get_execution_device(self) -> torch.device:
        """
        Return an execution device that has been reserved for current thread.

        Note that reservations are done using the current thread's TID.
        It would be better to do this using the session ID, but that involves
        too many detailed changes to model manager calls.

        May generate a ValueError if no GPU has been reserved.
        """
        current_thread = threading.current_thread().ident
        assert current_thread is not None
        assigned = [x for x, tid in self._execution_devices.items() if current_thread == tid]
        if not assigned:
            raise ValueError("No GPU has been reserved for the use of thread {current_thread}")
        return assigned[0]

    @contextmanager
    def reserve_execution_device(self, timeout: Optional[int] = None) -> Generator[torch.device, None, None]:
        """Reserve an execution device (e.g. GPU) for exclusive use by a generation thread.

        Note that the reservation is done using the current thread's TID.
        It would be better to do this using the session ID, but that involves
        too many detailed changes to model manager calls.
        """
        device = None
        with self._device_lock:
            current_thread = threading.current_thread().ident
            assert current_thread is not None

            # look for a device that has already been assigned to this thread
            assigned = [x for x, tid in self._execution_devices.items() if current_thread == tid]
            if assigned:
                device = assigned[0]

        # no device already assigned. Get one.
        if device is None:
            self._free_execution_device.acquire(timeout=timeout)
            with self._device_lock:
                free_device = [x for x, tid in self._execution_devices.items() if tid == 0]
                self._execution_devices[free_device[0]] = current_thread
                device = free_device[0]

        # we are outside the lock region now
        self.logger.info(f"Reserved torch device {device} for execution thread {current_thread}")

        # Tell TorchDevice to use this object to get the torch device.
        TorchDevice.set_model_cache(self)
        try:
            yield device
        finally:
            with self._device_lock:
                self.logger.info(f"Released torch device {device}")
                self._execution_devices[device] = 0
                self._free_execution_device.release()
                torch.cuda.empty_cache()

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
        with self._ram_lock:
            key = self._make_cache_key(key, submodel_type)
            if key in self._cached_models:
                return
            self.make_room(size)
            cache_record = CacheRecord(key, model=model, size=size)
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
        with self._ram_lock:
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

    def make_room(self, size: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size."""
        # calculate how much memory this model will require
        # multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = size
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
                    f"Removing {model_key} from RAM cache to free at least {(size/GIG):.2f} GB (-{(cache_entry.size/GIG):.2f} GB)"
                )
                current_size -= cache_entry.size
                models_cleared += 1
                self._delete_cache_entry(cache_entry)
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
        TorchDevice.empty_cache()
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

    def _delete_cache_entry(self, cache_entry: CacheRecord[AnyModel]) -> None:
        self._cache_stack.remove(cache_entry.key)
        del self._cached_models[cache_entry.key]

    @staticmethod
    def _device_name(device: torch.device) -> str:
        return f"{device.type}:{device.index}"
