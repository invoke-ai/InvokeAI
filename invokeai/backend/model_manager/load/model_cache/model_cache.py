import gc
import logging
import threading
import time
from functools import wraps
from logging import Logger
from typing import Any, Callable, Dict, List, Optional

import psutil
import torch

from invokeai.backend.model_manager import AnyModel, SubModelType
from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.prefix_logger_adapter import PrefixedLoggerAdapter

# Size of a GB in bytes.
GB = 2**30

# Size of a MB in bytes.
MB = 2**20


# TODO(ryand): Where should this go? The ModelCache shouldn't be concerned with submodels.
def get_model_cache_key(model_key: str, submodel_type: Optional[SubModelType] = None) -> str:
    """Get the cache key for a model based on the optional submodel type."""
    if submodel_type:
        return f"{model_key}:{submodel_type.value}"
    else:
        return model_key


def synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that applies the class's self._lock to the method."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:  # Automatically acquire and release the lock
            return method(self, *args, **kwargs)

    return wrapper


class ModelCache:
    """A cache for managing models in memory.

    The cache is based on two levels of model storage:
    - execution_device: The device where most models are executed (typically "cuda", "mps", or "cpu").
    - storage_device: The device where models are offloaded when not in active use (typically "cpu").

    The model cache is based on the following assumptions:
    - storage_device_mem_size > execution_device_mem_size
    - disk_to_storage_device_transfer_time >> storage_device_to_execution_device_transfer_time

    A copy of all models in the cache is always kept on the storage_device. A subset of the models also have a copy on
    the execution_device.

    Models are moved between the storage_device and the execution_device as necessary. Cache size limits are enforced
    on both the storage_device and the execution_device. The execution_device cache uses a smallest-first offload
    policy. The storage_device cache uses a least-recently-used (LRU) offload policy.

    Note: Neither of these offload policies has really been compared against alternatives. It's likely that different
    policies would be better, although the optimal policies are likely heavily dependent on usage patterns and HW
    configuration.

    The cache returns context manager generators designed to load the model into the execution device (often GPU) within
    the context, and unload outside the context.

    Example usage:
    ```
    cache = ModelCache(max_cache_size=7.5, max_vram_cache_size=6.0)
    with cache.get_model('runwayml/stable-diffusion-1-5') as SD1:
        do_something_on_gpu(SD1)
    ```
    """

    def __init__(
        self,
        execution_device_working_mem_gb: float,
        enable_partial_loading: bool,
        keep_ram_copy_of_weights: bool,
        max_ram_cache_size_gb: float | None = None,
        max_vram_cache_size_gb: float | None = None,
        execution_device: torch.device | str = "cuda",
        storage_device: torch.device | str = "cpu",
        log_memory_usage: bool = False,
        logger: Optional[Logger] = None,
    ):
        """Initialize the model RAM cache.

        :param execution_device_working_mem_gb: The amount of working memory to keep on the GPU (in GB) i.e. non-model
            VRAM.
        :param enable_partial_loading: Whether to enable partial loading of models.
        :param max_ram_cache_size_gb: The maximum amount of CPU RAM to use for model caching in GB. This parameter is
            kept to maintain compatibility with previous versions of the model cache, but should be deprecated in the
            future. If set, this parameter overrides the default cache size logic.
        :param max_vram_cache_size_gb: The amount of VRAM to use for model caching in GB. This parameter is kept to
            maintain compatibility with previous versions of the model cache, but should be deprecated in the future.
            If set, this parameter overrides the default cache size logic.
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        :param logger: InvokeAILogger to use (otherwise creates one)
        """
        self._enable_partial_loading = enable_partial_loading
        self._keep_ram_copy_of_weights = keep_ram_copy_of_weights
        self._execution_device_working_mem_gb = execution_device_working_mem_gb
        self._execution_device: torch.device = torch.device(execution_device)
        self._storage_device: torch.device = torch.device(storage_device)

        self._max_ram_cache_size_gb = max_ram_cache_size_gb
        self._max_vram_cache_size_gb = max_vram_cache_size_gb

        self._logger = PrefixedLoggerAdapter(
            logger or InvokeAILogger.get_logger(self.__class__.__name__), "MODEL CACHE"
        )
        self._log_memory_usage = log_memory_usage
        self._stats: Optional[CacheStats] = None

        self._cached_models: Dict[str, CacheRecord] = {}
        self._cache_stack: List[str] = []

        self._ram_cache_size_bytes = self._calc_ram_available_to_model_cache()

        # A lock applied to all public method calls to make the ModelCache thread-safe.
        # At the time of writing, the ModelCache should only be accessed from two threads:
        # - The graph execution thread
        # - Requests to empty the cache from a separate thread
        self._lock = threading.RLock()

    @property
    @synchronized
    def stats(self) -> Optional[CacheStats]:
        """Return collected CacheStats object."""
        return self._stats

    @stats.setter
    @synchronized
    def stats(self, stats: CacheStats) -> None:
        """Set the CacheStats object for collecting cache statistics."""
        self._stats = stats

    @synchronized
    def put(self, key: str, model: AnyModel) -> None:
        """Add a model to the cache."""
        if key in self._cached_models:
            self._logger.debug(
                f"Attempted to add model {key} ({model.__class__.__name__}), but it already exists in the cache. No action necessary."
            )
            return

        size = calc_model_size_by_data(self._logger, model)
        self.make_room(size)

        # Inject custom modules into the model.
        if isinstance(model, torch.nn.Module):
            apply_custom_layers_to_model(model)

        # Partial loading only makes sense on CUDA.
        # - When running on CPU, there is no 'loading' to do.
        # - When running on MPS, memory is shared with the CPU, so the default OS memory management already handles this
        #   well.
        running_with_cuda = self._execution_device.type == "cuda"

        # Wrap model.
        if isinstance(model, torch.nn.Module) and running_with_cuda and self._enable_partial_loading:
            wrapped_model = CachedModelWithPartialLoad(
                model, self._execution_device, keep_ram_copy=self._keep_ram_copy_of_weights
            )
        else:
            wrapped_model = CachedModelOnlyFullLoad(
                model, self._execution_device, size, keep_ram_copy=self._keep_ram_copy_of_weights
            )

        cache_record = CacheRecord(key=key, cached_model=wrapped_model)
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)
        self._logger.debug(
            f"Added model {key} (Type: {model.__class__.__name__}, Wrap mode: {wrapped_model.__class__.__name__}, Model size: {size / MB:.2f}MB)"
        )

    @synchronized
    def get(self, key: str, stats_name: Optional[str] = None) -> CacheRecord:
        """Retrieve a model from the cache.

        :param key: Model key
        :param stats_name: A human-readable id for the model for the purposes of stats reporting.

        Raises IndexError if the model is not in the cache.
        """
        if key in self._cached_models:
            if self.stats:
                self.stats.hits += 1
        else:
            if self.stats:
                self.stats.misses += 1
            self._logger.debug(f"Cache miss: {key}")
            raise IndexError(f"The model with key {key} is not in the cache.")

        cache_entry = self._cached_models[key]

        # more stats
        if self.stats:
            stats_name = stats_name or key
            self.stats.high_watermark = max(self.stats.high_watermark, self._get_ram_in_use())
            self.stats.in_cache = len(self._cached_models)
            self.stats.loaded_model_sizes[stats_name] = max(
                self.stats.loaded_model_sizes.get(stats_name, 0), cache_entry.cached_model.total_bytes()
            )

        # This moves the entry to the top (right end) of the stack.
        self._cache_stack = [k for k in self._cache_stack if k != key]
        self._cache_stack.append(key)

        self._logger.debug(f"Cache hit: {key} (Type: {cache_entry.cached_model.model.__class__.__name__})")
        return cache_entry

    @synchronized
    def lock(self, cache_entry: CacheRecord, working_mem_bytes: Optional[int]) -> None:
        """Lock a model for use and move it into VRAM."""
        if cache_entry.key not in self._cached_models:
            self._logger.info(
                f"Locking model cache entry {cache_entry.key} "
                f"(Type: {cache_entry.cached_model.model.__class__.__name__}), but it has already been dropped from "
                "the RAM cache. This is a sign that the model loading order is non-optimal in the invocation code "
                "(See https://github.com/invoke-ai/InvokeAI/issues/7513)."
            )
        # cache_entry = self._cached_models[key]
        cache_entry.lock()

        self._logger.debug(
            f"Locking model {cache_entry.key} (Type: {cache_entry.cached_model.model.__class__.__name__})"
        )

        if self._execution_device.type == "cpu":
            # Models don't need to be loaded into VRAM if we're running on CPU.
            return

        try:
            self._load_locked_model(cache_entry, working_mem_bytes)
            self._logger.debug(
                f"Finished locking model {cache_entry.key} (Type: {cache_entry.cached_model.model.__class__.__name__})"
            )
        except torch.cuda.OutOfMemoryError:
            self._logger.warning("Insufficient GPU memory to load model. Aborting")
            cache_entry.unlock()
            raise
        except Exception:
            cache_entry.unlock()
            raise

        self._log_cache_state()

    @synchronized
    def unlock(self, cache_entry: CacheRecord) -> None:
        """Unlock a model."""
        if cache_entry.key not in self._cached_models:
            self._logger.info(
                f"Unlocking model cache entry {cache_entry.key} "
                f"(Type: {cache_entry.cached_model.model.__class__.__name__}), but it has already been dropped from "
                "the RAM cache. This is a sign that the model loading order is non-optimal in the invocation code "
                "(See https://github.com/invoke-ai/InvokeAI/issues/7513)."
            )
        # cache_entry = self._cached_models[key]
        cache_entry.unlock()
        self._logger.debug(
            f"Unlocked model {cache_entry.key} (Type: {cache_entry.cached_model.model.__class__.__name__})"
        )

    def _load_locked_model(self, cache_entry: CacheRecord, working_mem_bytes: Optional[int] = None) -> None:
        """Helper function for self.lock(). Loads a locked model into VRAM."""
        start_time = time.time()

        # Calculate model_vram_needed, the amount of additional VRAM that will be used if we fully load the model into
        # VRAM.
        model_cur_vram_bytes = cache_entry.cached_model.cur_vram_bytes()
        model_total_bytes = cache_entry.cached_model.total_bytes()
        model_vram_needed = model_total_bytes - model_cur_vram_bytes

        vram_available = self._get_vram_available(working_mem_bytes)
        self._logger.debug(
            f"Before unloading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

        # Make room for the model in VRAM.
        # 1. If the model can fit entirely in VRAM, then make enough room for it to be loaded fully.
        # 2. If the model can't fit fully into VRAM, then unload all other models and load as much of the model as
        #    possible.
        vram_bytes_freed = self._offload_unlocked_models(model_vram_needed, working_mem_bytes)
        self._logger.debug(f"Unloaded models (if necessary): vram_bytes_freed={(vram_bytes_freed / MB):.2f}MB")

        # Check the updated vram_available after offloading.
        vram_available = self._get_vram_available(working_mem_bytes)
        self._logger.debug(
            f"After unloading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

        if vram_available < 0:
            # There is insufficient VRAM available. As a last resort, try to unload the model being locked from VRAM,
            # as it may still be loaded from a previous use.
            vram_bytes_freed_from_own_model = self._move_model_to_ram(cache_entry, -vram_available)
            vram_available = self._get_vram_available(working_mem_bytes)
            self._logger.debug(
                f"Unloaded {vram_bytes_freed_from_own_model / MB:.2f}MB from the model being locked ({cache_entry.key})."
            )

        # Move as much of the model as possible into VRAM.
        # For testing, only allow 10% of the model to be loaded into VRAM.
        # vram_available = int(model_vram_needed * 0.1)
        # We add 1 MB to the available VRAM to account for small errors in memory tracking (e.g. off-by-one). A fully
        # loaded model is much faster than a 95% loaded model.
        model_bytes_loaded = self._move_model_to_vram(cache_entry, vram_available + MB)

        model_cur_vram_bytes = cache_entry.cached_model.cur_vram_bytes()
        vram_available = self._get_vram_available(working_mem_bytes)
        loaded_percent = model_cur_vram_bytes / model_total_bytes if model_total_bytes > 0 else 0
        self._logger.info(
            f"Loaded model '{cache_entry.key}' ({cache_entry.cached_model.model.__class__.__name__}) onto "
            f"{self._execution_device.type} device in {(time.time() - start_time):.2f}s. "
            f"Total model size: {model_total_bytes / MB:.2f}MB, "
            f"VRAM: {model_cur_vram_bytes / MB:.2f}MB ({loaded_percent:.1%})"
        )
        self._logger.debug(
            f"Loaded model onto execution device: model_bytes_loaded={(model_bytes_loaded / MB):.2f}MB, "
        )
        self._logger.debug(
            f"After loading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

    def _move_model_to_vram(self, cache_entry: CacheRecord, vram_available: int) -> int:
        try:
            if isinstance(cache_entry.cached_model, CachedModelWithPartialLoad):
                return cache_entry.cached_model.partial_load_to_vram(vram_available)
            elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
                # Partial load is not supported, so we have not choice but to try and fit it all into VRAM.
                return cache_entry.cached_model.full_load_to_vram()
            else:
                raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                self._logger.warning("Insufficient GPU memory to load model. Aborting")
            # If an exception occurs, the model could be left in a bad state, so we delete it from the cache entirely.
            self._delete_cache_entry(cache_entry)
            raise

    def _move_model_to_ram(self, cache_entry: CacheRecord, vram_bytes_to_free: int) -> int:
        try:
            if isinstance(cache_entry.cached_model, CachedModelWithPartialLoad):
                return cache_entry.cached_model.partial_unload_from_vram(
                    vram_bytes_to_free, keep_required_weights_in_vram=cache_entry.is_locked
                )
            elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
                return cache_entry.cached_model.full_unload_from_vram()
            else:
                raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")
        except Exception:
            # If an exception occurs, the model could be left in a bad state, so we delete it from the cache entirely.
            self._delete_cache_entry(cache_entry)
            raise

    def _get_vram_available(self, working_mem_bytes: Optional[int]) -> int:
        """Calculate the amount of additional VRAM available for the cache to use (takes into account the working
        memory).
        """
        # If self._max_vram_cache_size_gb is set, then it overrides the default logic.
        if self._max_vram_cache_size_gb is not None:
            vram_total_available_to_cache = int(self._max_vram_cache_size_gb * GB)
            return vram_total_available_to_cache - self._get_vram_in_use()

        working_mem_bytes_default = int(self._execution_device_working_mem_gb * GB)
        working_mem_bytes = max(working_mem_bytes or working_mem_bytes_default, working_mem_bytes_default)

        if self._execution_device.type == "cuda":
            # TODO(ryand): It is debatable whether we should use memory_reserved() or memory_allocated() here.
            # memory_reserved() includes memory reserved by the torch CUDA memory allocator that may or may not be
            # re-used for future allocations. For now, we use memory_allocated() to be conservative.
            # vram_reserved = torch.cuda.memory_reserved(self._execution_device)
            vram_allocated = torch.cuda.memory_allocated(self._execution_device)
            vram_free, _vram_total = torch.cuda.mem_get_info(self._execution_device)
            vram_available_to_process = vram_free + vram_allocated
        elif self._execution_device.type == "mps":
            vram_reserved = torch.mps.driver_allocated_memory()
            # TODO(ryand): Is it accurate that MPS shares memory with the CPU?
            vram_free = psutil.virtual_memory().available
            vram_available_to_process = vram_free + vram_reserved
        else:
            raise ValueError(f"Unsupported execution device: {self._execution_device.type}")

        vram_total_available_to_cache = vram_available_to_process - working_mem_bytes
        vram_cur_available_to_cache = vram_total_available_to_cache - self._get_vram_in_use()
        return vram_cur_available_to_cache

    def _get_vram_in_use(self) -> int:
        """Get the amount of VRAM currently in use by the cache."""
        if self._execution_device.type == "cuda":
            return torch.cuda.memory_allocated()
        elif self._execution_device.type == "mps":
            return torch.mps.current_allocated_memory()
        else:
            raise ValueError(f"Unsupported execution device type: {self._execution_device.type}")
        # Alternative definition of VRAM in use:
        # return sum(ce.cached_model.cur_vram_bytes() for ce in self._cached_models.values())

    def _calc_ram_available_to_model_cache(self) -> int:
        """Calculate the amount of RAM available for the cache to use."""
        # If self._max_ram_cache_size_gb is set, then it overrides the default logic.
        if self._max_ram_cache_size_gb is not None:
            self._logger.info(f"Using user-defined RAM cache size: {self._max_ram_cache_size_gb} GB.")
            return int(self._max_ram_cache_size_gb * GB)

        # Heuristics for dynamically calculating the RAM cache size, **in order of increasing priority**:
        # 1. As an initial default, use 50% of the total RAM for InvokeAI.
        #   - Assume a 2GB baseline for InvokeAI's non-model RAM usage, and use the rest of the RAM for the model cache.
        # 2. On a system with a lot of RAM, users probably don't want InvokeAI to eat up too much RAM.
        #    There are diminishing returns to storing more and more models. So, we apply an upper bound. (Keep in mind
        #    that most OSes have some amount of disk caching, which we still benefit from if there is excess memory,
        #    even if we drop models from the cache.)
        #    - On systems without a CUDA device, the upper bound is 32GB.
        #    - On systems with a CUDA device, the upper bound is 1x the amount of VRAM (less the working memory).
        # 3. Absolute minimum of 4GB.

        # NOTE(ryand): We explored dynamically adjusting the RAM cache size based on memory pressure (using psutil), but
        # decided against it for now, for the following reasons:
        # - It was surprisingly difficult to get memory metrics with consistent definitions across OSes. (If you go
        #   down this path again, don't underestimate the amount of complexity here and be sure to test rigorously on all
        #   OSes.)
        # - Making the RAM cache size dynamic opens the door for performance regressions that are hard to diagnose and
        #   hard for users to understand. It is better for users to see that their RAM is maxed out, and then override
        #   the default value if desired.

        # Lookup the total VRAM size for the CUDA execution device.
        total_cuda_vram_bytes: int | None = None
        if self._execution_device.type == "cuda":
            _, total_cuda_vram_bytes = torch.cuda.mem_get_info(self._execution_device)

        # Apply heuristic 1.
        # ------------------
        heuristics_applied = [1]
        total_system_ram_bytes = psutil.virtual_memory().total
        # Assumed baseline RAM used by InvokeAI for non-model stuff.
        baseline_ram_used_by_invokeai = 2 * GB
        ram_available_to_model_cache = int(total_system_ram_bytes * 0.5 - baseline_ram_used_by_invokeai)

        # Apply heuristic 2.
        # ------------------
        max_ram_cache_size_bytes = 32 * GB
        if total_cuda_vram_bytes is not None:
            if self._max_vram_cache_size_gb is not None:
                max_ram_cache_size_bytes = int(self._max_vram_cache_size_gb * GB)
            else:
                max_ram_cache_size_bytes = total_cuda_vram_bytes - int(self._execution_device_working_mem_gb * GB)
        if ram_available_to_model_cache > max_ram_cache_size_bytes:
            heuristics_applied.append(2)
            ram_available_to_model_cache = max_ram_cache_size_bytes

        # Apply heuristic 3.
        # ------------------
        if ram_available_to_model_cache < 4 * GB:
            heuristics_applied.append(3)
            ram_available_to_model_cache = 4 * GB

        self._logger.info(
            f"Calculated model RAM cache size: {ram_available_to_model_cache / MB:.2f} MB. Heuristics applied: {heuristics_applied}."
        )
        return ram_available_to_model_cache

    def _get_ram_in_use(self) -> int:
        """Get the amount of RAM currently in use."""
        return sum(ce.cached_model.total_bytes() for ce in self._cached_models.values())

    def _get_ram_available(self) -> int:
        """Get the amount of RAM available for the cache to use."""
        return self._ram_cache_size_bytes - self._get_ram_in_use()

    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self._log_memory_usage:
            return MemorySnapshot.capture()
        return None

    def _get_vram_state_str(self, model_cur_vram_bytes: int, model_total_bytes: int, vram_available: int) -> str:
        """Helper function for preparing a VRAM state log string."""
        model_cur_vram_bytes_percent = model_cur_vram_bytes / model_total_bytes if model_total_bytes > 0 else 0
        return (
            f"model_total={model_total_bytes / MB:.0f} MB, "
            + f"model_vram={model_cur_vram_bytes / MB:.0f} MB ({model_cur_vram_bytes_percent:.1%} %), "
            # + f"vram_total={int(self._max_vram_cache_size * GB)/MB:.0f} MB, "
            + f"vram_available={(vram_available / MB):.0f} MB, "
        )

    def _offload_unlocked_models(self, vram_bytes_required: int, working_mem_bytes: Optional[int] = None) -> int:
        """Offload models from the execution_device until vram_bytes_required bytes are available, or all models are
        offloaded. Of course, locked models are not offloaded.

        Returns:
            int: The number of bytes freed based on believed model sizes. The actual change in VRAM may be different.
        """
        self._logger.debug(
            f"Offloading unlocked models with goal of making room for {vram_bytes_required / MB:.2f}MB of VRAM."
        )
        vram_bytes_freed = 0
        # TODO(ryand): Give more thought to the offloading policy used here.
        cache_entries_increasing_size = sorted(self._cached_models.values(), key=lambda x: x.cached_model.total_bytes())
        for cache_entry in cache_entries_increasing_size:
            # We do not fully trust the count of bytes freed, so we check again on each iteration.
            vram_available = self._get_vram_available(working_mem_bytes)
            vram_bytes_to_free = vram_bytes_required - vram_available
            if vram_bytes_to_free <= 0:
                break
            if cache_entry.is_locked:
                # TODO(ryand): In the future, we may want to partially unload locked models, but this requires careful
                # handling of model patches (e.g. LoRA).
                continue
            cache_entry_bytes_freed = self._move_model_to_ram(cache_entry, vram_bytes_to_free)
            if cache_entry_bytes_freed > 0:
                self._logger.debug(
                    f"Unloaded {cache_entry.key} from VRAM to free {(cache_entry_bytes_freed / MB):.0f} MB."
                )
            vram_bytes_freed += cache_entry_bytes_freed

        TorchDevice.empty_cache()
        return vram_bytes_freed

    def _log_cache_state(self, title: str = "Model cache state:", include_entry_details: bool = True):
        if self._logger.getEffectiveLevel() > logging.DEBUG:
            # Short circuit if the logger is not set to debug. Some of the data lookups could take a non-negligible
            # amount of time.
            return

        log = f"{title}\n"

        log_format = "  {:<30} Limit: {:>7.1f} MB, Used: {:>7.1f} MB ({:>5.1%}), Available: {:>7.1f} MB ({:>5.1%})\n"

        ram_in_use_bytes = self._get_ram_in_use()
        ram_available_bytes = self._get_ram_available()
        ram_size_bytes = ram_in_use_bytes + ram_available_bytes
        ram_in_use_bytes_percent = ram_in_use_bytes / ram_size_bytes if ram_size_bytes > 0 else 0
        ram_available_bytes_percent = ram_available_bytes / ram_size_bytes if ram_size_bytes > 0 else 0
        log += log_format.format(
            f"Storage Device ({self._storage_device.type})",
            ram_size_bytes / MB,
            ram_in_use_bytes / MB,
            ram_in_use_bytes_percent,
            ram_available_bytes / MB,
            ram_available_bytes_percent,
        )

        if self._execution_device.type != "cpu":
            vram_in_use_bytes = self._get_vram_in_use()
            vram_available_bytes = self._get_vram_available(None)
            vram_size_bytes = vram_in_use_bytes + vram_available_bytes
            vram_in_use_bytes_percent = vram_in_use_bytes / vram_size_bytes if vram_size_bytes > 0 else 0
            vram_available_bytes_percent = vram_available_bytes / vram_size_bytes if vram_size_bytes > 0 else 0
            log += log_format.format(
                f"Compute Device ({self._execution_device.type})",
                vram_size_bytes / MB,
                vram_in_use_bytes / MB,
                vram_in_use_bytes_percent,
                vram_available_bytes / MB,
                vram_available_bytes_percent,
            )

        if torch.cuda.is_available():
            log += "  {:<30} {:.1f} MB\n".format("CUDA Memory Allocated:", torch.cuda.memory_allocated() / MB)
        log += "  {:<30} {}\n".format("Total models:", len(self._cached_models))

        if include_entry_details and len(self._cached_models) > 0:
            log += "  Models:\n"
            log_format = (
                "    {:<80} total={:>7.1f} MB, vram={:>7.1f} MB ({:>5.1%}), ram={:>7.1f} MB ({:>5.1%}), locked={}\n"
            )
            for cache_record in self._cached_models.values():
                total_bytes = cache_record.cached_model.total_bytes()
                cur_vram_bytes = cache_record.cached_model.cur_vram_bytes()
                cur_vram_bytes_percent = cur_vram_bytes / total_bytes if total_bytes > 0 else 0
                cur_ram_bytes = total_bytes - cur_vram_bytes
                cur_ram_bytes_percent = cur_ram_bytes / total_bytes if total_bytes > 0 else 0

                log += log_format.format(
                    f"{cache_record.key} ({cache_record.cached_model.model.__class__.__name__}):",
                    total_bytes / MB,
                    cur_vram_bytes / MB,
                    cur_vram_bytes_percent,
                    cur_ram_bytes / MB,
                    cur_ram_bytes_percent,
                    cache_record.is_locked,
                )

        self._logger.debug(log)

    @synchronized
    def make_room(self, bytes_needed: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size.

        Note: This function deletes all of the cache's internal references to a model in order to free it. If there are
        external references to the model, there's nothing that the cache can do about it, and those models will not be
        garbage-collected.
        """
        self._logger.debug(f"Making room for {bytes_needed / MB:.2f}MB of RAM.")
        self._log_cache_state(title="Before dropping models:")

        ram_bytes_available = self._get_ram_available()
        ram_bytes_to_free = max(0, bytes_needed - ram_bytes_available)

        ram_bytes_freed = 0
        pos = 0
        models_cleared = 0
        while ram_bytes_freed < ram_bytes_to_free and pos < len(self._cache_stack):
            model_key = self._cache_stack[pos]
            cache_entry = self._cached_models[model_key]

            if not cache_entry.is_locked:
                ram_bytes_freed += cache_entry.cached_model.total_bytes()
                self._logger.debug(
                    f"Dropping {model_key} from RAM cache to free {(cache_entry.cached_model.total_bytes() / MB):.2f}MB."
                )
                self._delete_cache_entry(cache_entry)
                del cache_entry
                models_cleared += 1
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
        self._logger.debug(f"Dropped {models_cleared} models to free {ram_bytes_freed / MB:.2f}MB of RAM.")
        self._log_cache_state(title="After dropping models:")

    def _delete_cache_entry(self, cache_entry: CacheRecord) -> None:
        """Delete cache_entry from the cache if it exists. No exception is thrown if it doesn't exist."""
        self._cache_stack = [key for key in self._cache_stack if key != cache_entry.key]
        self._cached_models.pop(cache_entry.key, None)
