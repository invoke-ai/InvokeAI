import gc
from logging import Logger
from typing import Dict, List, Optional

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
        execution_device: torch.device | str = "cuda",
        storage_device: torch.device | str = "cpu",
        lazy_offloading: bool = True,
        log_memory_usage: bool = False,
        logger: Optional[Logger] = None,
    ):
        """Initialize the model RAM cache.

        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        :param logger: InvokeAILogger to use (otherwise creates one)
        """
        # TODO(ryand): Think about what lazy_offloading should mean in the new model cache.
        self._lazy_offloading = lazy_offloading
        self._execution_device_working_mem_gb = execution_device_working_mem_gb
        self._execution_device: torch.device = torch.device(execution_device)
        self._storage_device: torch.device = torch.device(storage_device)
        self._logger = PrefixedLoggerAdapter(
            logger or InvokeAILogger.get_logger(self.__class__.__name__), "MODEL CACHE"
        )
        self._log_memory_usage = log_memory_usage
        self._stats: Optional[CacheStats] = None

        self._cached_models: Dict[str, CacheRecord] = {}
        self._cache_stack: List[str] = []

    @property
    def stats(self) -> Optional[CacheStats]:
        """Return collected CacheStats object."""
        return self._stats

    @stats.setter
    def stats(self, stats: CacheStats) -> None:
        """Set the CacheStats object for collecting cache statistics."""
        self._stats = stats

    def put(self, key: str, model: AnyModel) -> None:
        """Add a model to the cache."""
        if key in self._cached_models:
            self._logger.debug(
                f"Attempted to add model {key} ({model.__class__.__name__}), but it already exists in the cache. No action necessary."
            )
            return

        size = calc_model_size_by_data(self._logger, model)
        self.make_room(size)

        running_on_cpu = self._execution_device.type == "cpu"

        # Wrap model.
        if isinstance(model, torch.nn.Module) and not running_on_cpu:
            wrapped_model = CachedModelWithPartialLoad(model, self._execution_device)
        else:
            wrapped_model = CachedModelOnlyFullLoad(model, self._execution_device, size)

        cache_record = CacheRecord(key=key, cached_model=wrapped_model)
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)
        self._logger.debug(
            f"Added model {key} (Type: {model.__class__.__name__}, Wrap mode: {wrapped_model.__class__.__name__}, Model size: {size/MB:.2f}MB)"
        )

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

        # this moves the entry to the top (right end) of the stack
        self._cache_stack = [k for k in self._cache_stack if k != key]
        self._cache_stack.append(key)

        self._logger.debug(f"Cache hit: {key} (Type: {cache_entry.cached_model.model.__class__.__name__})")

        return cache_entry

    def lock(self, key: str) -> None:
        """Lock a model for use and move it into VRAM."""
        cache_entry = self._cached_models[key]
        cache_entry.lock()

        self._logger.debug(f"Locking model {key} (Type: {cache_entry.cached_model.model.__class__.__name__})")

        if self._execution_device.type == "cpu":
            # Models don't need to be loaded into VRAM if we're running on CPU.
            return

        try:
            self._load_locked_model(cache_entry)
            self._logger.debug(
                f"Finished locking model {key} (Type: {cache_entry.cached_model.model.__class__.__name__})"
            )
        except torch.cuda.OutOfMemoryError:
            self._logger.warning("Insufficient GPU memory to load model. Aborting")
            cache_entry.unlock()
            raise
        except Exception:
            cache_entry.unlock()
            raise

        self._log_cache_state()

    def unlock(self, key: str) -> None:
        """Unlock a model."""
        cache_entry = self._cached_models[key]
        cache_entry.unlock()
        self._logger.debug(f"Unlocked model {key} (Type: {cache_entry.cached_model.model.__class__.__name__})")

    def _load_locked_model(self, cache_entry: CacheRecord) -> None:
        """Helper function for self.lock(). Loads a locked model into VRAM."""
        vram_available = self._get_vram_available()

        # Calculate model_vram_needed, the amount of additional VRAM that will be used if we fully load the model into
        # VRAM.
        model_cur_vram_bytes = cache_entry.cached_model.cur_vram_bytes()
        model_total_bytes = cache_entry.cached_model.total_bytes()
        model_vram_needed = model_total_bytes - model_cur_vram_bytes

        # The amount of VRAM that must be freed to make room for model_vram_needed.
        vram_bytes_to_free = max(0, model_vram_needed - vram_available)

        self._logger.debug(
            f"Before unloading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

        # Make room for the model in VRAM.
        # 1. If the model can fit entirely in VRAM, then make enough room for it to be loaded fully.
        # 2. If the model can't fit fully into VRAM, then unload all other models and load as much of the model as
        #    possible.
        vram_bytes_freed = self._offload_unlocked_models(vram_bytes_to_free)
        self._logger.debug(f"Unloaded models (if necessary): vram_bytes_freed={(vram_bytes_freed/MB):.2f}MB")

        # Check the updated vram_available after offloading.
        vram_available = self._get_vram_available()
        self._logger.debug(
            f"After unloading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

        # Move as much of the model as possible into VRAM.
        model_bytes_loaded = 0
        if isinstance(cache_entry.cached_model, CachedModelWithPartialLoad):
            model_bytes_loaded = cache_entry.cached_model.partial_load_to_vram(vram_available)
        elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
            # Partial load is not supported, so we have not choice but to try and fit it all into VRAM.
            model_bytes_loaded = cache_entry.cached_model.full_load_to_vram()
        else:
            raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")

        model_cur_vram_bytes = cache_entry.cached_model.cur_vram_bytes()
        vram_available = self._get_vram_available()
        self._logger.debug(f"Loaded model onto execution device: model_bytes_loaded={(model_bytes_loaded/MB):.2f}MB, ")
        self._logger.debug(
            f"After loading: {self._get_vram_state_str(model_cur_vram_bytes, model_total_bytes, vram_available)}"
        )

    def _get_vram_available(self) -> int:
        """Calculate the amount of additional VRAM available for the cache to use (takes into account the working
        memory).
        """
        if self._execution_device.type == "cuda":
            vram_reserved = torch.cuda.memory_reserved(self._execution_device)
            vram_free, _vram_total = torch.cuda.mem_get_info(self._execution_device)
            vram_available_to_process = vram_free + vram_reserved
        elif self._execution_device.type == "mps":
            vram_reserved = torch.mps.driver_allocated_memory()
            # TODO(ryand): Is it accurate that MPS shares memory with the CPU?
            vram_free = psutil.virtual_memory().available
            vram_available_to_process = vram_free + vram_reserved
        else:
            raise ValueError(f"Unsupported execution device: {self._execution_device.type}")

        vram_total_available_to_cache = vram_available_to_process - int(self._execution_device_working_mem_gb * GB)
        vram_cur_available_to_cache = vram_total_available_to_cache - self._get_vram_in_use()
        return vram_cur_available_to_cache

    def _get_vram_in_use(self) -> int:
        """Get the amount of VRAM currently in use by the cache."""
        return sum(ce.cached_model.cur_vram_bytes() for ce in self._cached_models.values())

    def _get_ram_available(self) -> int:
        """Get the amount of RAM available for the cache to use, while keeping memory pressure under control."""
        virtual_memory = psutil.virtual_memory()
        ram_total = virtual_memory.total
        ram_available = virtual_memory.available
        ram_used = ram_total - ram_available
        # Aim to keep 10% of RAM free.
        return int(ram_total * 0.9) - ram_used

    def _get_ram_in_use(self) -> int:
        """Get the amount of RAM currently in use."""
        return sum(ce.cached_model.total_bytes() for ce in self._cached_models.values())

    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self._log_memory_usage:
            return MemorySnapshot.capture()
        return None

    def _get_vram_state_str(self, model_cur_vram_bytes: int, model_total_bytes: int, vram_available: int) -> str:
        """Helper function for preparing a VRAM state log string."""
        model_cur_vram_bytes_percent = model_cur_vram_bytes / model_total_bytes if model_total_bytes > 0 else 0
        return (
            f"model_total={model_total_bytes/MB:.0f} MB, "
            + f"model_vram={model_cur_vram_bytes/MB:.0f} MB ({model_cur_vram_bytes_percent:.1%} %), "
            # + f"vram_total={int(self._max_vram_cache_size * GB)/MB:.0f} MB, "
            + f"vram_available={(vram_available/MB):.0f} MB, "
        )

    def _offload_unlocked_models(self, vram_bytes_to_free: int) -> int:
        """Offload models from the execution_device until vram_bytes_to_free bytes are freed, or all models are
        offloaded. Of course, locked models are not offloaded.

        Returns:
            int: The number of bytes freed.
        """
        self._logger.debug(f"Offloading unlocked models with goal of freeing {vram_bytes_to_free/MB:.2f}MB of VRAM.")
        vram_bytes_freed = 0
        # TODO(ryand): Give more thought to the offloading policy used here.
        cache_entries_increasing_size = sorted(self._cached_models.values(), key=lambda x: x.cached_model.total_bytes())
        for cache_entry in cache_entries_increasing_size:
            if vram_bytes_freed >= vram_bytes_to_free:
                break
            if cache_entry.is_locked:
                continue

            if isinstance(cache_entry.cached_model, CachedModelWithPartialLoad):
                cache_entry_bytes_freed = cache_entry.cached_model.partial_unload_from_vram(
                    vram_bytes_to_free - vram_bytes_freed
                )
            elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
                cache_entry_bytes_freed = cache_entry.cached_model.full_unload_from_vram()
            else:
                raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")
            if cache_entry_bytes_freed > 0:
                self._logger.debug(
                    f"Unloaded {cache_entry.key} from VRAM to free {(cache_entry_bytes_freed/MB):.0f} MB."
                )
            vram_bytes_freed += cache_entry_bytes_freed

        TorchDevice.empty_cache()
        return vram_bytes_freed

    # def _move_model_to_device(self, cache_entry: CacheRecord, target_device: torch.device) -> None:
    #     """Move model into the indicated device.

    #     :param cache_entry: The CacheRecord for the model
    #     :param target_device: The torch.device to move the model into

    #     May raise a torch.cuda.OutOfMemoryError
    #     """
    #     self._logger.debug(f"Called to move {cache_entry.key} to {target_device}")
    #     source_device = cache_entry.device

    #     # Note: We compare device types only so that 'cuda' == 'cuda:0'.
    #     # This would need to be revised to support multi-GPU.
    #     if torch.device(source_device).type == torch.device(target_device).type:
    #         return

    #     # Some models don't have a `to` method, in which case they run in RAM/CPU.
    #     if not hasattr(cache_entry.model, "to"):
    #         return

    #     # This roundabout method for moving the model around is done to avoid
    #     # the cost of moving the model from RAM to VRAM and then back from VRAM to RAM.
    #     # When moving to VRAM, we copy (not move) each element of the state dict from
    #     # RAM to a new state dict in VRAM, and then inject it into the model.
    #     # This operation is slightly faster than running `to()` on the whole model.
    #     #
    #     # When the model needs to be removed from VRAM we simply delete the copy
    #     # of the state dict in VRAM, and reinject the state dict that is cached
    #     # in RAM into the model. So this operation is very fast.
    #     start_model_to_time = time.time()
    #     snapshot_before = self._capture_memory_snapshot()

    #     try:
    #         if cache_entry.state_dict is not None:
    #             assert hasattr(cache_entry.model, "load_state_dict")
    #             if target_device == self._storage_device:
    #                 cache_entry.model.load_state_dict(cache_entry.state_dict, assign=True)
    #             else:
    #                 new_dict: Dict[str, torch.Tensor] = {}
    #                 for k, v in cache_entry.state_dict.items():
    #                     new_dict[k] = v.to(target_device, copy=True)
    #                 cache_entry.model.load_state_dict(new_dict, assign=True)
    #         cache_entry.model.to(target_device)
    #         cache_entry.device = target_device
    #     except Exception as e:  # blow away cache entry
    #         self._delete_cache_entry(cache_entry)
    #         raise e

    #     snapshot_after = self._capture_memory_snapshot()
    #     end_model_to_time = time.time()
    #     self._logger.debug(
    #         f"Moved model '{cache_entry.key}' from {source_device} to"
    #         f" {target_device} in {(end_model_to_time-start_model_to_time):.2f}s."
    #         f"Estimated model size: {(cache_entry.size/GB):.3f} GB."
    #         f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
    #     )

    #     if (
    #         snapshot_before is not None
    #         and snapshot_after is not None
    #         and snapshot_before.vram is not None
    #         and snapshot_after.vram is not None
    #     ):
    #         vram_change = abs(snapshot_before.vram - snapshot_after.vram)

    #         # If the estimated model size does not match the change in VRAM, log a warning.
    #         if not math.isclose(
    #             vram_change,
    #             cache_entry.size,
    #             rel_tol=0.1,
    #             abs_tol=10 * MB,
    #         ):
    #             self._logger.debug(
    #                 f"Moving model '{cache_entry.key}' from {source_device} to"
    #                 f" {target_device} caused an unexpected change in VRAM usage. The model's"
    #                 " estimated size may be incorrect. Estimated model size:"
    #                 f" {(cache_entry.size/GB):.3f} GB.\n"
    #                 f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
    #             )

    def _log_cache_state(self, title: str = "Model cache state:", include_entry_details: bool = True):
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
            vram_available_bytes = self._get_vram_available()
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

    def make_room(self, bytes_needed: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size.

        Note: This function deletes all of the cache's internal references to a model in order to free it. If there are
        external references to the model, there's nothing that the cache can do about it, and those models will not be
        garbage-collected.
        """
        self._logger.debug(f"Making room for {bytes_needed/MB:.2f}MB of RAM.")
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
                    f"Dropping {model_key} from RAM cache to free {(cache_entry.cached_model.total_bytes()/MB):.2f}MB."
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
        self._logger.debug(f"Dropped {models_cleared} models to free {ram_bytes_freed/MB:.2f}MB of RAM.")
        self._log_cache_state(title="After dropping models:")

    def _delete_cache_entry(self, cache_entry: CacheRecord) -> None:
        self._cache_stack.remove(cache_entry.key)
        del self._cached_models[cache_entry.key]
