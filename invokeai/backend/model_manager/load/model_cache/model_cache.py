import gc
from logging import Logger
from typing import Dict, List, Optional

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

# Size of a GB in bytes.
GB = 2**30

# Size of a MB in bytes.
MB = 2**20


def get_model_cache_key(model_key: str, submodel_type: Optional[SubModelType] = None) -> str:
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
        max_cache_size: float,
        max_vram_cache_size: float,
        execution_device: torch.device = torch.device("cuda"),
        storage_device: torch.device = torch.device("cpu"),
        lazy_offloading: bool = True,
        log_memory_usage: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the model RAM cache.

        :param max_cache_size: Maximum size of the storage_device cache in GBs.
        :param max_vram_cache_size: Maximum size of the execution_device cache in GBs.
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        :param logger: InvokeAILogger to use (otherwise creates one)
        """
        # allow lazy offloading only when vram cache enabled
        self._lazy_offloading = lazy_offloading and max_vram_cache_size > 0
        self._max_cache_size: float = max_cache_size
        self._max_vram_cache_size: float = max_vram_cache_size
        self._execution_device: torch.device = execution_device
        self._storage_device: torch.device = storage_device
        self._logger = logger or InvokeAILogger.get_logger(self.__class__.__name__)
        self._log_memory_usage = log_memory_usage
        self._stats: Optional[CacheStats] = None

        self._cached_models: Dict[str, CacheRecord] = {}
        self._cache_stack: List[str] = []

    @property
    def max_cache_size(self) -> float:
        """Return the cap on cache size."""
        return self._max_cache_size

    @max_cache_size.setter
    def max_cache_size(self, value: float) -> None:
        """Set the cap on cache size."""
        self._max_cache_size = value

    @property
    def max_vram_cache_size(self) -> float:
        """Return the cap on vram cache size."""
        return self._max_vram_cache_size

    @max_vram_cache_size.setter
    def max_vram_cache_size(self, value: float) -> None:
        """Set the cap on vram cache size."""
        self._max_vram_cache_size = value

    @property
    def stats(self) -> Optional[CacheStats]:
        """Return collected CacheStats object."""
        return self._stats

    @stats.setter
    def stats(self, stats: CacheStats) -> None:
        """Set the CacheStats object for collectin cache statistics."""
        self._stats = stats

    def put(self, key: str, model: AnyModel) -> None:
        if key in self._cached_models:
            return
        size = calc_model_size_by_data(self._logger, model)
        self.make_room(size)

        # Wrap model.
        if isinstance(model, torch.nn.Module):
            wrapped_model = CachedModelWithPartialLoad(model, self._execution_device)
        else:
            wrapped_model = CachedModelOnlyFullLoad(model, self._execution_device, size)

        # running_on_cpu = self._execution_device == torch.device("cpu")
        # state_dict = model.state_dict() if isinstance(model, torch.nn.Module) and not running_on_cpu else None
        cache_record = CacheRecord(key=key, cached_model=wrapped_model)
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)

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
            raise IndexError(f"The model with key {key} is not in the cache.")

        cache_entry = self._cached_models[key]

        # more stats
        if self.stats:
            stats_name = stats_name or key
            self.stats.cache_size = int(self._max_cache_size * GB)
            self.stats.high_watermark = max(self.stats.high_watermark, self._get_ram_in_use())
            self.stats.in_cache = len(self._cached_models)
            self.stats.loaded_model_sizes[stats_name] = max(
                self.stats.loaded_model_sizes.get(stats_name, 0), cache_entry.cached_model.total_bytes()
            )

        # this moves the entry to the top (right end) of the stack
        self._cache_stack = [k for k in self._cache_stack if k != key]
        self._cache_stack.append(key)

        return cache_entry

    def lock(self, key: str) -> None:
        """Lock a model for use and move it into VRAM."""
        cache_entry = self._cached_models[key]
        cache_entry.lock()

        try:
            vram_available = self._get_vram_available()

            # The amount of additional VRAM that will be used if we fully load the model into VRAM.
            vram_needed_for_model = cache_entry.cached_model.total_bytes() - cache_entry.cached_model.cur_vram_bytes()

            # Make room for the model in VRAM.
            # 1. If the model can fit entirely in VRAM, then make enough room for it to be loaded fully.
            # 2. If the model can't fit fully into VRAM, then unload all other models and load as much of the model as
            #    possible.
            self._offload_unlocked_models(vram_needed_for_model)

            # Check the updated vram_available after offloading.
            vram_available = self._get_vram_available()

            # Move as much of the model as possible into VRAM.
            if isinstance(cache_entry.cached_model, CachedModelWithPartialLoad):
                cache_entry.cached_model.partial_load_to_vram(vram_available)
            elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
                # Partial load is not supported, so we have not choice but to try and fit it all into VRAM.
                cache_entry.cached_model.full_load_to_vram()
            else:
                raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")

            self._logger.debug(f"Locking {cache_entry.key} in {self._execution_device}")

            # TODO(ryand): Revive this.
            # self._print_cuda_stats()
        except torch.cuda.OutOfMemoryError:
            self._logger.warning("Insufficient GPU memory to load model. Aborting")
            cache_entry.unlock()
            raise
        except Exception:
            cache_entry.unlock()
            raise

        # try:
        #     if self._lazy_offloading:
        #         self._offload_unlocked_models(cache_entry.size)
        #     self._move_model_to_device(cache_entry, self._execution_device)
        #     cache_entry.loaded = True
        #     self._logger.debug(f"Locking {cache_entry.key} in {self._execution_device}")
        #     self._print_cuda_stats()
        # except torch.cuda.OutOfMemoryError:
        #     self._logger.warning("Insufficient GPU memory to load model. Aborting")
        #     cache_entry.unlock()
        #     raise
        # except Exception:
        #     cache_entry.unlock()
        #     raise

    def unlock(self, key: str) -> None:
        """Unlock a model."""
        cache_entry = self._cached_models[key]
        cache_entry.unlock()
        if not self._lazy_offloading:
            self._offload_unlocked_models(0)
            # self._print_cuda_stats()

    def _get_vram_available(self) -> int:
        """Get the amount of VRAM available in the cache."""
        # Calculate the total amount of VRAM currently in use.
        total_vram_in_use = sum(ce.cached_model.cur_vram_bytes() for ce in self._cached_models.values())
        # The amount of VRAM available in the cache.
        return int(self._max_vram_cache_size * GB) - total_vram_in_use

    def _get_ram_available(self) -> int:
        """Get the amount of RAM available in the cache."""
        total_ram_in_use = self._get_ram_in_use()
        return int(self._max_cache_size * GB) - total_ram_in_use

    def _get_ram_in_use(self) -> int:
        """Get the amount of RAM currently in use."""
        return sum(ce.cached_model.total_bytes() for ce in self._cached_models.values())

    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self._log_memory_usage:
            return MemorySnapshot.capture()
        return None

    def _make_cache_key(self, model_key: str, submodel_type: Optional[SubModelType] = None) -> str:
        if submodel_type:
            return f"{model_key}:{submodel_type.value}"
        else:
            return model_key

    def _offload_unlocked_models(self, vram_bytes_to_free: int) -> int:
        """Offload models from the execution_device until vram_bytes_to_free bytes are freed, or all models are
        offloaded. Of course, locked models are not offloaded.

        Returns:
            int: The number of bytes freed.
        """
        # TODO(ryand): Should we support both LRU and smallest-first offloading policies? I can imagine scenarios where
        # each would win.
        self._logger.debug(f"Offloading unlocked models to free {vram_bytes_to_free/GB:.2f}GB of VRAM.")
        vram_bytes_freed = 0
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
                self._logger.debug(
                    f"Partially unloaded {cache_entry.key} from VRAM to free {(cache_entry_bytes_freed/GB):.2f}GB."
                )
            elif isinstance(cache_entry.cached_model, CachedModelOnlyFullLoad):  # type: ignore
                cache_entry_bytes_freed = cache_entry.cached_model.full_unload_from_vram()
                self._logger.debug(
                    f"Unloaded {cache_entry.key} from VRAM to free {(cache_entry_bytes_freed/GB):.2f}GB."
                )
            else:
                raise ValueError(f"Unsupported cached model type: {type(cache_entry.cached_model)}")
            vram_bytes_freed += cache_entry_bytes_freed

        return vram_bytes_freed
        # reserved = self._max_vram_cache_size * GB
        # vram_in_use = torch.cuda.memory_allocated() + size_required
        # self._logger.debug(f"{(vram_in_use/GB):.2f}GB VRAM needed for models; max allowed={(reserved/GB):.2f}GB")
        # for _, cache_entry in sorted(self._cached_models.items(), key=lambda x: x[1].size):
        #     if vram_in_use <= reserved:
        #         break
        #     if not cache_entry.loaded:
        #         continue
        #     if not cache_entry.is_locked:
        #         self._move_model_to_device(cache_entry, self._storage_device)
        #         cache_entry.loaded = False
        #         vram_in_use = torch.cuda.memory_allocated() + size_required
        #         self._logger.debug(
        #             f"Removing {cache_entry.key} from VRAM to free {(cache_entry.size/GB):.2f}GB; vram free = {(torch.cuda.memory_allocated()/GB):.2f}GB"
        #         )

        TorchDevice.empty_cache()

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

    def _print_cuda_stats(self) -> None:
        """Log CUDA diagnostics."""
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GB)
        ram = "%4.2fG" % (self._get_ram_in_use() / GB)

        in_ram_models = 0
        in_vram_models = 0
        locked_in_vram_models = 0
        for cache_record in self._cached_models.values():
            if hasattr(cache_record.model, "device"):
                if cache_record.model.device == self._storage_device:
                    in_ram_models += 1
                else:
                    in_vram_models += 1
                if cache_record.is_locked:
                    locked_in_vram_models += 1

                self._logger.debug(
                    f"Current VRAM/RAM usage: {vram}/{ram}; models_in_ram/models_in_vram(locked) ="
                    f" {in_ram_models}/{in_vram_models}({locked_in_vram_models})"
                )

    def make_room(self, bytes_needed: int) -> None:
        """Make enough room in the cache to accommodate a new model of indicated size.

        Note: This function deletes all of the cache's internal references to a model in order to free it. If there are
        external references to the model, there's nothing that the cache can do about it, and those models will not be
        garbage-collected.
        """
        # TODO(ryand): Add debug logging.
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
                self._delete_cache_entry(cache_entry)
                del cache_entry
                models_cleared += 1
            else:
                pos += 1

        # if current_size + bytes_needed > maximum_size:
        #     self._logger.debug(
        #         f"Max cache size exceeded: {(current_size/GB):.2f}/{self.max_cache_size:.2f} GB, need an additional"
        #         f" {(bytes_needed/GB):.2f} GB"
        #     )

        # self._logger.debug(f"Before making_room: cached_models={len(self._cached_models)}")

        # pos = 0
        # models_cleared = 0
        # while current_size + bytes_needed > maximum_size and pos < len(self._cache_stack):
        #     model_key = self._cache_stack[pos]
        #     cache_entry = self._cached_models[model_key]
        #     device = cache_entry.model.device if hasattr(cache_entry.model, "device") else None
        #     self._logger.debug(
        #         f"Model: {model_key}, locks: {cache_entry._locks}, device: {device}, loaded: {cache_entry.loaded}"
        #     )

        #     if not cache_entry.is_locked:
        #         self._logger.debug(
        #             f"Removing {model_key} from RAM cache to free at least {(size/GB):.2f} GB (-{(cache_entry.size/GB):.2f} GB)"
        #         )
        #         current_size -= cache_entry.size
        #         models_cleared += 1
        #         self._delete_cache_entry(cache_entry)
        #         del cache_entry

        #     else:
        #         pos += 1

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
        self._logger.debug(f"After making room: cached_models={len(self._cached_models)}")

    def _delete_cache_entry(self, cache_entry: CacheRecord) -> None:
        self._cache_stack.remove(cache_entry.key)
        del self._cached_models[cache_entry.key]
