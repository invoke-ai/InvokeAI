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
import hashlib
import math
import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, types

import torch

import invokeai.backend.util.logging as logger
from invokeai.backend.model_management.memory_snapshot import MemorySnapshot, get_pretty_snapshot_diff
from invokeai.backend.model_management.model_load_optimizations import skip_torch_weight_init

from ..util.devices import choose_torch_device
from .models import BaseModelType, ModelBase, ModelType, SubModelType

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


@dataclass
class CacheStats(object):
    hits: int = 0  # cache hits
    misses: int = 0  # cache misses
    high_watermark: int = 0  # amount of cache used
    in_cache: int = 0  # number of models in cache
    cleared: int = 0  # number of models cleared to make space
    cache_size: int = 0  # total size of cache
    # {submodel_key => size}
    loaded_model_sizes: Dict[str, int] = field(default_factory=dict)


class ModelLocker(object):
    "Forward declaration"

    pass


class ModelCache(object):
    "Forward declaration"

    pass


class _CacheRecord:
    size: int
    model: Any
    cache: ModelCache
    _locks: int

    def __init__(self, cache, model: Any, size: int):
        self.size = size
        self.model = model
        self.cache = cache
        self._locks = 0

    def lock(self):
        self._locks += 1

    def unlock(self):
        self._locks -= 1
        assert self._locks >= 0

    @property
    def locked(self):
        return self._locks > 0

    @property
    def loaded(self):
        if self.model is not None and hasattr(self.model, "device"):
            return self.model.device != self.cache.storage_device
        else:
            return False


class ModelCache(object):
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
        logger: types.ModuleType = logger,
        log_memory_usage: bool = False,
    ):
        """
        :param max_cache_size: Maximum size of the RAM cache [6.0 GB]
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param precision: Precision for loaded models [torch.float16]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param sha_chunksize: Chunksize to use when calculating sha256 model hash
        :param log_memory_usage: If True, a memory snapshot will be captured before and after every model cache
            operation, and the result will be logged (at debug level). There is a time cost to capturing the memory
            snapshots, so it is recommended to disable this feature unless you are actively inspecting the model cache's
            behaviour.
        """
        self.model_infos: Dict[str, ModelBase] = {}
        # allow lazy offloading only when vram cache enabled
        self.lazy_offloading = lazy_offloading and max_vram_cache_size > 0
        self.precision: torch.dtype = precision
        self.max_cache_size: float = max_cache_size
        self.max_vram_cache_size: float = max_vram_cache_size
        self.execution_device: torch.device = execution_device
        self.storage_device: torch.device = storage_device
        self.sha_chunksize = sha_chunksize
        self.logger = logger
        self._log_memory_usage = log_memory_usage

        # used for stats collection
        self.stats = None

        self._cached_models = {}
        self._cache_stack = []

    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self._log_memory_usage:
            return MemorySnapshot.capture()
        return None

    def get_key(
        self,
        model_path: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel_type: Optional[SubModelType] = None,
    ):
        key = f"{model_path}:{base_model}:{model_type}"
        if submodel_type:
            key += f":{submodel_type}"
        return key

    def _get_model_info(
        self,
        model_path: str,
        model_class: Type[ModelBase],
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        model_info_key = self.get_key(
            model_path=model_path,
            base_model=base_model,
            model_type=model_type,
            submodel_type=None,
        )

        if model_info_key not in self.model_infos:
            self.model_infos[model_info_key] = model_class(
                model_path,
                base_model,
                model_type,
            )

        return self.model_infos[model_info_key]

    # TODO: args
    def get_model(
        self,
        model_path: Union[str, Path],
        model_class: Type[ModelBase],
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        gpu_load: bool = True,
    ) -> Any:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not os.path.exists(model_path):
            raise Exception(f"Model not found: {model_path}")

        model_info = self._get_model_info(
            model_path=model_path,
            model_class=model_class,
            base_model=base_model,
            model_type=model_type,
        )
        key = self.get_key(
            model_path=model_path,
            base_model=base_model,
            model_type=model_type,
            submodel_type=submodel,
        )
        # TODO: lock for no copies on simultaneous calls?
        cache_entry = self._cached_models.get(key, None)
        if cache_entry is None:
            self.logger.info(
                f"Loading model {model_path}, type"
                f" {base_model.value}:{model_type.value}{':'+submodel.value if submodel else ''}"
            )
            if self.stats:
                self.stats.misses += 1

            self_reported_model_size_before_load = model_info.get_size(submodel)
            # Remove old models from the cache to make room for the new model.
            self._make_cache_room(self_reported_model_size_before_load)

            # Load the model from disk and capture a memory snapshot before/after.
            start_load_time = time.time()
            snapshot_before = self._capture_memory_snapshot()
            with skip_torch_weight_init():
                model = model_info.get_model(child_type=submodel, torch_dtype=self.precision)
            snapshot_after = self._capture_memory_snapshot()
            end_load_time = time.time()

            self_reported_model_size_after_load = model_info.get_size(submodel)

            self.logger.debug(
                f"Moved model '{key}' from disk to cpu in {(end_load_time-start_load_time):.2f}s.\n"
                f"Self-reported size before/after load: {(self_reported_model_size_before_load/GIG):.3f}GB /"
                f" {(self_reported_model_size_after_load/GIG):.3f}GB.\n"
                f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
            )

            if abs(self_reported_model_size_after_load - self_reported_model_size_before_load) > 10 * MB:
                self.logger.debug(
                    f"Model '{key}' mis-reported its size before load. Self-reported size before/after load:"
                    f" {(self_reported_model_size_before_load/GIG):.2f}GB /"
                    f" {(self_reported_model_size_after_load/GIG):.2f}GB."
                )

            cache_entry = _CacheRecord(self, model, self_reported_model_size_after_load)
            self._cached_models[key] = cache_entry
        else:
            if self.stats:
                self.stats.hits += 1

        if self.stats:
            self.stats.cache_size = self.max_cache_size * GIG
            self.stats.high_watermark = max(self.stats.high_watermark, self._cache_size())
            self.stats.in_cache = len(self._cached_models)
            self.stats.loaded_model_sizes[key] = max(
                self.stats.loaded_model_sizes.get(key, 0), model_info.get_size(submodel)
            )

        with suppress(Exception):
            self._cache_stack.remove(key)
        self._cache_stack.append(key)

        return self.ModelLocker(self, key, cache_entry.model, gpu_load, cache_entry.size)

    def _move_model_to_device(self, key: str, target_device: torch.device):
        cache_entry = self._cached_models[key]

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
            f"Moved model '{key}' from {source_device} to"
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
                    f"Moving model '{key}' from {source_device} to"
                    f" {target_device} caused an unexpected change in VRAM usage. The model's"
                    " estimated size may be incorrect. Estimated model size:"
                    f" {(cache_entry.size/GIG):.3f} GB.\n"
                    f"{get_pretty_snapshot_diff(snapshot_before, snapshot_after)}"
                )

    class ModelLocker(object):
        def __init__(self, cache, key, model, gpu_load, size_needed):
            """
            :param cache: The model_cache object
            :param key: The key of the model to lock in GPU
            :param model: The model to lock
            :param gpu_load: True if load into gpu
            :param size_needed: Size of the model to load
            """
            self.gpu_load = gpu_load
            self.cache = cache
            self.key = key
            self.model = model
            self.size_needed = size_needed
            self.cache_entry = self.cache._cached_models[self.key]

        def __enter__(self) -> Any:
            if not hasattr(self.model, "to"):
                return self.model

            # NOTE that the model has to have the to() method in order for this
            # code to move it into GPU!
            if self.gpu_load:
                self.cache_entry.lock()

                try:
                    if self.cache.lazy_offloading:
                        self.cache._offload_unlocked_models(self.size_needed)

                    self.cache._move_model_to_device(self.key, self.cache.execution_device)

                    self.cache.logger.debug(f"Locking {self.key} in {self.cache.execution_device}")
                    self.cache._print_cuda_stats()

                except Exception:
                    self.cache_entry.unlock()
                    raise

            # TODO: not fully understand
            # in the event that the caller wants the model in RAM, we
            # move it into CPU if it is in GPU and not locked
            elif self.cache_entry.loaded and not self.cache_entry.locked:
                self.cache._move_model_to_device(self.key, self.cache.storage_device)

            return self.model

        def __exit__(self, type, value, traceback):
            if not hasattr(self.model, "to"):
                return

            self.cache_entry.unlock()
            if not self.cache.lazy_offloading:
                self.cache._offload_unlocked_models()
                self.cache._print_cuda_stats()

    # TODO: should it be called untrack_model?
    def uncache_model(self, cache_id: str):
        with suppress(ValueError):
            self._cache_stack.remove(cache_id)
        self._cached_models.pop(cache_id, None)

    def model_hash(
        self,
        model_path: Union[str, Path],
    ) -> str:
        """
        Given the HF repo id or path to a model on disk, returns a unique
        hash. Works for legacy checkpoint files, HF models on disk, and HF repo IDs

        :param model_path: Path to model file/directory on disk.
        """
        return self._local_model_hash(model_path)

    def cache_size(self) -> float:
        """Return the current size of the cache, in GB."""
        return self._cache_size() / GIG

    def _has_cuda(self) -> bool:
        return self.execution_device.type == "cuda"

    def _print_cuda_stats(self):
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GIG)
        ram = "%4.2fG" % self.cache_size()

        cached_models = 0
        loaded_models = 0
        locked_models = 0
        for model_info in self._cached_models.values():
            cached_models += 1
            if model_info.loaded:
                loaded_models += 1
            if model_info.locked:
                locked_models += 1

        self.logger.debug(
            f"Current VRAM/RAM usage: {vram}/{ram}; cached_models/loaded_models/locked_models/ ="
            f" {cached_models}/{loaded_models}/{locked_models}"
        )

    def _cache_size(self) -> int:
        return sum([m.size for m in self._cached_models.values()])

    def _make_cache_room(self, model_size):
        # calculate how much memory this model will require
        # multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = model_size
        maximum_size = self.max_cache_size * GIG  # stored in GB, convert to bytes
        current_size = self._cache_size()

        if current_size + bytes_needed > maximum_size:
            self.logger.debug(
                f"Max cache size exceeded: {(current_size/GIG):.2f}/{self.max_cache_size:.2f} GB, need an additional"
                f" {(bytes_needed/GIG):.2f} GB"
            )

        self.logger.debug(f"Before unloading: cached_models={len(self._cached_models)}")

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
                    f"Unloading model {model_key} to free {(model_size/GIG):.2f} GB (-{(cache_entry.size/GIG):.2f} GB)"
                )
                current_size -= cache_entry.size
                models_cleared += 1
                if self.stats:
                    self.stats.cleared += 1
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
            gc.collect()

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

        self.logger.debug(f"After unloading: cached_models={len(self._cached_models)}")

    def _offload_unlocked_models(self, size_needed: int = 0):
        reserved = self.max_vram_cache_size * GIG
        vram_in_use = torch.cuda.memory_allocated()
        self.logger.debug(f"{(vram_in_use/GIG):.2f}GB VRAM used for models; max allowed={(reserved/GIG):.2f}GB")
        for model_key, cache_entry in sorted(self._cached_models.items(), key=lambda x: x[1].size):
            if vram_in_use <= reserved:
                break
            if not cache_entry.locked and cache_entry.loaded:
                self._move_model_to_device(model_key, self.storage_device)

                vram_in_use = torch.cuda.memory_allocated()
                self.logger.debug(f"{(vram_in_use/GIG):.2f}GB VRAM used for models; max allowed={(reserved/GIG):.2f}GB")

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

    def _local_model_hash(self, model_path: Union[str, Path]) -> str:
        sha = hashlib.sha256()
        path = Path(model_path)

        hashpath = path / "checksum.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash

        self.logger.debug(f"computing hash of model {path.name}")
        for file in list(path.rglob("*.ckpt")) + list(path.rglob("*.safetensors")) + list(path.rglob("*.pth")):
            with open(file, "rb") as f:
                while chunk := f.read(self.sha_chunksize):
                    sha.update(chunk)
        hash = sha.hexdigest()
        with open(hashpath, "w") as f:
            f.write(hash)
        return hash
