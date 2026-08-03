import gc
import logging
import queue
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol

import psutil
import torch

from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import (
    SHARED_CPU_WEIGHTS,
    SharedCpuWeightsStore,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.prefix_logger_adapter import PrefixedLoggerAdapter

# Size of a GB in bytes.
GB = 2**30

# Size of a MB in bytes.
MB = 2**20

# Default RAM-cache sizing constants. These are used both by the per-device heuristic
# (_calc_ram_available_to_model_cache) and by the multi-GPU global budget cap
# (ModelManagerService.build_model_manager), so the two stay consistent.
#
# - RAM_CACHE_SYSTEM_FRACTION: fraction of total system RAM the model cache may use by default.
# - RAM_CACHE_BASELINE_BYTES:  assumed non-model RAM used by InvokeAI itself, reserved before sizing.
# - MIN_RAM_CACHE_BYTES:       absolute floor so the cache is never sized uselessly small.
RAM_CACHE_SYSTEM_FRACTION = 0.5
RAM_CACHE_BASELINE_BYTES = 2 * GB
MIN_RAM_CACHE_BYTES = 4 * GB

_DEFERRED_RECONCILE = object()
_DEFERRED_STOP = object()


def _run_deferred_work(cache_ref: "weakref.ReferenceType[ModelCache]", work_queue: "queue.SimpleQueue[object]") -> None:
    """Drain one ModelCache's deferred-work queue until it is stopped or the cache is collected.

    Deliberately a module-level function taking a *weak* reference rather than a bound method: a
    running thread is reachable from `threading._active` and keeps its target alive, so a bound
    method would make every ModelCache that has ever admitted a model immortal — pinning all of its
    cached models in RAM for the life of the process unless shutdown() happened to be called. The
    `weakref.finalize` registered alongside this thread (see ModelCache._ensure_deferred_worker)
    pushes _DEFERRED_STOP when the cache is collected, so a parked worker wakes and exits rather
    than leaking a thread per abandoned cache.
    """
    while True:
        work = work_queue.get()
        cache = None
        try:
            if work is _DEFERRED_STOP:
                return
            cache = cache_ref()
            if cache is None:
                # The cache was collected; nothing can ever need doing again.
                return
            if cache._shutdown_event.is_set():
                continue
            if work is _DEFERRED_RECONCILE:
                cache._reconcile_budget_if_pending()
            else:
                assert isinstance(work, CacheRecord)
                cache._release_first_use_grace(work)
        except Exception:
            if cache is not None:
                cache._logger.exception("Error processing deferred model-cache work")
        finally:
            # Drop both references before blocking on the next get(): locals stay bound for as long
            # as this frame lives. `work` may be a CacheRecord, which transitively holds its model's
            # CPU weights — and _release_first_use_grace's release hook can evict that very record,
            # removing it from the cache AND subtracting its bytes from the RamBudget, so holding it
            # would leave the budget under-reporting a model that is still resident. `cache` must go
            # for the same reason this function takes a weakref at all.
            work = None
            cache = None


class _ModelLoadReadWriteLock:
    """A write-preferring readers-writer lock that serializes model construction against VRAM moves.

    The model load machinery depends on PROCESS-GLOBAL monkey-patches that are not thread-safe:
    model CONSTRUCTION (diffusers `from_pretrained` / `accelerate.init_empty_weights`) temporarily
    replaces `torch.nn.Module.register_parameter` so that every newly-registered parameter is routed
    to the `meta` device. While that patch is installed, ANY `register_parameter` call in ANY thread
    is hijacked onto `meta`. VRAM load/unload uses `nn.Module.load_state_dict(assign=True)`, which
    assigns `Parameter`s via `__setattr__` -> `register_parameter` — so if it runs concurrently with
    a construction on another worker thread, its real weights get stranded on `meta`. That surfaces
    later as "Cannot copy out of meta tensor; no data!" or "unrecognized device meta".

    - Construction takes the WRITE lock (exclusive — no reader and no other writer may run).
    - VRAM load/unload takes the READ lock (shared, so concurrent moves on different GPUs still
      overlap each other; they only block while a construction holds the write lock).

    Write-preferring: once a construction is waiting, new readers queue behind it, so a steady stream
    of VRAM moves from busy workers can't starve a pending load.

    Lock-ordering contract: callers MUST acquire this lock *before* any `ModelCache._lock`, never
    after. Readers do so by taking the read lock around the outer `ModelCache.lock()` call (see
    `LoadedModelWithoutConfig`), and writers around the whole construction (see
    `ModelLoader._load_and_cache`). Acquiring it in the other order — cache lock first, then this
    lock — would risk an AB-BA deadlock with a writer that takes a cache lock during `put()`.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        with self._cond:
            # Defer to any active or waiting writer (write-preferring).
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()


# Process-global lock guarding the non-thread-safe model load machinery. See _ModelLoadReadWriteLock.
MODEL_LOAD_LOCK = _ModelLoadReadWriteLock()


# TODO(ryand): Where should this go? The ModelCache shouldn't be concerned with submodels.
def get_model_cache_key(model_key: str, submodel_type: Optional[SubModelType] = None) -> str:
    """Get the cache key for a model based on the optional submodel type."""
    if submodel_type:
        return f"{model_key}:{submodel_type.value}"
    else:
        return model_key


def synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that applies the class's self._lock to the method.

    After the lock is released, any pending peer-eviction request is honored (see
    ModelCache.request_budget_reconcile): a peer whose eviction request found this cache's
    lock contended relies on this hook — without it, nothing re-checks the shared RAM
    budget when the lock frees, and the budget could stay exceeded indefinitely.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            with self._lock:  # Automatically acquire and release the lock
                return method(self, *args, **kwargs)
        finally:
            # Only the outermost frame reconciles: the RLock may still be held by an
            # enclosing synchronized method on this thread, and eviction must not run in
            # the middle of its operation.
            if not self._lock._is_owned():
                try:
                    self._reconcile_budget_if_pending()
                except Exception:
                    # The reconcile is deferrable housekeeping and must not fail the operation
                    # that triggered it. It can raise from a sick CUDA context (empty_cache after
                    # an eviction) — and this hook runs inside the caller's frame AFTER the
                    # method body, so a raise out of lock_in_ram()/lock() here would propagate
                    # before LoadedModel's unlock-pairing try block is entered, leaking a
                    # permanently locked record (the same shape as a keep-alive Timer.start()
                    # failure, handled in _record_activity). The pending flag is only cleared
                    # once the budget is satisfied, so the next lock release retries.
                    self._logger.exception("Error reconciling the shared RAM budget after a lock release")

    return wrapper


def record_activity(method: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that records activity after a method completes successfully.

    Note: This decorator should be applied to methods that already hold self._lock.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._record_activity()
        return result

    return wrapper


@dataclass
class CacheEntrySnapshot:
    cache_key: str
    total_bytes: int
    current_vram_bytes: int


@dataclass(frozen=True)
class CacheClearResult:
    models_cleared: int
    bytes_freed: int


class CacheMissCallback(Protocol):
    def __call__(
        self,
        model_key: str,
        cache_snapshot: dict[str, CacheEntrySnapshot],
    ) -> None: ...


class CacheHitCallback(Protocol):
    def __call__(
        self,
        model_key: str,
        cache_snapshot: dict[str, CacheEntrySnapshot],
    ) -> None: ...


class CacheModelsClearedCallback(Protocol):
    def __call__(
        self,
        models_cleared: int,
        bytes_requested: int,
        bytes_freed: int,
        cache_snapshot: dict[str, CacheEntrySnapshot],
    ) -> None: ...


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
        keep_alive_minutes: float = 0,
        shared_cpu_weights: SharedCpuWeightsStore | None = SHARED_CPU_WEIGHTS,
        ram_budget: RamBudget | None = None,
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
        :param keep_alive_minutes: How long to keep models in cache after last use (in minutes). 0 means keep indefinitely.
        :param shared_cpu_weights: Process-global store that lets per-device caches share a single CPU copy of each
            model's weights (see SharedCpuWeightsStore). Defaults to the global store so that, in multi-GPU mode, a
            model loaded on multiple GPUs occupies RAM only once. Pass None to disable sharing for this cache.
        :param ram_budget: Optional shared RamBudget used as the single global RAM authority across all per-device
            caches. When provided, eviction decisions are made against the deduplicated, system-wide RAM total rather
            than this cache's local (double-counted) sum. When None, the cache uses its own local RAM accounting.
        """
        self._shared_cpu_weights = shared_cpu_weights
        self._ram_budget = ram_budget
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
        # Set by a peer whose eviction request found this cache's lock contended; honored
        # by the synchronized-decorator hook as soon as the current operation releases the
        # lock. See request_budget_reconcile / _reconcile_budget_if_pending.
        self._budget_reconcile_pending = threading.Event()

        self._on_cache_hit_callbacks: set[CacheHitCallback] = set()
        self._on_cache_miss_callbacks: set[CacheMissCallback] = set()
        self._on_cache_models_cleared_callbacks: set[CacheModelsClearedCallback] = set()

        # Keep-alive timeout support
        self._keep_alive_minutes = keep_alive_minutes
        self._last_activity_time: Optional[float] = None
        self._timeout_timer: Optional[threading.Timer] = None
        self._shutdown_event = threading.Event()
        self._deferred_work_queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._deferred_work_thread: Optional[threading.Thread] = None
        self._deferred_work_finalizer: Optional[weakref.finalize] = None

        if ram_budget is not None:
            ram_budget.register_cache(self)

    def on_cache_hit(self, cb: CacheHitCallback) -> Callable[[], None]:
        self._on_cache_hit_callbacks.add(cb)

        def unsubscribe() -> None:
            self._on_cache_hit_callbacks.discard(cb)

        return unsubscribe

    def on_cache_miss(self, cb: CacheMissCallback) -> Callable[[], None]:
        self._on_cache_miss_callbacks.add(cb)

        def unsubscribe() -> None:
            self._on_cache_miss_callbacks.discard(cb)

        return unsubscribe

    def on_cache_models_cleared(self, cb: CacheModelsClearedCallback) -> Callable[[], None]:
        self._on_cache_models_cleared_callbacks.add(cb)

        def unsubscribe() -> None:
            self._on_cache_models_cleared_callbacks.discard(cb)

        return unsubscribe

    @property
    def execution_device(self) -> torch.device:
        """Return the default execution device this cache loads models onto."""
        return self._execution_device

    @property
    def shared_cpu_weights(self) -> SharedCpuWeightsStore | None:
        """The process-global store this cache deduplicates CPU weights into, or None if disabled.

        Exposed so the loader can check (via `peek`) whether another device already holds a model's
        canonical CPU weights and adopt them at construction time instead of re-reading from disk.
        """
        return self._shared_cpu_weights

    def set_ram_budget(self, ram_budget: RamBudget) -> None:
        """Attach the shared global RamBudget after construction.

        Used by the model manager once all per-device caches exist and the global cap has been
        computed from their individual sizes (see ModelManagerService.build_model_manager).
        """
        self._ram_budget = ram_budget
        ram_budget.register_cache(self)

    @property
    def local_ram_cache_size_bytes(self) -> int:
        """The RAM cache size this cache computed for itself (from max_cache_ram_gb or the heuristic).

        Used by the model manager to seed the global RamBudget cap when no explicit limit is set.
        """
        return self._ram_cache_size_bytes

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
        # Populate the cache size in the stats object when it's set. Prefer the global budget cap
        # (the real system-wide limit) when one is attached.
        if self._stats is not None:
            self._stats.cache_size = (
                self._ram_budget.max_bytes if self._ram_budget is not None else self._ram_cache_size_bytes
            )

    def _record_activity(self) -> None:
        """Record model activity and reset the timeout timer if configured.

        Note: This method should only be called when self._lock is already held.
        """
        # A shut-down cache must not arm a new timer: shutdown() is idempotent (it returns early
        # once the event is set), so a timer armed by a post-shutdown operation would never be
        # cancelled by a later shutdown() and would outlive the cache it belongs to.
        if self._keep_alive_minutes <= 0 or self._shutdown_event.is_set():
            return

        self._last_activity_time = time.time()

        # Cancel any existing timer
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()

        # Start a new timer
        timeout_seconds = self._keep_alive_minutes * 60
        self._timeout_timer = threading.Timer(timeout_seconds, self._on_timeout)
        # Set as daemon so it doesn't prevent application shutdown
        self._timeout_timer.daemon = True
        try:
            self._timeout_timer.start()
        except RuntimeError:
            # Thread/pid exhaustion (RLIMIT_NPROC, a container's pids.max). The keep-alive timer is
            # an optimization; it must not fail the cache operation that triggered it. Every
            # @record_activity method runs this AFTER its real work — e.g. lock_in_ram() has already
            # incremented the record's lock count — so an exception here would propagate to callers
            # like LoadedModelWithoutConfig.model_in_ram() before their unlock-pairing try block is
            # entered, leaking a permanent pin. Swallow it: the next activity re-arms the timer.
            self._timeout_timer = None
            self._logger.warning(
                "Could not start the model-cache keep-alive timer; the keep-alive timeout is disabled until the "
                "next cache activity."
            )
            return
        self._logger.debug(f"Model cache activity recorded. Timeout set to {self._keep_alive_minutes} minutes.")

    @synchronized
    @record_activity
    def _on_timeout(self) -> None:
        """Called when the keep-alive timeout expires. Clears the model cache."""
        if self._shutdown_event.is_set():
            return

        # Double-check if there has been activity since the timer was set
        # This handles the race condition where activity occurred just before the timer fired
        if self._last_activity_time is not None and self._keep_alive_minutes > 0:
            elapsed_minutes = (time.time() - self._last_activity_time) / 60
            if elapsed_minutes < self._keep_alive_minutes:
                # Activity occurred, don't clear cache
                self._logger.debug(
                    f"Model cache timeout fired but activity detected {elapsed_minutes:.2f} minutes ago. "
                    f"Skipping cache clear."
                )
                return

        # Check if there are any unlocked models that can be cleared
        unlocked_models = [key for key, entry in self._cached_models.items() if not entry.is_locked]

        if len(unlocked_models) > 0:
            self._logger.info(
                f"Model cache keep-alive timeout of {self._keep_alive_minutes} minutes expired. "
                f"Clearing {len(unlocked_models)} unlocked model(s) from cache."
            )
            # Clear the cache by requesting a very large amount of space.
            # This is the same logic used by the "Clear Model Cache" button.
            # Using 1000 GB ensures all unlocked models are removed.
            self._make_room_internal(1000 * GB)
        elif len(self._cached_models) > 0:
            # All models are locked, don't log at info level
            self._logger.debug(
                f"Model cache timeout fired but all {len(self._cached_models)} model(s) are locked. "
                f"Skipping cache clear."
            )
        else:
            self._logger.debug("Model cache timeout fired but cache is already empty.")

    @synchronized
    def shutdown(self) -> None:
        """Shutdown the model cache, cancelling any pending timers."""
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        self._deferred_work_queue.put(_DEFERRED_STOP)
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    @synchronized
    @record_activity
    def put(
        self, key: str, model: AnyModel, execution_device: Optional[torch.device] = None, prefetch: bool = False
    ) -> None:
        """Add a model to the cache.

        Args:
            key: Cache key for the model
            model: The model to cache
            execution_device: Optional device to use for this specific model. If None, uses the cache's default
                execution_device. Use torch.device("cpu") to force a model to run on CPU.
            prefetch: The model is being cached opportunistically (e.g. the unused submodels of a
                single-file pipeline load) and no loader will retrieve it after this call. It is
                admitted without the post-admission grace, so budget reconciles may evict it
                immediately.
        """
        if key in self._cached_models:
            self._logger.debug(
                f"Attempted to add model {key} ({model.__class__.__name__}), but it already exists in the cache. No action necessary."
            )
            return

        # Start (or revive) the worker before mutating cache state. Every deferred task concerns an
        # admitted record, so this makes later dispatch queue-only while avoiding a thread for an
        # unused cache.
        self._ensure_deferred_worker()

        # Any entry still carrying the post-admission grace belongs to an earlier load: cold loads
        # are serialized under MODEL_LOAD_LOCK's write lock and each load's only graced put() is
        # its final one, so a flag that survives to the next admission is stale — its loader
        # either errored out before retrieving the model or dropped the LoadedModel without ever
        # locking it. Clear such flags so an orphaned record cannot dodge budget reconciles
        # indefinitely. (An entry retrieved but not yet locked loses its shield here; if a
        # reconcile then evicts it, lock() falls back to the tolerated issue-7513 path and
        # proceeds on the detached record.)
        for stale_entry in self._cached_models.values():
            stale_entry.awaiting_first_use = False

        size = calc_model_size_by_data(self._logger, model)
        self._make_room_internal(size)

        # Inject custom modules into the model.
        if isinstance(model, torch.nn.Module):
            apply_custom_layers_to_model(model)

        # Use the provided execution device, or fall back to the cache's default
        effective_execution_device = execution_device if execution_device is not None else self._execution_device

        # Partial loading only makes sense on CUDA.
        # - When running on CPU, there is no 'loading' to do.
        # - When running on MPS, memory is shared with the CPU, so the default OS memory management already handles this
        #   well.
        running_with_cuda = effective_execution_device.type == "cuda"

        # Wrap model.
        if isinstance(model, torch.nn.Module) and running_with_cuda and self._enable_partial_loading:
            wrapped_model = CachedModelWithPartialLoad(
                model,
                effective_execution_device,
                keep_ram_copy=self._keep_ram_copy_of_weights,
                shared_store=self._shared_cpu_weights,
                cache_key=key,
            )
        else:
            wrapped_model = CachedModelOnlyFullLoad(
                model,
                effective_execution_device,
                size,
                keep_ram_copy=self._keep_ram_copy_of_weights,
                shared_store=self._shared_cpu_weights,
                cache_key=key,
            )

        # awaiting_first_use protects the new entry from the asynchronous eviction paths (budget
        # reconcile, peer-requested eviction) until the loader locks it — see CacheRecord. A
        # prefetch admission gets no grace: nothing will come back for it, so it must stay
        # ordinarily evictable.
        #
        # Neither does an admission made while no deferred worker is running to release the grace
        # if the loader abandons the model. Both states are reachable — put() after shutdown()
        # (Invoker.stop() stops model_manager before session_processor, so an in-flight generation
        # can land here), and a worker that could not be started under thread exhaustion — and in
        # both the flag would never be cleared, leaving the record permanently invisible to every
        # asynchronous eviction path while its bytes stay charged to the shared budget. Without the
        # grace the record is merely ordinarily evictable; lock() still clears the flag on the
        # normal path, so nothing changes when the worker is healthy.
        worker_running = self._deferred_work_thread is not None and self._deferred_work_thread.is_alive()
        cache_record = CacheRecord(
            key=key, cached_model=wrapped_model, awaiting_first_use=not prefetch and worker_running
        )
        self._cached_models[key] = cache_record
        self._cache_stack.append(key)
        # Account this model's RAM in the global budget. Shared weights are tracked once by the
        # SharedCpuWeightsStore; only non-deduplicated models are added to the budget's non-shared
        # total (a non-shared model resident on N devices correctly counts N times).
        if self._ram_budget is not None and not wrapped_model.uses_shared_weights:
            self._ram_budget.add_non_shared(wrapped_model.total_bytes(), cache=self)
        self._logger.debug(
            f"Added model {key} (Type: {model.__class__.__name__}, Wrap mode: {wrapped_model.__class__.__name__}, Model size: {size / MB:.2f}MB)"
        )

        if self._ram_budget is not None and self._ram_budget.available() < 0:
            # Admission left the shared budget exceeded: make-room above exhausted this cache's
            # own evictable entries, and best-effort peer eviction fell short (a peer's lock was
            # contended, or the remaining RAM is held by locked in-use entries). Record a
            # reconcile request on every peer so each sheds unlocked entries as soon as it can —
            # the cap must not stay exceeded merely because a peer was busy at the moment we
            # asked. This runs after the new model is counted so the peers' budget checks see
            # the true (exceeded) state.
            for peer in self._ram_budget.peer_caches(exclude=self):
                peer.request_budget_reconcile()
            # If the overshoot is held by THIS cache's own locked entries, the peers may have
            # nothing to evict. That case is handled by unlock(): any unlock that leaves the
            # shared budget exceeded records a reconcile request on its own cache, so the entry
            # that eventually becomes evictable triggers the reconcile itself. (A request on
            # self here would be worse than useless: _make_room_internal already evicted
            # everything unlocked, so the only entry a self-reconcile could ever claim is the
            # model just admitted — evicting it out from under its own loader.)

        # Synced last, not right after the model is counted: request_budget_reconcile() above can
        # evict inline on a peer, and cache_used reads the shared budget's global usage, so syncing
        # before that loop would report a pre-reconcile (too high) figure.
        self._sync_current_stats()

    def cached_model_keys(self) -> set[str]:
        """Return the base model keys of every model currently resident in this cache.

        Used by the session queue's device-affinity heuristic to prefer pending items whose
        models are already warm on the claiming device. Two properties matter to that caller:

        - Entries keyed by `get_model_cache_key` (model key plus optional ``:submodel`` suffix)
          are reported with the suffix stripped so they match plain model keys. Entries keyed by
          filesystem path (`load_model_from_path`) are excluded: a path — or the bare Windows
          drive letter that splitting ``C:\\...`` on ``:`` would leave — is not a model key, and
          such short strings would corrupt substring-based affinity scoring.
        - Non-blocking: the cache lock is held across long operations (VRAM transfers, cache
          clears), and this feeds an opportunistic scheduling heuristic — returning an empty set
          when the lock is contended is better than stalling a worker's dequeue. For the same
          reason, a pending budget reconcile observed at release time is handed to a background
          thread rather than performed inline.
        """
        if not self._lock.acquire(blocking=False):
            return set()
        try:
            base_keys = (key.split(":", 1)[0] for key in self._cached_models)
            return {k for k in base_keys if len(k) >= 16 and "/" not in k and "\\" not in k}
        finally:
            self._lock.release()
            # This manual release must honor a pending budget reconcile just like the
            # synchronized-decorator hook: a peer whose request found the lock held by this
            # method relies on the release to process the flag (see request_budget_reconcile).
            # But reconciliation evicts models and runs gc.collect(), which can pause for
            # seconds — running it inline would break this method's no-stall contract. Hand the
            # work to the cache's background worker instead: it may wait on the cache lock and do
            # the slow work; this caller returns immediately.
            if self._ram_budget is not None and self._budget_reconcile_pending.is_set() and not self._lock._is_owned():
                self._dispatch_deferred(_DEFERRED_RECONCILE)

    def release_first_use_grace(self, cache_entry: CacheRecord) -> None:
        """Make an abandoned, never-locked record available for budget eviction.

        Called from a `weakref.finalize` callback (see LoadedModelWithoutConfig), which runs at an
        arbitrary decref/garbage-collection point in an arbitrary thread. That thread may already
        hold the SharedCpuWeightsStore or RamBudget lock — `SharedCpuWeightsStore.acquire()` sums
        tensor sizes inside its critical section, so a generational collection can fire there — and
        both of those are plain, non-reentrant locks.

        So this method must do NO locking work itself. Taking the cache lock here would run the
        synchronized-decorator release hook, whose `_reconcile_budget_if_pending` reads
        `RamBudget.available()` -> `SharedCpuWeightsStore.total_bytes_in_use()`, i.e. back through
        the very locks the collecting thread may be holding. That inverts the documented
        cache-lock -> (store-lock | budget-lock) order (see RamBudget) and deadlocks the thread
        against itself, wedging the whole process. The same hook would also run evictions,
        gc.collect() and empty_cache() inline in whatever unrelated thread dropped the reference.

        The work is therefore handed to the cache's background worker, exactly as
        cached_model_keys() does with its own reconcile. SimpleQueue.put() is reentrant and never
        waits on the cache, store or budget locks, so it is safe from a finalizer.
        """
        # Unsynchronized read: the flag is monotonic (put() is the only writer that sets it, and
        # only on a brand-new record), so a False reading is always final and there is nothing to
        # release. Losing a race here at worst queues work that no-ops under the lock.
        if not cache_entry.awaiting_first_use:
            return
        self._dispatch_deferred(cache_entry)

    def _ensure_deferred_worker(self) -> None:
        """Start the background worker if it is not currently running. Caller must hold the lock.

        A `threading.Thread` cannot be restarted, so a fresh one is created whenever the previous
        worker has exited. Without that, a worker lost to an unexpected error (e.g. a logging
        handler that raises, or `os.fork()`, neither of which clears `Thread.ident`) would silently
        disable every later grace release and budget reconcile for the life of the process — the
        record would keep shielding an idle cache from eviction, which is exactly the failure this
        mechanism exists to prevent.

        Never revives the worker after shutdown(): that call's `_DEFERRED_STOP` is still queued, so
        a new thread would consume it and exit immediately, and a shut-down cache has no deferred
        work worth doing.
        """
        if self._shutdown_event.is_set():
            return
        if self._deferred_work_thread is not None and self._deferred_work_thread.is_alive():
            return
        thread = threading.Thread(
            target=_run_deferred_work,
            args=(weakref.ref(self), self._deferred_work_queue),
            name="model-cache-deferred-work",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            # Thread/pid exhaustion (RLIMIT_NPROC, a container's pids.max). Deferred work is an
            # optimization, so it must not take the model load that this put() is completing down
            # with it. _dispatch_deferred drops work while no worker is running, and the next put()
            # both retries the start and clears stale grace flags itself, so no record can stay
            # shielded from eviction indefinitely.
            self._logger.warning(
                "Could not start the model-cache deferred-work thread; deferred cache work is disabled until the "
                "next model admission."
            )
            return
        if self._deferred_work_finalizer is None:
            # Wake the parked worker when this cache is collected so it exits instead of leaking.
            # This binds the queue and the sentinel only — never `self`, which would reintroduce the
            # strong reference that passing a weakref to the thread exists to avoid.
            self._deferred_work_finalizer = weakref.finalize(self, self._deferred_work_queue.put, _DEFERRED_STOP)
            self._deferred_work_finalizer.atexit = False
        self._deferred_work_thread = thread

    def _dispatch_deferred(self, work: object) -> None:
        """Hand work to the background worker, without blocking and without a lock.

        Neither caller may block: release_first_use_grace() runs inside a weakref finalizer (see its
        docstring) and cached_model_keys() has a no-stall contract. `SimpleQueue.put()` satisfies
        both, but only the worker ever drains the queue, so enqueueing while no worker is running
        would grow it without bound — and, for a CacheRecord, pin that model's CPU weights for the
        life of the process. Drop the item instead.

        Dropping loses nothing real, because put() only grants the first-use grace when a worker is
        running to release it (see put()). So a dropped CacheRecord is one that was never shielded
        from eviction in the first place, and a dropped reconcile is re-run by the synchronized
        release hook of the next cache operation — put() additionally clears any stale grace flags
        itself. What must never happen is a record that is shielded with nothing left to unshield
        it; that is what the pairing of these two rules prevents.

        A shutdown() landing between the check and the put() can still strand a single item in the
        queue. The cache is being torn down at that point and the cost is bounded by one record, so
        that race is tolerated rather than paid for with a lock this method cannot take.
        """
        if self._shutdown_event.is_set():
            return
        thread = self._deferred_work_thread
        if thread is None or not thread.is_alive():
            return
        self._deferred_work_queue.put(work)

    @synchronized
    def _release_first_use_grace(self, cache_entry: CacheRecord) -> None:
        """Clear an abandoned record's grace, then let the release hook reconcile the budget."""
        if self._cached_models.get(cache_entry.key) is cache_entry and not cache_entry.is_locked:
            cache_entry.awaiting_first_use = False

    @synchronized
    def _get_cache_snapshot(self) -> dict[str, CacheEntrySnapshot]:
        overview: dict[str, CacheEntrySnapshot] = {}
        for cache_key, cache_entry in self._cached_models.items():
            total_bytes = cache_entry.cached_model.total_bytes()
            current_vram_bytes = cache_entry.cached_model.cur_vram_bytes()
            overview[cache_key] = CacheEntrySnapshot(
                cache_key=cache_key,
                total_bytes=total_bytes,
                current_vram_bytes=current_vram_bytes,
            )

        return overview

    def _sync_current_stats(self) -> None:
        if self.stats:
            self.stats.cache_used = self._get_ram_in_use()
            self.stats.in_cache = len(self._cached_models)

    @synchronized
    @record_activity
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
            for cb in self._on_cache_miss_callbacks:
                cb(model_key=key, cache_snapshot=self._get_cache_snapshot())
            if self.stats:
                self.stats.misses += 1
            self._logger.debug(f"Cache miss: {key}")
            raise IndexError(f"The model with key {key} is not in the cache.")

        cache_entry = self._cached_models[key]
        # Deliberately NOT clearing awaiting_first_use here: this method is synchronized, so its
        # own lock-release hook may run a pending budget reconcile before returning — ending the
        # grace now would let that hook evict the very record this call just selected, detaching
        # a live model from the cache (and from the RAM accounting) before the caller can lock
        # it. The grace ends at lock(), when the entry becomes pinned anyway.

        # more stats
        if self.stats:
            stats_name = stats_name or key
            self.stats.high_watermark = max(self.stats.high_watermark, self._get_ram_in_use())
            self.stats.cache_used = self._get_ram_in_use()
            self.stats.in_cache = len(self._cached_models)
            self.stats.loaded_model_sizes[stats_name] = max(
                self.stats.loaded_model_sizes.get(stats_name, 0), cache_entry.cached_model.total_bytes()
            )

        # This moves the entry to the top (right end) of the stack.
        self._cache_stack = [k for k in self._cache_stack if k != key]
        self._cache_stack.append(key)

        self._logger.debug(f"Cache hit: {key} (Type: {cache_entry.cached_model.model.__class__.__name__})")
        for cb in self._on_cache_hit_callbacks:
            cb(model_key=key, cache_snapshot=self._get_cache_snapshot())

        return cache_entry

    @synchronized
    @record_activity
    def lock_in_ram(self, cache_entry: CacheRecord) -> None:
        """Pin a cache entry in RAM without moving it to its execution device."""
        if cache_entry.key not in self._cached_models:
            # Same diagnostic as lock()/unlock(): without it, a detached record pinned here would
            # produce only the unlock-side warning at the end of the generation, with no matching
            # lock-side message to correlate it with.
            self._logger.info(
                f"Pinning model cache entry {cache_entry.key} "
                f"(Type: {cache_entry.cached_model.model.__class__.__name__}), but it has already been dropped from "
                "the RAM cache. This is a sign that the model loading order is non-optimal in the invocation code "
                "(See https://github.com/invoke-ai/InvokeAI/issues/7513)."
            )
        cache_entry.lock()
        cache_entry.awaiting_first_use = False

    @synchronized
    @record_activity
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
        # End of the post-admission grace: from here the entry is pinned by its lock count, and
        # after the final unlock it is ordinary evictable cache content.
        cache_entry.awaiting_first_use = False

        self._logger.debug(
            f"Locking model {cache_entry.key} (Type: {cache_entry.cached_model.model.__class__.__name__})"
        )

        # A CPU compute_device means there's no VRAM load to do. This happens in two distinct situations:
        #   1. The model is explicitly configured cpu_only, but the cache's default execution device is a GPU.
        #   2. The whole install is CPU-only (no GPU), so every model's compute_device is CPU by default.
        # Only case 1 is noteworthy — surface it at INFO with the "(cpu_only)" cause so it mirrors the
        # "Loaded model ... onto <device> device" line emitted for GPU loads below. Case 2 would fire for every
        # lock of every model and says nothing about a per-model choice, so keep it at DEBUG and drop the wording.
        model_compute_device = cache_entry.cached_model.compute_device
        if model_compute_device.type == "cpu":
            if self._execution_device.type != "cpu":
                self._logger.info(
                    f"Loaded model '{cache_entry.key}' ({cache_entry.cached_model.model.__class__.__name__}) onto "
                    f"cpu device (cpu_only); skipping VRAM load"
                )
            else:
                self._logger.debug(
                    f"Loaded model '{cache_entry.key}' ({cache_entry.cached_model.model.__class__.__name__}) onto "
                    f"cpu device; skipping VRAM load"
                )
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
    @record_activity
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

        # If `drop_model()` marked this entry stale (e.g. settings changed while a generation
        # was using it), evict now so the next load rebuilds with the new settings rather than
        # silently reusing the pre-change cached module.
        if cache_entry.is_stale and not cache_entry.is_locked and cache_entry.key in self._cached_models:
            bytes_freed = cache_entry.cached_model.total_bytes()
            self._delete_cache_entry(cache_entry)
            if self.stats:
                self.stats.cleared = (self.stats.cleared or 0) + 1
                self._sync_current_stats()
            snapshot = self._get_cache_snapshot()
            for cb in self._on_cache_models_cleared_callbacks:
                cb(
                    models_cleared=1,
                    bytes_requested=0,
                    bytes_freed=bytes_freed,
                    cache_snapshot=snapshot,
                )
            gc.collect()
            TorchDevice.empty_cache()
            self._logger.debug(f"Evicted stale cache entry {cache_entry.key} after unlock.")

        # If the shared budget is exceeded, this unlock may have just made the offending entry
        # evictable. This is the only reconcile trigger for an overshoot held by this cache's
        # OWN locked entries: put() requests reconciles from its peers only (see put()), so
        # when the admitting cache itself holds the locked RAM, no pending request exists
        # anywhere until the lock releases. Recording it here lets this unlock's own release
        # hook perform the eviction.
        if self._ram_budget is not None and self._ram_budget.available() < 0:
            self._budget_reconcile_pending.set()

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
        # Use the model's actual compute_device for logging, not the cache's default
        model_device = cache_entry.cached_model.compute_device
        if model_device.type == "cuda":
            device_label = f"cuda device #{model_device.index}" if model_device.index is not None else "cuda device"
        else:
            device_label = f"{model_device.type} device"
        self._logger.info(
            f"Loaded model '{cache_entry.key}' ({cache_entry.cached_model.model.__class__.__name__}) onto "
            f"{device_label} in {(time.time() - start_time):.2f}s. "
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
            # Must be queried for THIS cache's execution device, not the process-current device. In
            # multi-GPU mode each worker calls torch.cuda.set_device for its own GPU, so the current
            # device flips between workers; querying without the device argument can read a different
            # (e.g. idle) GPU's allocation. That breaks the cancellation in _get_vram_available
            # (which adds vram_allocated(execution_device)), inflating "available" toward total VRAM
            # so the cache never offloads — causing VRAM OOMs that ignore device_working_mem_gb.
            return torch.cuda.memory_allocated(self._execution_device)
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
        # This runs at startup (one ModelCache is built per generation device before any request is
        # served), and torch.cuda.mem_get_info() would create a CUDA context that permanently holds
        # ~100-300 MiB of VRAM in an otherwise idle process. cudaGetDeviceProperties reports the same
        # total without creating a context (#9413).
        total_cuda_vram_bytes: int | None = None
        if self._execution_device.type == "cuda":
            total_cuda_vram_bytes = torch.cuda.get_device_properties(self._execution_device).total_memory

        # Apply heuristic 1.
        # ------------------
        heuristics_applied = [1]
        total_system_ram_bytes = psutil.virtual_memory().total
        # Assumed baseline RAM used by InvokeAI for non-model stuff.
        baseline_ram_used_by_invokeai = RAM_CACHE_BASELINE_BYTES
        ram_available_to_model_cache = int(
            total_system_ram_bytes * RAM_CACHE_SYSTEM_FRACTION - baseline_ram_used_by_invokeai
        )

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
        if ram_available_to_model_cache < MIN_RAM_CACHE_BYTES:
            heuristics_applied.append(3)
            ram_available_to_model_cache = MIN_RAM_CACHE_BYTES

        self._logger.info(
            f"Calculated model RAM cache size: {ram_available_to_model_cache / MB:.2f} MB. Heuristics applied: {heuristics_applied}."
        )
        return ram_available_to_model_cache

    @staticmethod
    def calc_system_ram_headroom_bytes() -> int:
        """The default system-wide cap on TOTAL model-cache RAM, leaving headroom for the OS.

        This is the maximum RAM the model caches should collectively use when the user has not set an
        explicit `max_cache_ram_gb`. It mirrors heuristic 1 of `_calc_ram_available_to_model_cache`
        (a fraction of system RAM, less InvokeAI's baseline) with the same minimum floor.

        In multi-GPU mode there is one cache per device, and each device's heuristic independently
        allows up to this fraction of system RAM; summed across N devices that would claim ~N× as
        much RAM and cause the system to swap. The model manager uses this value to cap that sum so a
        safe amount of RAM is always left for the OS and other processes.
        """
        total_system_ram_bytes = psutil.virtual_memory().total
        return max(
            int(total_system_ram_bytes * RAM_CACHE_SYSTEM_FRACTION) - RAM_CACHE_BASELINE_BYTES,
            MIN_RAM_CACHE_BYTES,
        )

    def _get_ram_in_use(self) -> int:
        """Get the amount of RAM currently in use.

        With a shared RamBudget attached, this returns the deduplicated, system-wide total across all
        per-device caches (shared model weights counted once). Without one, it returns this cache's
        local sum.
        """
        if self._ram_budget is not None:
            return self._ram_budget.total_in_use()
        return sum(ce.cached_model.total_bytes() for ce in self._cached_models.values())

    def _get_ram_available(self) -> int:
        """Get the amount of RAM available for the cache to use."""
        if self._ram_budget is not None:
            return self._ram_budget.available()
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
            # Query this cache's execution device (not the process-current one) for correct
            # per-device numbers in multi-GPU mode. See _get_vram_in_use.
            allocated = (
                torch.cuda.memory_allocated(self._execution_device) if self._execution_device.type == "cuda" else 0
            )
            log += "  {:<30} {:.1f} MB\n".format("CUDA Memory Allocated:", allocated / MB)
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
    def make_room(self, bytes_needed: int) -> CacheClearResult:
        """Make enough room in the cache to accommodate a new model of indicated size.

        Note: This function deletes all of the cache's internal references to a model in order to free it. If there are
        external references to the model, there's nothing that the cache can do about it, and those models will not be
        garbage-collected.
        """
        return self._make_room_internal(bytes_needed)

    def _make_room_internal(self, bytes_needed: int) -> CacheClearResult:
        """Internal implementation of make_room(). Assumes the lock is already held."""
        self._logger.debug(f"Making room for {bytes_needed / MB:.2f}MB of RAM.")
        self._log_cache_state(title="Before dropping models:")

        ram_bytes_available = self._get_ram_available()
        ram_bytes_to_free = max(0, bytes_needed - ram_bytes_available)

        ram_bytes_freed = 0
        pos = 0
        models_cleared = 0
        while pos < len(self._cache_stack):
            # Stop once there is enough room. With a shared RamBudget, re-check the global,
            # deduplicated availability each iteration: evicting a model that other devices still
            # hold frees no RAM (its shared weights stay live until the last reference is released),
            # so a fixed "bytes freed" tally would be wrong. Without a budget, the local tally is
            # exact, so the original cheaper check is kept.
            if self._ram_budget is not None:
                if bytes_needed <= self._get_ram_available():
                    break
            elif ram_bytes_freed >= ram_bytes_to_free:
                break

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

        if self._ram_budget is not None and bytes_needed > self._get_ram_available():
            # This cache's own evictable entries are exhausted, but the global budget is still
            # short: the remaining usage is held by other device caches — e.g. a shared model whose
            # weights stay live because another (possibly idle) cache retains them. Without
            # cross-cache eviction the cap would be exceeded for as long as that cache stays idle,
            # so ask each peer to drop its unlocked entries. Whatever still can't be freed is held
            # by locked (in-use) entries, which release soon — the same transient overshoot the
            # single-cache path has always allowed.
            for peer in self._ram_budget.peer_caches(exclude=self):
                if bytes_needed <= self._get_ram_available():
                    break
                peer_cleared = peer.evict_unlocked_for_peer(
                    is_satisfied=lambda: bytes_needed <= self._get_ram_available()
                )
                models_cleared += peer_cleared or 0
            # A peer whose lock was contended (peer_cleared is None) could not be evicted here.
            # That is handled AFTER admission: put() re-checks the budget once the new model is
            # actually counted and records a reconcile request on every peer (see
            # request_budget_reconcile). Requesting here, pre-admission, would race the peers'
            # reconcile checks against a budget that does not yet include the incoming model —
            # a peer could see the budget as satisfied, clear the request, and leave the cap
            # exceeded once the model lands.

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
            for cb in self._on_cache_models_cleared_callbacks:
                cb(
                    models_cleared=models_cleared,
                    bytes_requested=bytes_needed,
                    bytes_freed=ram_bytes_freed,
                    cache_snapshot=self._get_cache_snapshot(),
                )
            gc.collect()

        self._sync_current_stats()

        TorchDevice.empty_cache()
        self._logger.debug(f"Dropped {models_cleared} models to free {ram_bytes_freed / MB:.2f}MB of RAM.")
        self._log_cache_state(title="After dropping models:")
        return CacheClearResult(models_cleared=models_cleared, bytes_freed=ram_bytes_freed)

    def evict_unlocked_for_peer(self, is_satisfied: Callable[[], bool]) -> Optional[int]:
        """Evict this cache's unlocked entries on behalf of another device's cache (best effort).

        Called by a peer whose own eviction stack is exhausted while the shared RamBudget is still
        over-committed — typically because this cache holds the last reference to shared weights the
        peer already dropped. `is_satisfied` is re-checked after every eviction so no more entries
        are dropped than the peer actually needs.

        The peer calls this while holding its own cache lock, so this cache's lock is taken
        NON-blocking: if it is contended, this device is actively working and the peer simply skips
        it — blocking here could deadlock two caches making room for each other simultaneously.

        Returns the number of entries evicted, or None if the lock was contended and nothing could
        be attempted. Either way, if the shared budget is still exceeded once the caller admits its
        model, the caller records a deferred reconcile request on every peer (see
        request_budget_reconcile) — and every unlock that leaves the budget exceeded records one on
        its own cache — so a skip never leaves the budget exceeded indefinitely.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            models_cleared = 0
            pos = 0
            while pos < len(self._cache_stack) and not is_satisfied():
                cache_entry = self._cached_models[self._cache_stack[pos]]
                if cache_entry.is_locked or cache_entry.awaiting_first_use:
                    pos += 1
                    continue
                self._logger.debug(
                    f"Dropping {cache_entry.key} from RAM cache on behalf of a peer device cache "
                    f"({(cache_entry.cached_model.total_bytes() / MB):.2f}MB)."
                )
                self._delete_cache_entry(cache_entry)
                models_cleared += 1
            if models_cleared > 0:
                # Peer eviction mutates residency just like make_room/unlock/drop_model, so it owes
                # the same stat sync. Without it this cache reports a stale cache_used/in_cache
                # until its device next runs a session — and because the /models/stats aggregate
                # takes max(cache_used) across per-device caches, one stale idle cache pins the
                # reported usage high for the whole system.
                self._sync_current_stats()
                if self.stats:
                    self.stats.cleared = (self.stats.cleared or 0) + models_cleared
            return models_cleared
        finally:
            self._lock.release()
            # This manual release must honor a pending budget reconcile just like the
            # synchronized-decorator hook — a requester whose inline attempt found the lock
            # held by this method depends on it (see request_budget_reconcile). Non-blocking:
            # the peer calling this method holds its own cache lock, and blocking on this
            # cache's lock while holding another cache's lock is the one shape that could
            # deadlock; on contention the new holder's release hook processes the flag.
            if not self._lock._is_owned():
                self._reconcile_budget_if_pending(blocking=False)

    def request_budget_reconcile(self) -> None:
        """Ask this cache to shed unlocked entries until the shared budget is satisfied.

        Called by a peer that admitted a model while the global RamBudget was exceeded and could
        not free enough by evicting from this cache directly (evict_unlocked_for_peer found the
        lock contended, or everything evictable was already gone). Setting the pending flag alone
        is not enough: this cache's busy operation may have released its lock — running its
        reconcile hook while the flag was still unset — just before the flag was set here, and if
        the cache then stays idle no future lock release would ever honor the request (lost
        wakeup). The inline attempt below closes that window: either the lock is free now and the
        reconcile runs immediately, or it is still held and the eventual release hook — which runs
        strictly after the flag set below — performs it.
        """
        self._budget_reconcile_pending.set()
        # Non-blocking: the caller holds its own cache lock, and blocking here could deadlock two
        # caches requesting reconciles from each other.
        self._reconcile_budget_if_pending(blocking=False)

    def _reconcile_budget_if_pending(self, blocking: bool = True) -> None:
        """Evict this cache's unlocked entries while a requested reconcile is pending and the
        shared budget is exceeded.

        Called (blocking) from the synchronized-decorator hook after each lock release,
        (non-blocking) inline from request_budget_reconcile and from the manual lock release in
        evict_unlocked_for_peer, and (blocking, on a dedicated background thread) from
        cached_model_keys' release, which must not stall. The pending flag stays set until the
        budget is actually satisfied: if the remaining overshoot is held by locked (in-use)
        entries, the unlock that eventually frees them is itself a synchronized method, whose hook
        re-runs this reconcile.
        """
        if self._ram_budget is None or not self._budget_reconcile_pending.is_set():
            return
        while True:
            if self._ram_budget.available() >= 0:
                # Budget satisfied: retire the request. Clearing races a concurrent admission —
                # a peer counts its new model (driving the budget negative) BEFORE setting the
                # flag, so if the budget is negative again after the clear, the clear may have
                # wiped a request whose own inline attempt already saw the flag unset and
                # returned. Restore the flag and keep reconciling in that case.
                self._budget_reconcile_pending.clear()
                if self._ram_budget.available() >= 0:
                    return
                self._budget_reconcile_pending.set()
            if not self._lock.acquire(blocking=blocking):
                # Contended (non-blocking caller only): the holder releases through a reconcile
                # hook, which re-runs this reconcile with the flag still set.
                return
            models_cleared = 0
            try:
                pos = 0
                while pos < len(self._cache_stack) and self._ram_budget.available() < 0:
                    cache_entry = self._cached_models[self._cache_stack[pos]]
                    if cache_entry.is_locked or cache_entry.awaiting_first_use:
                        pos += 1
                        continue
                    self._logger.debug(
                        f"Dropping {cache_entry.key} from RAM cache to reconcile the shared RAM budget "
                        f"({(cache_entry.cached_model.total_bytes() / MB):.2f}MB)."
                    )
                    self._delete_cache_entry(cache_entry)
                    models_cleared += 1
                if models_cleared > 0:
                    # Same reasoning as evict_unlocked_for_peer: this path drops entries outside
                    # the synchronized methods that normally re-sync, so it must do so itself
                    # while the lock is still held.
                    self._sync_current_stats()
                    if self.stats:
                        self.stats.cleared = (self.stats.cleared or 0) + models_cleared
            finally:
                self._lock.release()
            if models_cleared > 0:
                gc.collect()
                TorchDevice.empty_cache()
            if self._ram_budget.available() < 0:
                # Everything evictable here is gone; the remainder is held by locked (in-use)
                # or just-admitted entries. Leave the flag set — the eventual unlock()'s hook
                # (or the loader's next cache operation) retries.
                return
            # Satisfied: loop back to retire the flag via the guarded clear above.

    def _delete_cache_entry(self, cache_entry: CacheRecord) -> None:
        """Delete cache_entry from the cache if it exists. No exception is thrown if it doesn't exist."""
        was_present = cache_entry.key in self._cached_models
        self._cache_stack = [key for key in self._cache_stack if key != cache_entry.key]
        self._cached_models.pop(cache_entry.key, None)
        # Drop this device's reference to the shared canonical CPU weights so they can be freed once
        # the last device releases them. Guard on was_present so a double-delete doesn't
        # double-release (release_shared_weights is itself idempotent, but a re-added entry under the
        # same key must not be released by a stale delete).
        if was_present:
            uses_shared = cache_entry.cached_model.uses_shared_weights
            total_bytes = cache_entry.cached_model.total_bytes()
            cache_entry.cached_model.release_shared_weights()
            # Drop the matching non-shared contribution from the global budget (shared weights are
            # released via the store above). Captured before release_shared_weights() flips the flag.
            if self._ram_budget is not None and not uses_shared:
                self._ram_budget.remove_non_shared(total_bytes, cache=self)

    @synchronized
    def drop_model(self, model_key: str) -> int:
        """Drop all cache entries belonging to a model so the next load rebuilds them.

        Cache keys are `<model_key>` or `<model_key>:<submodel>` (see `get_model_cache_key`),
        so a single model may have multiple entries. Locked entries are marked `is_stale` and
        evicted by `unlock()` as soon as the last lock releases — without that, a setting
        toggled during an in-flight generation would survive on the locked entry and quietly
        get reused by the next generation.

        Returns the number of entries immediately dropped (locked entries that are only marked
        stale do not count).
        """
        prefix = f"{model_key}:"
        matching: list[CacheRecord] = [
            entry for key, entry in self._cached_models.items() if key == model_key or key.startswith(prefix)
        ]

        dropped: list[CacheRecord] = []
        bytes_freed = 0
        for entry in matching:
            if entry.is_locked:
                entry.is_stale = True
                continue
            bytes_freed += entry.cached_model.total_bytes()
            self._delete_cache_entry(entry)
            dropped.append(entry)

        # Also forget this model's canonical shared CPU weights. A locked (stale-marked) entry keeps
        # its shared-store reference alive until unlock; without this, another device's rebuild of
        # the same key would acquire() that old canonical and silently adopt the pre-change weights.
        if self._shared_cpu_weights is not None:
            self._shared_cpu_weights.invalidate(model_key)

        if dropped:
            if self.stats:
                self.stats.cleared = len(dropped)
                self._sync_current_stats()
            snapshot = self._get_cache_snapshot()
            for cb in self._on_cache_models_cleared_callbacks:
                cb(
                    models_cleared=len(dropped),
                    bytes_requested=0,
                    bytes_freed=bytes_freed,
                    cache_snapshot=snapshot,
                )
            gc.collect()
            TorchDevice.empty_cache()
        return len(dropped)

    @synchronized
    def offload_model_from_vram(self, model_key: str) -> int:
        """Move a model (and its submodels) from VRAM to RAM without dropping it from the cache.

        Unlike `drop_model`, the cache entry is kept, so the model stays resident in RAM and the next load does
        not have to rebuild it from disk - only re-stream its weights back to VRAM. This is useful for freeing
        VRAM after a one-shot use (e.g. a text encoder that has already produced its embeddings) before a much
        larger model loads. Locked (in-use) entries are skipped.

        Returns the number of VRAM bytes freed.
        """
        prefix = f"{model_key}:"
        bytes_freed = 0
        for key, entry in list(self._cached_models.items()):
            if (key == model_key or key.startswith(prefix)) and not entry.is_locked:
                bytes_freed += self._move_model_to_ram(entry, entry.cached_model.total_bytes())
        if bytes_freed > 0:
            gc.collect()
            TorchDevice.empty_cache()
        return bytes_freed
