"""End-to-end tests of the global RamBudget driving eviction across per-device caches.

Validates that the budget counts a shared model once (not once-per-GPU), counts non-deduplicated
models per-instance, and that eviction is made against the global deduplicated total — including the
case where a cache cannot free RAM because another device still holds the model. Runs on CPU.
"""

import gc
import logging
import threading
import time
import weakref
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats
from invokeai.backend.model_manager.load.model_cache.model_cache import (
    GB,
    MIN_RAM_CACHE_BYTES,
    RAM_CACHE_BASELINE_BYTES,
    RAM_CACHE_SYSTEM_FRACTION,
    ModelCache,
)
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule

# Persistent state-dict bytes of one DummyModule (what the shared store accounts for a shared model).
S = sum(calc_tensor_size(v) for v in DummyModule().state_dict().values())


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache(store, budget, logger, keep_ram_copy=True) -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=keep_ram_copy,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=store,
        ram_budget=budget,
    )


def _use_and_release(cache: ModelCache, key: str):
    """Mirror a production load-use cycle: retrieve, lock for inference (CPU execution, so no
    VRAM move), unlock. The lock() ends the post-admission grace, leaving the entry an evictable
    idle resident."""
    record = cache.get(key)
    cache.lock(record, None)
    cache.unlock(record)
    return record


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    """Poll until predicate() is true — budget reconciles may run on a background thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_shared_model_counts_once_in_global_budget(mock_logger):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("m", DummyModule())
        one_device = budget.total_in_use()
        assert one_device == S

        cache_b.put("m", DummyModule())
        # Second device shares the weights -> the global budget total does NOT grow.
        assert budget.total_in_use() == one_device
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_non_shared_model_counts_per_device(mock_logger):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    # keep_ram_copy=False -> not deduplicated, so each device's copy is real RAM.
    cache_a = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    cache_b = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    try:
        cache_a.put("m", DummyModule())
        one = budget.total_in_use()
        assert one > 0
        cache_b.put("m", DummyModule())
        # Two independent copies -> counted twice.
        assert budget.total_in_use() == 2 * one
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_model_in_ram_pins_warm_non_shared_model_against_peer_eviction(mock_logger):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    cache = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    try:
        cache.put("lora", DummyModule())
        record = _use_and_release(cache, "lora")
        loaded_model = LoadedModelWithoutConfig(cache_record=record, cache=cache)
        charged_bytes = budget.total_in_use()

        with loaded_model.model_in_ram():
            assert record.is_locked
            assert cache.evict_unlocked_for_peer(lambda: False) == 0
            assert "lora" in cache._cached_models
            assert budget.total_in_use() == charged_bytes

        assert not record.is_locked
        assert cache.evict_unlocked_for_peer(lambda: False) == 1
        assert budget.total_in_use() == 0
    finally:
        cache.shutdown()


def test_model_in_ram_on_a_cold_record_ends_the_grace_and_detaches_the_finalizer(mock_logger):
    """A pin on a never-locked record must clear awaiting_first_use and detach the first-use
    finalizer — otherwise the warm-path-only coverage leaves both lines of lock_in_ram dead code."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    cache = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    try:
        cache.put("lora", DummyModule())
        record = cache.get("lora")
        assert record.awaiting_first_use  # cold: still in the post-admission grace

        loaded_model = LoadedModelWithoutConfig(cache_record=record, cache=cache)
        assert loaded_model._first_use_finalizer is not None

        with loaded_model.model_in_ram():
            assert record.is_locked
            assert not record.awaiting_first_use
            # The grace has been consumed by the pin; the finalizer must be detached so a later GC
            # of the handle does not queue a redundant grace release.
            assert not loaded_model._first_use_finalizer.alive

        assert not record.is_locked
        # Post-pin, the record is ordinary evictable cache content.
        assert cache.evict_unlocked_for_peer(lambda: False) == 1
    finally:
        cache.shutdown()


def test_keep_alive_timer_start_failure_does_not_leak_a_pin(mock_logger, monkeypatch):
    """Timer.start() raising (thread/pid exhaustion) inside @record_activity must not propagate:
    lock_in_ram has already incremented the lock count, and the caller's unlock-pairing try block
    in model_in_ram() has not been entered yet — an exception here would pin the record forever."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    cache = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    cache._keep_alive_minutes = 5  # arm the timer path
    try:
        cache.put("lora", DummyModule())
        record = cache.get("lora")
        loaded_model = LoadedModelWithoutConfig(cache_record=record, cache=cache)

        def raising_start(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Timer, "start", raising_start)

        with loaded_model.model_in_ram():
            assert record.is_locked

        assert not record.is_locked  # the unlock paired with the lock; nothing leaked
        assert cache._timeout_timer is None
        assert any(
            call.args[0] == logging.WARNING and "keep-alive timer" in call.args[1]
            for call in mock_logger.log.call_args_list
        )
    finally:
        monkeypatch.undo()
        cache.shutdown()


def test_reconcile_hook_failure_does_not_leak_a_pin(mock_logger, monkeypatch):
    """The synchronized-decorator's post-release reconcile runs inside the caller's frame after the
    method body — a raise there (e.g. TorchDevice.empty_cache on a sick CUDA context) escapes
    lock_in_ram() after the lock count was incremented, before model_in_ram()'s unlock-pairing try
    block. It must be swallowed and logged; the pending flag persists so the next release retries."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    cache = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    try:
        cache.put("lora", DummyModule())
        record = cache.get("lora")
        loaded_model = LoadedModelWithoutConfig(cache_record=record, cache=cache)

        def raising_reconcile(blocking=True):
            raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(cache, "_reconcile_budget_if_pending", raising_reconcile)

        with loaded_model.model_in_ram():
            assert record.is_locked

        assert not record.is_locked  # the unlock paired with the lock; nothing leaked
        # PrefixedLoggerAdapter forwards exception() through Logger.log at ERROR level.
        assert any(
            call.args[0] == logging.ERROR and "reconciling" in call.args[1] for call in mock_logger.log.call_args_list
        )
    finally:
        monkeypatch.undo()
        cache.shutdown()


def test_global_budget_evicts_lru_in_single_cache(mock_logger):
    # Budget fits one model but not two -> putting the second evicts the first.
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache = _make_cache(store, budget, mock_logger)
    try:
        cache.put("a", DummyModule())
        cache.put("b", DummyModule())
        assert "a" not in cache._cached_models  # evicted to make room for b
        assert "b" in cache._cached_models
        assert "a" not in store and store.refcount("b") == 1
        assert budget.total_in_use() == S
    finally:
        cache.shutdown()


def test_get_vram_in_use_queries_this_caches_execution_device(mock_logger):
    """Regression: _get_vram_in_use must query its own execution device, not the process-current one.

    In multi-GPU mode each worker calls torch.cuda.set_device for its GPU, so a no-argument
    memory_allocated() can read a different device. That breaks the cancellation in
    _get_vram_available and inflates "available" VRAM, so the cache never offloads and OOMs while
    ignoring device_working_mem_gb.
    """
    import torch

    mc = "invokeai.backend.model_manager.load.model_cache.model_cache"
    with (
        patch(f"{mc}.torch.cuda.mem_get_info", return_value=(10 * GB, 48 * GB)),
        patch(f"{mc}.torch.cuda.get_device_properties", return_value=MagicMock(total_memory=48 * GB)),
        patch(f"{mc}.torch.cuda.memory_allocated", return_value=42) as mock_alloc,
    ):
        cache = ModelCache(
            execution_device_working_mem_gb=3.0,
            enable_partial_loading=True,
            keep_ram_copy_of_weights=True,
            execution_device="cuda:1",
            storage_device="cpu",
            logger=mock_logger,
        )
        try:
            assert cache._get_vram_in_use() == 42
            mock_alloc.assert_called_with(torch.device("cuda:1"))
        finally:
            cache.shutdown()


def test_cuda_cache_init_queries_total_vram_without_mem_get_info(mock_logger):
    """CUDA cache sizing must not call the VRAM-holding mem_get_info API during idle startup."""
    import torch

    mc = "invokeai.backend.model_manager.load.model_cache.model_cache"
    with (
        patch(f"{mc}.torch.cuda.get_device_properties", return_value=MagicMock(total_memory=48 * GB)) as mock_props,
        patch(f"{mc}.torch.cuda.mem_get_info", return_value=(10 * GB, 48 * GB)) as mock_mem_get_info,
    ):
        cache = ModelCache(
            execution_device_working_mem_gb=3.0,
            enable_partial_loading=True,
            keep_ram_copy_of_weights=True,
            execution_device="cuda:1",
            storage_device="cpu",
            logger=mock_logger,
        )
        try:
            mock_props.assert_called_once_with(torch.device("cuda:1"))
            mock_mem_get_info.assert_not_called()
        finally:
            cache.shutdown()


def _mock_total_ram(total_bytes: int):
    """Patch psutil.virtual_memory().total as seen by model_cache."""
    vm = MagicMock()
    vm.total = total_bytes
    return patch(
        "invokeai.backend.model_manager.load.model_cache.model_cache.psutil.virtual_memory",
        return_value=vm,
    )


def test_system_ram_headroom_is_fraction_minus_baseline():
    # On a 96 GB box, the default cap is 50% - 2 GB = 46 GB, leaving real headroom for the OS.
    with _mock_total_ram(96 * GB):
        headroom = ModelCache.calc_system_ram_headroom_bytes()
    assert headroom == int(96 * GB * RAM_CACHE_SYSTEM_FRACTION) - RAM_CACHE_BASELINE_BYTES
    assert headroom == 46 * GB
    # And it must leave at least half the system for everything else.
    assert headroom <= 96 * GB * 0.5


def test_system_ram_headroom_respects_floor_on_tiny_systems():
    # A machine with almost no RAM still gets the absolute minimum, never a negative/zero budget.
    with _mock_total_ram(2 * GB):
        headroom = ModelCache.calc_system_ram_headroom_bytes()
    assert headroom == MIN_RAM_CACHE_BYTES


def test_headroom_clamps_summed_multi_gpu_budget():
    # Reproduces the multi-GPU blowup: two 45 GB per-device caches sum to 90 GB, which would leave
    # only ~6 GB on a 96 GB machine. The headroom cap must clamp the budget below that sum.
    per_device_cache_bytes = 45 * GB
    summed = 2 * per_device_cache_bytes  # 90 GB, as the old code used verbatim
    with _mock_total_ram(96 * GB):
        headroom = ModelCache.calc_system_ram_headroom_bytes()
    clamped = min(summed, headroom)
    assert clamped == headroom < summed
    assert clamped == 46 * GB


def test_cache_stats_reflect_shared_global_budget(mock_logger):
    """Two distinct caches attached to the same RamBudget report system-wide stats: each cache's
    cache_size is the SAME single global limit, and each high_watermark observes the SAME global
    usage. An aggregator must therefore take the max of these fields — summing them would report
    an N-GPU system's capacity and high-water usage N times too large."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 10), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.stats = CacheStats()
        cache_b.stats = CacheStats()
        # Each per-device cache reports the shared global capacity, not a per-device slice.
        assert cache_a.stats.cache_size == budget.max_bytes
        assert cache_b.stats.cache_size == budget.max_bytes

        cache_a.put("m", DummyModule())
        cache_b.put("m", DummyModule())
        cache_a.get("m")
        cache_b.get("m")
        # Both watermarks sample the same deduplicated global usage (S, counted once), so the true
        # system high watermark is their max — not their sum.
        assert cache_a.stats.high_watermark == S
        assert cache_b.stats.high_watermark == S
        # cache_used is system-wide for the same reason, and is the field the aggregate reports as
        # *current* usage. Asserted here because it is the one the aggregate takes a max() over:
        # a cache that stops syncing it silently pins the reported usage for the whole system.
        assert cache_a.stats.cache_used == S
        assert cache_b.stats.cache_used == S
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_peer_eviction_syncs_the_evicted_caches_stats(mock_logger):
    """A cache evicted on behalf of a peer must re-sync its own stats.

    evict_unlocked_for_peer() is reached from another device's make_room, not from one of this
    cache's synchronized methods, so nothing else re-syncs it. If it does not sync, the evicted
    cache keeps advertising its pre-eviction cache_used/in_cache until its device next runs a
    session — and because /models/stats aggregates cache_used with max() across per-device caches,
    a single stale idle cache reports peak usage for the entire system, which reads as a cache
    that never releases memory."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 10), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.stats = CacheStats()
        cache_b.stats = CacheStats()
        cache_a.put("a1", DummyModule())
        cache_b.put("b1", DummyModule())
        _use_and_release(cache_a, "a1")
        _use_and_release(cache_b, "b1")
        # Two distinct models resident across the two devices: global usage is 2S, and b holds one.
        assert budget.total_in_use() == 2 * S
        assert cache_b.stats.in_cache == 1
        assert cache_b.stats.cache_used == 2 * S

        # The multi-GPU path: a's own evictable entries are exhausted, so it asks b to shed.
        # `is_satisfied` never returns True, so b drops every unlocked entry it has.
        assert cache_b.evict_unlocked_for_peer(is_satisfied=lambda: False) == 1
        assert "b1" not in cache_b._cached_models
        assert budget.total_in_use() == S

        assert cache_b.stats.in_cache == 0
        assert cache_b.stats.cache_used == S
        # Peer evictions are real evictions and must be counted; otherwise `cleared` only ever
        # reflects the requesting cache and under-reports system-wide.
        assert cache_b.stats.cleared == 1
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_budget_reconcile_syncs_the_reconciling_caches_stats(mock_logger):
    """The deferred-reconcile path drops entries outside any synchronized method too, so it owes
    the same stat sync as evict_unlocked_for_peer (same defect, second door)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 10), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_b.stats = CacheStats()
        cache_a.put("a1", DummyModule())
        cache_b.put("b1", DummyModule())
        _use_and_release(cache_a, "a1")
        _use_and_release(cache_b, "b1")
        assert cache_b.stats.in_cache == 1
        assert cache_b.stats.cache_used == 2 * S

        # Drive the budget negative so a pending reconcile actually evicts, then ask b to honor it
        # exactly as a peer's post-admission request would. The cap is lowered directly because
        # RamBudget exposes max_bytes read-only; reaching this state through admissions would only
        # add setup without exercising anything more.
        budget._max_bytes = int(S * 0.5)
        cache_b.request_budget_reconcile()

        assert _wait_until(lambda: "b1" not in cache_b._cached_models)
        assert cache_b.stats.in_cache == 0
        assert cache_b.stats.cache_used == budget.total_in_use()
        assert cache_b.stats.cleared == 1
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_eviction_coordinates_across_device_caches(mock_logger):
    """RAM held only by another device's cache must still be evictable: after this cache drops its
    own reference to a shared model, the peer cache's (unlocked, idle) reference is evicted too, so
    the global budget stays under max_bytes instead of exceeding it for as long as the peer stays
    idle."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())  # both devices hold "shared" (refcount 2, counted once)
        # Mirror production: each model is retrieved and locked for use, then released — the
        # lock() clears the post-admission grace, leaving the entries as evictable idle residents.
        _use_and_release(cache_a, "shared")
        _use_and_release(cache_b, "shared")
        assert budget.total_in_use() == S

        cache_a.put("new", DummyModule())  # triggers make_room; "shared" is a's only droppable entry
        # a dropped its ref to "shared"; that alone frees nothing (b still held it), so make_room
        # asks b — idle, entry unlocked — to drop its reference too, releasing the shared weights.
        assert "shared" not in cache_a._cached_models
        assert "shared" not in cache_b._cached_models
        assert store.refcount("shared") == 0
        assert "new" in cache_a._cached_models
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_contended_peer_reconciles_budget_after_its_operation_completes(mock_logger):
    """A peer whose cache lock is held at the moment of a cross-cache eviction request must not
    leave the budget exceeded indefinitely: the request is recorded on the busy peer, and the
    peer sheds its unlocked entries as soon as its current (lock-holding) operation completes —
    in production every operation releases the lock through the synchronized-decorator reconcile
    hook, which the holder thread emulates here. No other cache access may be needed to trigger
    the reconcile (JPPhoto merge blocker, 2026-07-22)."""
    import threading

    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())
        # Mirror production: each model is used and released — the lock() clears the
        # post-admission grace, leaving b's entry an evictable idle resident.
        _use_and_release(cache_a, "shared")
        _use_and_release(cache_b, "shared")
        assert budget.total_in_use() == S

        # Simulate b being mid-operation on another thread: its lock is contended when a's
        # make_room asks it to evict. (RLock is reentrant, so the hold must come from a
        # different thread than the one running a's put.)
        holding = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with cache_b._lock:
                holding.set()
                assert release.wait(timeout=10)
            # The synchronized-decorator hook that ends every real cache operation:
            cache_b._reconcile_budget_if_pending()

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert holding.wait(timeout=10)

        cache_a.put("new", DummyModule())
        # b was busy: its unlocked "shared" entry could not be dropped, so the budget is
        # transiently exceeded — but the reconcile request has been recorded on b.
        assert "shared" in cache_b._cached_models
        assert budget.total_in_use() == 2 * S

        # b's operation completes; its lock-release hook must reconcile without any
        # further access to cache_b.
        release.set()
        holder.join(timeout=10)

        assert "shared" not in cache_b._cached_models
        assert store.refcount("shared") == 0
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_reconcile_request_after_peer_finished_operation_is_not_lost(mock_logger):
    """Lost-wakeup regression: the peer's busy operation can release its lock and run its
    (no-op — flag not yet set) reconcile hook BEFORE the requester records the reconcile
    request. If the peer then stays idle, no future lock release exists to honor the request,
    so request_budget_reconcile must attempt the reconcile inline. The interleaving is forced
    by delaying the request until the peer's operation has fully finished, and the budget must
    return under its cap with no subsequent access to the peer cache (JPPhoto merge blocker,
    2026-07-25)."""
    import threading

    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())
        # Mirror production: each model is used and released — the lock() clears the
        # post-admission grace, leaving b's entry an evictable idle resident.
        _use_and_release(cache_a, "shared")
        _use_and_release(cache_b, "shared")
        assert budget.total_in_use() == S

        holding = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            # b's "busy operation": holds the lock (so evict_unlocked_for_peer fails), then
            # releases and runs the synchronized-decorator hook — with the request flag still
            # unset, a no-op. After this thread finishes, b is idle forever.
            with cache_b._lock:
                holding.set()
                assert release.wait(timeout=10)
            cache_b._reconcile_budget_if_pending()

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert holding.wait(timeout=10)

        # Delay the reconcile request until b's operation has completely finished (lock
        # released, hook run, thread dead) — the exact window in which a flag-only request
        # would be lost.
        original_request = cache_b.request_budget_reconcile

        def request_after_peer_finished() -> None:
            release.set()
            holder.join(timeout=10)
            assert not holder.is_alive()
            original_request()

        cache_b.request_budget_reconcile = request_after_peer_finished  # type: ignore[method-assign]

        cache_a.put("new", DummyModule())

        # No subsequent cache access: the request itself must have reconciled the budget.
        assert "shared" not in cache_b._cached_models
        assert store.refcount("shared") == 0
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_peer_eviction_skips_locked_entries(mock_logger):
    """A peer's LOCKED entry (a model in active use on that device) is never evicted from under it.
    The new model is still admitted -> transiently over budget until the peer's lock releases, same
    as the single-cache behavior when everything is locked."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())
        cache_b._cached_models["shared"].lock()

        cache_a.put("new", DummyModule())
        # b's entry was locked, so the shared weights could not be freed.
        assert "shared" not in cache_a._cached_models
        assert "shared" in cache_b._cached_models
        assert store.refcount("shared") == 1
        assert "new" in cache_a._cached_models
        assert budget.total_in_use() == 2 * S  # transiently over the 1.4*S cap
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_manual_release_in_cached_model_keys_honors_pending_reconcile(mock_logger):
    """cached_model_keys() manages the cache lock manually; its release must process a pending
    budget-reconcile request just like the synchronized-decorator hook does. Otherwise a request
    arriving while it holds the lock is stranded: the requester's inline attempt fails on the
    contended lock, the manual release runs no hook, and an idle cache leaves the budget exceeded
    indefinitely (JPPhoto review, 2026-07-27). The release hands the work to a background
    reconcile thread (it must not stall this method — see
    test_cached_model_keys_returns_without_waiting_for_reconcile_work), so the outcome is
    awaited with a deadline."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())
        _use_and_release(cache_a, "shared")
        _use_and_release(cache_b, "shared")

        inside = threading.Event()
        proceed = threading.Event()

        class PausingDict(dict):
            """Pauses cached_model_keys inside its lock, on its first key iteration."""

            paused = False

            def __iter__(self):
                if not PausingDict.paused:
                    PausingDict.paused = True
                    inside.set()
                    assert proceed.wait(timeout=10)
                return super().__iter__()

        cache_b._cached_models = PausingDict(cache_b._cached_models)

        holder = threading.Thread(target=cache_b.cached_model_keys)
        holder.start()
        assert inside.wait(timeout=10)

        # While b's lock is held inside cached_model_keys: a's admission overshoots the budget
        # and records a reconcile request on b (the request's inline attempt fails on the
        # contended lock).
        cache_a.put("new", DummyModule())
        assert budget.total_in_use() == 2 * S  # transiently exceeded
        assert "shared" in cache_b._cached_models

        proceed.set()
        holder.join(timeout=10)
        assert not holder.is_alive()

        # cached_model_keys' release handed the pending request to a background reconcile
        # thread — no other cache access needed, only time.
        assert _wait_until(lambda: "shared" not in cache_b._cached_models)
        assert store.refcount("shared") == 0
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_manual_release_in_evict_unlocked_for_peer_honors_pending_reconcile(mock_logger):
    """evict_unlocked_for_peer() also manages the cache lock manually; a reconcile request that
    arrives while it holds the lock must be processed when it releases, since the requester's
    inline attempt failed on the contended lock (JPPhoto review, 2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())
        _use_and_release(cache_a, "shared")
        _use_and_release(cache_b, "shared")

        inside = threading.Event()
        proceed = threading.Event()
        paused = [False]

        def satisfied_after_pause() -> bool:
            # The first check runs inside b's lock: rendezvous there, then report "satisfied"
            # so this peer-eviction pass evicts nothing itself.
            if not paused[0]:
                paused[0] = True
                inside.set()
                assert proceed.wait(timeout=10)
            return True

        holder = threading.Thread(target=lambda: cache_b.evict_unlocked_for_peer(is_satisfied=satisfied_after_pause))
        holder.start()
        assert inside.wait(timeout=10)

        cache_a.put("new", DummyModule())
        assert budget.total_in_use() == 2 * S  # transiently exceeded
        assert "shared" in cache_b._cached_models

        proceed.set()
        holder.join(timeout=10)
        assert not holder.is_alive()

        assert "shared" not in cache_b._cached_models
        assert store.refcount("shared") == 0
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_reconcile_clear_does_not_wipe_concurrent_request(mock_logger):
    """The satisfied-path clear() in _reconcile_budget_if_pending races a concurrent requester:
    a peer counts its admission (driving the budget negative) and sets the flag just before the
    clear lands, then its own inline attempt observes the just-wiped flag and returns. The
    reconciler must detect the wiped request — the budget is negative again after its clear —
    restore the flag, and keep reconciling (JPPhoto review, 2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_b.put("evictable", DummyModule())
        _use_and_release(cache_b, "evictable")
        assert budget.total_in_use() == S  # under the 1.4*S cap

        fired = [False]

        class RacingEvent(threading.Event):
            """On the first clear(): a peer admission lands and sets the flag — which this
            clear then wipes. That is exactly the interleaving under test."""

            def clear(self) -> None:
                if not fired[0]:
                    fired[0] = True
                    budget.add_non_shared(S)  # the peer's admission, counted pre-set...
                    self.set()  # ...and its request flag, set post-admission
                super().clear()

        cache_b._budget_reconcile_pending = RacingEvent()
        cache_b._budget_reconcile_pending.set()  # a previously recorded, now-satisfied request

        # A lock release runs the reconcile hook: it sees the budget satisfied and clears the
        # flag — colliding with the racing request. (The requester's inline attempt is emulated
        # by the no-op it becomes once it observes the wiped flag.)
        cache_b._reconcile_budget_if_pending()

        assert "evictable" not in cache_b._cached_models
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_b.shutdown()


def test_reconcile_post_eviction_clear_does_not_wipe_concurrent_request(mock_logger):
    """The same set-versus-clear race, entered through the eviction path: the reconciler
    satisfies the budget by evicting and then retires the flag, colliding with a concurrent
    admission's freshly set request (JPPhoto review, 2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 2.5), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_b.put("evictable_1", DummyModule())
        _use_and_release(cache_b, "evictable_1")
        cache_b.put("evictable_2", DummyModule())
        _use_and_release(cache_b, "evictable_2")
        assert budget.total_in_use() == 2 * S

        fired = [False]

        class RacingEvent(threading.Event):
            def clear(self) -> None:
                if not fired[0]:
                    fired[0] = True
                    budget.add_non_shared(S)
                    self.set()
                super().clear()

        cache_b._budget_reconcile_pending = RacingEvent()

        # A peer admission overshoots (2*S resident + S non-shared > 2.5*S cap) and requests a
        # reconcile. Evicting evictable_1 satisfies the budget; the post-eviction clear then
        # collides with a second concurrent admission's request (the RacingEvent).
        budget.add_non_shared(S)
        cache_b.request_budget_reconcile()

        assert "evictable_1" not in cache_b._cached_models
        assert "evictable_2" not in cache_b._cached_models
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_b.shutdown()


def test_admitting_cache_reconciles_its_own_overshoot_after_unlock(mock_logger):
    """When the overshoot is held by the ADMITTING cache's own locked entry and its peers have
    nothing to evict, the admitting cache itself must carry the pending request: the unlock that
    eventually frees the entry reconciles the budget. With peer-only requests, the unlock hook
    runs with the flag unset and the budget stays exceeded indefinitely (JPPhoto review,
    2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)  # stays empty: peers have nothing to evict
    try:
        cache_a.put("locked_model", DummyModule())
        record = cache_a.get("locked_model")
        cache_a.lock(record, None)  # in use on this device (CPU execution -> no VRAM move)

        cache_a.put("new", DummyModule())
        # The locked entry cannot be evicted and b holds nothing: transiently exceeded.
        assert budget.total_in_use() == 2 * S

        cache_a.unlock(record)
        # unlock()'s release hook processes this cache's own pending request — no peer action,
        # no further model load.
        assert "locked_model" not in cache_a._cached_models
        assert "new" in cache_a._cached_models
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_pending_reconcile_cannot_evict_just_admitted_model_before_first_use(mock_logger):
    """A reconcile request pending while put() runs (a peer's request that found this cache
    busy) is processed by put()'s own release hook — before the loader can call get(). The hook
    must never evict the model that was just admitted: the loader retrieves it via get()
    immediately after put() returns (load_default.py), and evicting it first would break the
    in-flight load with an IndexError. The post-admission grace (CacheRecord.awaiting_first_use)
    shields it until it is locked for use."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("locked_model", DummyModule())
        record = cache_a.get("locked_model")
        cache_a.lock(record, None)

        # A peer's reconcile request arrived while this cache was busy: the flag is pending
        # when put() runs.
        cache_a._budget_reconcile_pending.set()

        # Overshoots the budget with "new" as the only unlocked entry. put()'s release hook
        # processes the pending request before put() returns — it must skip the just-admitted
        # model.
        cache_a.put("new", DummyModule())
        assert "new" in cache_a._cached_models
        new_record = cache_a.get("new")  # the loader's immediate retrieval must succeed
        assert new_record.key == "new"
        # The in-use model was never touched.
        assert "locked_model" in cache_a._cached_models
    finally:
        cache_a.shutdown()


def test_pending_reconcile_cannot_evict_model_between_get_and_lock(mock_logger):
    """The admission grace must survive get(): get() is synchronized, so its own lock-release
    hook can run a pending reconcile before returning to the caller — if the grace ended inside
    get(), the hook would evict the very record the call just selected, detaching a live model
    from the cache and its RAM accounting and forcing a reload on the next request. The record
    must stay cached from put() through get() and the subsequent lock() (JPPhoto review,
    2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("locked_model", DummyModule())
        record = cache_a.get("locked_model")
        cache_a.lock(record, None)

        # A peer's reconcile request stays pending throughout the load (the locked model keeps
        # the budget exceeded, so no hook can retire it).
        cache_a._budget_reconcile_pending.set()

        cache_a.put("new", DummyModule())  # budget exceeded; "new" is the only unlocked entry
        new_record = cache_a.get("new")
        # get()'s own release hook ran the pending reconcile — it must not have evicted the
        # record it just returned.
        assert "new" in cache_a._cached_models
        cache_a.lock(new_record, None)
        assert "new" in cache_a._cached_models  # still cached once locked for inference
        cache_a.unlock(new_record)
        # After use the grace is gone: the still-pending request may now evict it normally.
        assert "new" not in cache_a._cached_models
    finally:
        cache_a.shutdown()


def test_prefetched_record_without_get_is_evictable_by_reconcile(mock_logger):
    """StableDiffusionLoader._load_from_singlefile() proactively put()s the pipeline submodels
    it was not asked for; nothing ever get()s or lock()s them. Such prefetch admissions must not
    receive the post-admission grace: with it, budget reconciles would skip the records
    indefinitely and an idle cache could keep the shared budget exceeded forever (JPPhoto
    review, 2026-07-27)."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        # A put()-only preload, exactly as the single-file pipeline loader admits the submodels
        # that were not requested.
        cache_b.put("prefetched_submodel", DummyModule(), prefetch=True)
        assert budget.total_in_use() == S

        # A peer's admission overshoots the budget and requests a reconcile here.
        budget.add_non_shared(S)
        cache_b.request_budget_reconcile()

        assert "prefetched_submodel" not in cache_b._cached_models
        assert store.refcount("prefetched_submodel") == 0
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_b.shutdown()


def test_next_admission_sweeps_stale_grace_from_prior_load(mock_logger):
    """A graced record whose loader never came back for it (the load errored between put() and
    get(), or the LoadedModel was dropped before lock()) must not keep its shield forever: cold
    loads are serialized, so the next admission on the same cache clears stale grace flags,
    making the orphan evictable by the still-pending reconcile."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 2.5), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_b.put("orphan", DummyModule())  # graced; its load died before get()
        cache_b.put("next", DummyModule())  # the next admission sweeps the stale flag
        _use_and_release(cache_b, "next")

        # A peer's overshoot: with the orphan's grace swept, the reconcile can evict it.
        budget.add_non_shared(int(S * 2))
        cache_b.request_budget_reconcile()

        assert "orphan" not in cache_b._cached_models
    finally:
        cache_b.shutdown()


def test_abandoned_loaded_model_releases_first_use_grace(mock_logger):
    """A LoadedModel dropped before its first lock must release the admission grace immediately.
    Otherwise an idle cache can shield the abandoned record from a peer's budget reconcile
    indefinitely."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("abandoned", DummyModule())
        loaded_model = LoadedModelWithoutConfig(cache_record=cache_a.get("abandoned"), cache=cache_a)

        # The live wrapper keeps the record protected while a peer admission exceeds the budget.
        cache_b.put("peer", DummyModule())
        assert "abandoned" in cache_a._cached_models
        assert budget.total_in_use() == 2 * S

        # Dropping the never-locked wrapper is the only subsequent cache-A lifecycle event. The
        # release is handed to a background thread (the finalizer must not do locking work on the
        # collecting thread — see ModelCache.release_first_use_grace), so poll for the outcome.
        del loaded_model
        gc.collect()

        assert _wait_until(lambda: "abandoned" not in cache_a._cached_models)
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_deferred_worker_start_failure_does_not_fail_the_admission(mock_logger, monkeypatch):
    """Thread exhaustion must degrade deferred work, not take the model load down with it.

    put() runs under MODEL_LOAD_LOCK's write lock while completing a load, so a RuntimeError from
    Thread.start() (RLIMIT_NPROC, a container's pids.max) escaping it would fail the generation.
    Deferred work is only an optimization, and the next admission retries the start.
    """
    store = SharedCpuWeightsStore()
    # Room for both admissions, so the second does not evict the first and the stale-flag sweep
    # below is actually observable.
    budget = RamBudget(max_bytes=int(S * 4), shared_store=store)
    cache = _make_cache(store, budget, mock_logger)
    real_thread_start = threading.Thread.start
    failed_once = False

    def fail_first_worker_start(thread: threading.Thread) -> None:
        nonlocal failed_once
        # Identify the worker by name: _ensure_deferred_worker only publishes the thread to
        # cache._deferred_work_thread once start() has succeeded, so identity is unusable here.
        if thread.name == "model-cache-deferred-work" and not failed_once:
            failed_once = True
            raise RuntimeError("forced thread start failure")
        real_thread_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_worker_start)
    try:
        cache.put("model", DummyModule())

        assert failed_once, "the worker start was never attempted"
        assert "model" in cache._cached_models
        assert budget.total_in_use() == S
        assert cache._deferred_work_thread is None
        # PrefixedLoggerAdapter forwards through Logger.log, so assert on the level, not .warning.
        assert any(call.args and call.args[0] == logging.WARNING for call in mock_logger.log.call_args_list)

        # With no worker, dispatch is dropped rather than queued into a void.
        cache._dispatch_deferred(model_cache_module._DEFERRED_RECONCILE)
        assert cache._deferred_work_queue.qsize() == 0

        # The next admission retries the start and clears the stale grace flag itself.
        cache.put("model2", DummyModule())
        assert cache._deferred_work_thread is not None and cache._deferred_work_thread.is_alive()
        assert not cache._cached_models["model"].awaiting_first_use
    finally:
        cache.shutdown()


def test_shutdown_stops_deferred_worker(mock_logger):
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache = _make_cache(store, budget, mock_logger)
    cache.put("model", DummyModule())

    cache.shutdown()
    cache._deferred_work_thread.join(timeout=10)

    assert not cache._deferred_work_thread.is_alive()


@pytest.mark.parametrize("finalizer_order", ["before_cache_release", "after_cache_release"])
def test_grace_release_from_finalizer_never_blocks_and_still_reconciles(mock_logger, finalizer_order):
    """The finalizer's one-shot release must neither block nor be lost.

    Exercise both relevant orders: the finalizer runs while another cache operation owns the
    lock, or after that operation has released the lock and checked for pending reconciliation.
    Neither order gets another cache operation to rescue a lost wakeup, so the deferred worker
    is the only thing that can carry the release through.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    holding = threading.Event()
    release = threading.Event()
    finalized = threading.Event()

    def hold_cache_lock() -> None:
        with cache_a._lock:
            holding.set()
            assert release.wait(timeout=10)
        cache_a._reconcile_budget_if_pending()

    def finalize_wrapper() -> None:
        nonlocal loaded_model
        loaded_model = None
        gc.collect()
        finalized.set()

    holder = threading.Thread(target=hold_cache_lock)
    finalizer = threading.Thread(target=finalize_wrapper)
    try:
        cache_a.put("abandoned", DummyModule())
        loaded_model = LoadedModelWithoutConfig(cache_record=cache_a.get("abandoned"), cache=cache_a)
        cache_b.put("peer", DummyModule())
        assert cache_a._budget_reconcile_pending.is_set()
        assert budget.total_in_use() == 2 * S

        holder.start()
        assert holding.wait(timeout=10)

        if finalizer_order == "before_cache_release":
            finalizer.start()
            assert finalized.wait(timeout=10), "grace finalizer deadlocked while cache lock was held"
            finalizer.join(timeout=10)
            release.set()
            holder.join(timeout=10)
        else:
            release.set()
            holder.join(timeout=10)
            finalizer.start()
            assert finalized.wait(timeout=10), "grace finalizer deadlocked after cache lock was released"
            finalizer.join(timeout=10)

        assert not holder.is_alive()
        assert not finalizer.is_alive()
        assert _wait_until(lambda: "abandoned" not in cache_a._cached_models, timeout=2)
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        release.set()
        holder.join(timeout=10)
        if finalizer.ident is not None:
            finalizer.join(timeout=10)
        cache_a.shutdown()
        cache_b.shutdown()


class _ReentryReportingLock:
    """Stand-in for a non-reentrant `threading.Lock` that reports same-thread re-acquisition
    instead of deadlocking on it, so a lock-order violation fails the test rather than hanging it.
    """

    def __init__(self, violations: list[str]) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._violations = violations

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._owner == threading.get_ident():
            self._violations.append(f"re-acquired on thread {threading.current_thread().name}")
            raise RuntimeError("non-reentrant lock re-acquired by its holding thread")
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._owner = threading.get_ident()
        return acquired

    def release(self) -> None:
        self._owner = None
        self._lock.release()

    def __enter__(self) -> "_ReentryReportingLock":
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def test_grace_release_does_not_take_cache_lock_on_the_collecting_thread(mock_logger):
    """The grace release runs from a weakref.finalize callback, i.e. at an arbitrary
    garbage-collection point in an arbitrary thread — including one that already holds the
    SharedCpuWeightsStore lock (`acquire()` sums tensor sizes inside its critical section, so a
    generational collection can fire there, and `put()` reaches it via the cached-model wrapper).

    If the callback takes the cache lock, the synchronized release hook reads
    RamBudget.available() -> SharedCpuWeightsStore.total_bytes_in_use(), which needs the store lock
    the collecting thread is already holding. Both store and budget locks are non-reentrant, so the
    thread deadlocks against itself and wedges the process (every other cache then blocks on the
    store lock at its next eviction). Guard: collecting an abandoned wrapper while holding the
    store lock must return promptly, and the release must still happen afterwards.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    # The whole point of this test is WHERE the collection happens, so automatic collection must
    # not pre-empt it: the setup below allocates freely (DummyModule alone is ~70 tracked objects),
    # and a gen-0 pass landing before the collector thread starts would reclaim the cycle on the
    # main thread with no store lock held, making the test pass vacuously — or trip the setup
    # assertions, depending on the ambient allocation count. Disable automatic gc for the setup and
    # let only the explicit collect below reclaim the cycle.
    gc.disable()
    try:
        cache_a.put("abandoned", DummyModule())
        loaded_model = LoadedModelWithoutConfig(cache_record=cache_a.get("abandoned"), cache=cache_a)

        # Trap the wrapper in a reference cycle so only the cyclic collector can reclaim it, which
        # is what an exception traceback does to an errored loader's local — precisely the
        # abandoned case the grace release exists for. This forces the callback to run inside
        # gc.collect() rather than at a refcount-zero decref.
        cycle: dict[str, object] = {"wrapper": loaded_model}
        cycle["self"] = cycle
        del loaded_model, cycle

        # A peer admission drives the shared budget over cap and leaves a reconcile pending on
        # cache A, so the release hook has real work to do.
        cache_b.put("peer", DummyModule())
        assert cache_a._budget_reconcile_pending.is_set()
        assert "abandoned" in cache_a._cached_models
        assert cache_a._cached_models["abandoned"].awaiting_first_use

        # The store's real lock is non-reentrant, so a violation would hang the process (and this
        # test's teardown) rather than fail it. Swap in a lock that reports same-thread
        # re-acquisition instead of blocking on it, so the defect surfaces as a failed assertion.
        violations: list[str] = []
        store._lock = _ReentryReportingLock(violations)

        collected = threading.Event()

        def collect_holding_store_lock() -> None:
            with store._lock:
                gc.collect()
            collected.set()

        collector = threading.Thread(target=collect_holding_store_lock, daemon=True)
        collector.start()
        assert collected.wait(timeout=20), (
            "the grace-release finalizer deadlocked the collecting thread against the shared-store lock"
        )
        collector.join(timeout=5)
        assert violations == [], (
            "the grace-release finalizer re-entered the shared-store lock on the collecting thread "
            f"(would deadlock on the real non-reentrant lock): {violations}"
        )

        # The release itself still has to happen, just off the collecting thread.
        assert _wait_until(lambda: "abandoned" not in cache_a._cached_models)
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        gc.enable()
        cache_a.shutdown()
        cache_b.shutdown()


def test_cached_model_keys_returns_without_waiting_for_reconcile_work(mock_logger, monkeypatch):
    """cached_model_keys() feeds the session queue's dequeue-affinity heuristic and must never
    stall. Its manual-release hook therefore hands a pending budget reconcile to a background
    thread instead of running it inline: reconciliation evicts models and calls gc.collect(),
    which can pause for seconds (JPPhoto review, 2026-07-27). gc.collect() is blocked here to
    prove the lookup does not wait for the reconcile work."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    release_gc = threading.Event()
    try:
        cache_b.put("victim", DummyModule())
        _use_and_release(cache_b, "victim")

        # A peer's admission overshoots the budget; its request found this cache busy, so only
        # the pending flag was left behind.
        budget.add_non_shared(S)
        cache_b._budget_reconcile_pending.set()

        real_collect = model_cache_module.gc.collect

        def blocking_collect(*args, **kwargs):
            assert release_gc.wait(timeout=10)
            return real_collect(*args, **kwargs)

        monkeypatch.setattr(model_cache_module.gc, "collect", blocking_collect)

        result = []
        lookup = threading.Thread(target=lambda: result.append(cache_b.cached_model_keys()))
        lookup.start()
        lookup.join(timeout=5)
        # The lookup returned even though the reconcile's gc.collect() is still blocked.
        assert not lookup.is_alive()
        assert result

        release_gc.set()
        # The background thread completes the reconcile on its own.
        assert _wait_until(lambda: "victim" not in cache_b._cached_models)
        assert _wait_until(lambda: not cache_b._budget_reconcile_pending.is_set())
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        release_gc.set()
        cache_b.shutdown()


@pytest.mark.parametrize("request_order", ["before_lookup", "while_lookup_holds_lock"])
def test_cached_model_keys_never_blocks_and_still_reconciles(mock_logger, request_order):
    """cached_model_keys() must preserve its nonblocking contract without leaving an idle cache
    permanently over budget: the reconcile it observes at release time is handed to the deferred
    worker rather than run inline or dropped.

    Exercise a request already pending at lookup entry and one racing with the lookup while its
    manually managed lock is held.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_b = _make_cache(store, budget, mock_logger)
    inside = threading.Event()
    proceed = threading.Event()
    results: list[set[str]] = []
    errors: list[BaseException] = []

    class PausingDict(dict):
        paused = False

        def __iter__(self):
            if not PausingDict.paused:
                PausingDict.paused = True
                inside.set()
                assert proceed.wait(timeout=10)
            return super().__iter__()

    def lookup() -> None:
        try:
            results.append(cache_b.cached_model_keys())
        except BaseException as error:
            errors.append(error)

    lookup_thread = threading.Thread(target=lookup)
    try:
        cache_b.put("victim", DummyModule())
        _use_and_release(cache_b, "victim")
        budget.add_non_shared(S)

        if request_order == "before_lookup":
            cache_b._budget_reconcile_pending.set()
            lookup_thread.start()
            lookup_thread.join(timeout=10)
            assert not lookup_thread.is_alive(), "cached_model_keys deadlocked with a pending reconcile"
        else:
            cache_b._cached_models = PausingDict(cache_b._cached_models)
            lookup_thread.start()
            assert inside.wait(timeout=10)
            cache_b.request_budget_reconcile()
            proceed.set()
            lookup_thread.join(timeout=10)
            assert not lookup_thread.is_alive()

        recovered = _wait_until(
            lambda: "victim" not in cache_b._cached_models and not cache_b._budget_reconcile_pending.is_set(),
            timeout=2,
        )
        assert (errors, results, recovered) == ([], [set()], True)
        assert budget.total_in_use() <= budget.max_bytes
    finally:
        proceed.set()
        if lookup_thread.ident is not None:
            lookup_thread.join(timeout=10)
        cache_b.shutdown()


def test_deferred_worker_does_not_pin_the_record_it_just_released(mock_logger):
    """The worker must drop its reference to a processed CacheRecord before blocking again.

    Releasing the grace lets the synchronized release hook evict the record: it leaves
    _cached_models AND its bytes are subtracted from the RamBudget. If the worker's frame still
    referenced it while parked in queue.get(), the budget would report RAM it had not actually
    freed, and the model would stay resident until some unrelated item was queued behind it.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("abandoned", DummyModule())
        record = cache_a.get("abandoned")
        module_ref = weakref.ref(record.cached_model.model)
        record_ref = weakref.ref(record)

        # Push the shared budget over its cap so the release hook actually evicts the record.
        cache_b.put("peer", DummyModule())
        assert cache_a._budget_reconcile_pending.is_set()

        cache_a.release_first_use_grace(record)
        del record

        # Poll the budget rather than reading it the instant the key disappears: _delete_cache_entry
        # pops from _cached_models before it releases the weights, so the two are briefly out of step.
        assert _wait_until(lambda: "abandoned" not in cache_a._cached_models)
        assert _wait_until(lambda: budget.total_in_use() == S)

        # The budget now says those bytes are gone; they must really be gone. No further queue
        # traffic is generated here — that is the point.
        assert _wait_until(lambda: (gc.collect(), record_ref() is None)[1]), "worker pinned the CacheRecord"
        assert module_ref() is None, "evicted model is still resident in RAM"
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_deferred_worker_is_revived_after_an_unexpected_death(mock_logger):
    """A worker lost to an unexpected error must not silently disable deferred work forever.

    Thread objects cannot be restarted and Thread.ident is never cleared, so a liveness check —
    not a "was it ever started" check — is what keeps a later put() able to recover.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("first", DummyModule())
        first_worker = cache_a._deferred_work_thread
        assert first_worker is not None and first_worker.is_alive()

        # Kill the worker the way a raising log handler would: the sentinel makes it return.
        cache_a._deferred_work_queue.put(model_cache_module._DEFERRED_STOP)
        first_worker.join(timeout=10)
        assert not first_worker.is_alive()
        assert first_worker.ident is not None  # so an `ident is None` guard would never recover

        # The dead-worker drop is covered by test_deferred_dispatch_is_dropped_when_no_worker_is_running;
        # keeping it out of this test lets this one fail on the revival defect itself.

        # The next admission revives it, and deferred work flows again.
        _use_and_release(cache_a, "first")
        cache_a.put("second", DummyModule())
        revived = cache_a._deferred_work_thread
        assert revived is not None and revived is not first_worker and revived.is_alive()

        record = cache_a.get("second")
        cache_b.put("peer", DummyModule())
        cache_a.release_first_use_grace(record)
        del record
        assert _wait_until(lambda: "second" not in cache_a._cached_models), "revived worker did no work"
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_deferred_dispatch_is_dropped_when_no_worker_is_running(mock_logger):
    """Only the worker drains the queue, so dispatching without one would grow it without bound.

    cached_model_keys() runs on every dequeue, so an idle-device cache that never admitted a model
    (and therefore has no worker and nothing to evict) would otherwise accumulate one queued
    reconcile per dequeue for the life of the process.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    idle_cache = _make_cache(store, budget, mock_logger)
    busy_cache = _make_cache(store, budget, mock_logger)
    try:
        # Peer RAM keeps the shared budget exceeded, so the request cannot retire itself.
        busy_cache.put("peer", DummyModule())
        _use_and_release(busy_cache, "peer")
        budget.add_non_shared(S)
        idle_cache._budget_reconcile_pending.set()
        assert budget.available() < 0
        assert idle_cache._deferred_work_thread is None

        for _ in range(200):
            assert idle_cache.cached_model_keys() == set()
        assert idle_cache._deferred_work_queue.qsize() == 0

        # After shutdown the same must hold for a cache that *does* have a (now stopped) worker.
        busy_cache.shutdown()
        assert busy_cache._deferred_work_thread is not None
        busy_cache._deferred_work_thread.join(timeout=10)
        assert not busy_cache._deferred_work_thread.is_alive()
        # The request must be pending, or cached_model_keys' release hook short-circuits before it
        # ever reaches the dispatch guard and the loop below proves nothing.
        busy_cache._budget_reconcile_pending.set()
        depth = busy_cache._deferred_work_queue.qsize()
        for _ in range(200):
            busy_cache.cached_model_keys()
        assert busy_cache._deferred_work_queue.qsize() == depth
    finally:
        idle_cache.shutdown()
        busy_cache.shutdown()


def test_post_shutdown_activity_does_not_arm_a_keep_alive_timer(mock_logger):
    """shutdown() is idempotent, so nothing after it may arm a timer a later shutdown won't cancel."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 4), shared_store=store)
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
        shared_cpu_weights=store,
        ram_budget=budget,
        keep_alive_minutes=60,
    )
    try:
        cache.put("model", DummyModule())
        assert cache._timeout_timer is not None

        cache.shutdown()
        assert cache._timeout_timer is None

        cache.put("after", DummyModule())
        assert cache._timeout_timer is None, "a shut-down cache re-armed its keep-alive timer"
    finally:
        if cache._timeout_timer is not None:
            cache._timeout_timer.cancel()


def test_dropped_cache_is_collectable_and_its_worker_exits(mock_logger):
    """A ModelCache released without shutdown() must not be kept alive by its own worker thread.

    A running thread is reachable from threading._active and keeps its target alive, so a worker
    holding a bound method would pin the cache — and every model in it — in RAM for the life of the
    process. The worker takes a weakref and a finalizer wakes it when the cache goes away.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 8), shared_store=store)

    def worker_count() -> int:
        return sum(1 for t in threading.enumerate() if t.name == "model-cache-deferred-work")

    before = worker_count()
    cache = _make_cache(store, budget, mock_logger)
    module = DummyModule()
    cache.put("model", module)
    assert worker_count() == before + 1

    cache_ref = weakref.ref(cache)
    module_ref = weakref.ref(module)
    del cache, module
    gc.collect()

    assert cache_ref() is None, "the worker thread kept the ModelCache alive"
    assert module_ref() is None, "the dropped cache's model is still resident in RAM"
    # The finalizer wakes the parked worker so it exits rather than leaking a thread per cache.
    assert _wait_until(lambda: worker_count() == before), "the worker thread outlived its cache"


def test_dropped_non_shared_cache_releases_only_its_budget_charge(mock_logger):
    """Collecting a cache must remove only its non-shared bytes from the surviving global budget."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 8), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    cache_b = _make_cache(store, budget, mock_logger, keep_ram_copy=False)
    cache_a.put("model-a", DummyModule())
    cache_bytes = budget.total_in_use()
    cache_b.put("model-b", DummyModule())
    budget.add_non_shared(S)
    assert budget.total_in_use() == cache_bytes * 2 + S

    cache_a_ref = weakref.ref(cache_a)
    del cache_a

    assert _wait_until(lambda: (gc.collect(), cache_a_ref() is None)[1]), "the cache was not collected"
    assert budget.total_in_use() == cache_bytes + S

    cache_b_ref = weakref.ref(cache_b)
    del cache_b

    assert _wait_until(lambda: (gc.collect(), cache_b_ref() is None)[1]), "the cache was not collected"
    assert budget.total_in_use() == S


def test_admission_without_a_worker_gets_no_first_use_grace(mock_logger):
    """A record must never be shielded from eviction with nothing left able to unshield it.

    put() after shutdown() is reachable in production — Invoker.stop() stops model_manager before
    session_processor, so an in-flight generation can admit a model after every cache has been shut
    down — and _ensure_deferred_worker deliberately does not revive the worker there. Granting the
    grace anyway would leave the record permanently invisible to both asynchronous eviction paths
    while its bytes stayed charged to the shared budget.
    """
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 8), shared_store=store)
    cache = _make_cache(store, budget, mock_logger)
    try:
        cache.put("normal", DummyModule())
        assert cache._cached_models["normal"].awaiting_first_use, "a healthy admission keeps its grace"

        cache.shutdown()
        assert cache._deferred_work_thread is not None
        cache._deferred_work_thread.join(timeout=10)

        cache.put("late", DummyModule())
        record = cache._cached_models["late"]
        assert cache._deferred_work_thread is not None and not cache._deferred_work_thread.is_alive()
        assert not record.awaiting_first_use, "admitted with a grace no worker can ever release"

        # Being unshielded, it is reachable by the synchronous eviction path.
        cache._delete_cache_entry(record)
        assert "late" not in cache._cached_models
    finally:
        cache.shutdown()
