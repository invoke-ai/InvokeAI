"""End-to-end tests of the global RamBudget driving eviction across per-device caches.

Validates that the budget counts a shared model once (not once-per-GPU), counts non-deduplicated
models per-instance, and that eviction is made against the global deduplicated total — including the
case where a cache cannot free RAM because another device still holds the model. Runs on CPU.
"""

import gc
import logging
import threading
import time
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

        # Dropping the never-locked wrapper is the only subsequent cache-A lifecycle event.
        del loaded_model
        gc.collect()

        assert "abandoned" not in cache_a._cached_models
        assert budget.total_in_use() == S
        assert budget.total_in_use() <= budget.max_bytes
    finally:
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
