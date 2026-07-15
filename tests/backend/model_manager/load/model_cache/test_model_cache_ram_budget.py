"""End-to-end tests of the global RamBudget driving eviction across per-device caches.

Validates that the budget counts a shared model once (not once-per-GPU), counts non-deduplicated
models per-instance, and that eviction is made against the global deduplicated total — including the
case where a cache cannot free RAM because another device still holds the model. Runs on CPU.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

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


def test_eviction_cannot_free_ram_held_by_another_device(mock_logger):
    """If a cache's only droppable model is still held by another device, eviction frees nothing
    globally (the shared weights stay live) and the new model is still admitted -> transiently over
    budget until the other device releases. The eviction loop must handle this without spinning."""
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=int(S * 1.4), shared_store=store)
    cache_a = _make_cache(store, budget, mock_logger)
    cache_b = _make_cache(store, budget, mock_logger)
    try:
        cache_a.put("shared", DummyModule())
        cache_b.put("shared", DummyModule())  # both devices hold "shared" (refcount 2, counted once)
        assert budget.total_in_use() == S

        cache_a.put("new", DummyModule())  # triggers make_room; "shared" is a's only droppable entry
        # a dropped its ref to "shared", but b still holds it, so the shared weights weren't freed.
        assert "shared" not in cache_a._cached_models
        assert "shared" in cache_b._cached_models
        assert store.refcount("shared") == 1
        assert "new" in cache_a._cached_models
        # "shared" (still alive via b) + "new" -> over the 1.4*S cap, as expected.
        assert budget.total_in_use() == 2 * S
    finally:
        cache_a.shutdown()
        cache_b.shutdown()
