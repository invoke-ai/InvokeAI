"""End-to-end tests of the global RamBudget driving eviction across per-device caches.

Validates that the budget counts a shared model once (not once-per-GPU), counts non-deduplicated
models per-instance, and that eviction is made against the global deduplicated total — including the
case where a cache cannot free RAM because another device still holds the model. Runs on CPU.
"""

import logging
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
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
