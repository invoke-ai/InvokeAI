"""End-to-end test of CPU-weight sharing through ModelCache.put()/eviction.

Simulates the multi-GPU topology — one ModelCache per device, all sharing a single
SharedCpuWeightsStore — and asserts that the same model loaded into both caches keeps exactly one
CPU copy, with RAM freed only when the last device evicts it. Runs on CPU (no VRAM moves).
"""

import logging
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache(store: SharedCpuWeightsStore, logger: MagicMock) -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=store,
    )


def test_two_device_caches_share_one_cpu_copy(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    cache_a = _make_cache(store, mock_logger)
    cache_b = _make_cache(store, mock_logger)
    try:
        cache_a.put("m", DummyModule())
        ram_one_device = store.total_bytes_in_use()
        assert ram_one_device > 0

        cache_b.put("m", DummyModule())

        # One canonical CPU copy shared by both "devices": the second device's put adds NO RAM.
        assert store.refcount("m") == 2
        assert store.total_bytes_in_use() == ram_one_device
        sd_a = cache_a.get("m").cached_model.get_cpu_state_dict()
        sd_b = cache_b.get("m").cached_model.get_cpu_state_dict()
        assert sd_a is sd_b

        # Evicting from one device drops only its reference; the weights stay for the other.
        cache_a.make_room(10**12)
        assert "m" not in cache_a._cached_models
        assert store.refcount("m") == 1
        assert "m" in store

        # Evicting from the last device frees the shared RAM.
        cache_b.make_room(10**12)
        assert store.refcount("m") == 0
        assert "m" not in store
        assert store.total_bytes_in_use() == 0
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_drop_model_releases_shared_weights(mock_logger: MagicMock):
    store = SharedCpuWeightsStore()
    cache_a = _make_cache(store, mock_logger)
    cache_b = _make_cache(store, mock_logger)
    try:
        cache_a.put("m", DummyModule())
        cache_b.put("m", DummyModule())
        assert store.refcount("m") == 2

        assert cache_a.drop_model("m") == 1
        assert store.refcount("m") == 1
        assert cache_b.drop_model("m") == 1
        assert "m" not in store
    finally:
        cache_a.shutdown()
        cache_b.shutdown()
