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


def test_drop_model_invalidates_shared_weights(mock_logger: MagicMock):
    """drop_model means "the next load must rebuild" (a load-affecting setting changed), so it
    must forget the shared canonical entirely — leaving it adoptable would turn the other
    device's "rebuild" into a silent reuse of the pre-change weights.
    """
    store = SharedCpuWeightsStore()
    cache_a = _make_cache(store, mock_logger)
    cache_b = _make_cache(store, mock_logger)
    try:
        cache_a.put("m", DummyModule())
        cache_b.put("m", DummyModule())
        assert store.refcount("m") == 2

        assert cache_a.drop_model("m") == 1
        # The canonical is gone for everyone, not just cache A.
        assert "m" not in store

        # Cache B's entry still works (its module aliases the retired tensors), and dropping it
        # releases cleanly even though the store has already forgotten the key.
        assert cache_b.get("m").cached_model is not None
        assert cache_b.drop_model("m") == 1
        assert "m" not in store
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


def test_settings_change_rebuild_does_not_adopt_stale_shared_weights(mock_logger: MagicMock):
    """The full cross-device staleness scenario: device A is mid-generation (entry locked) when a
    load-affecting setting changes and every cache gets drop_model(). A's locked entry is only
    marked stale, so it keeps aliasing the old canonical tensors — but device B's rebuild must
    NOT adopt them, and A's eventual stale-eviction must not corrupt B's new canonical.
    """
    store = SharedCpuWeightsStore()
    cache_a = _make_cache(store, mock_logger)
    cache_b = _make_cache(store, mock_logger)
    try:
        cache_a.put("m", DummyModule())
        cache_b.put("m", DummyModule())
        old_canonical = store.peek("m")
        assert old_canonical is not None

        # Device A is generating with the model: its entry is locked.
        entry_a = cache_a._cached_models["m"]
        entry_a.lock()

        # Load-affecting setting changes: the router drops the model from every per-device cache.
        assert cache_a.drop_model("m") == 0  # locked -> marked stale, not dropped
        assert entry_a.is_stale is True
        assert cache_b.drop_model("m") == 1

        # Even though A's stale entry still aliases the old canonical, the store must have
        # forgotten it so no future load can adopt it.
        assert store.peek("m") is None

        # Device B rebuilds under the new settings: a fresh canonical, not the old tensors.
        cache_b.put("m", DummyModule())
        new_canonical = cache_b.get("m").cached_model.get_cpu_state_dict()
        assert new_canonical is not old_canonical
        assert store.peek("m") is new_canonical
        assert store.refcount("m") == 1

        # A's generation finishes; unlock evicts the stale entry. Its release must be an
        # identity-mismatched no-op, NOT a decrement of B's new canonical.
        cache_a.unlock(entry_a)
        assert "m" not in cache_a._cached_models
        assert store.refcount("m") == 1
        assert store.peek("m") is new_canonical
    finally:
        cache_a.shutdown()
        cache_b.shutdown()
