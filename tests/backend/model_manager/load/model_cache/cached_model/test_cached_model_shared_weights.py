"""Tests for sharing a single canonical CPU copy of model weights across per-device cached models.

These exercise the multi-GPU RAM-dedup path: two cached models built for the same cache key (as
would happen on two GPUs) must end up aliasing one set of CPU tensors instead of holding two
copies. They run on CPU — the wrapper constructors never touch VRAM, so no GPU is required.
"""

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule

CPU = torch.device("cpu")


def _data_ptrs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    return {k: v.data_ptr() for k, v in state_dict.items()}


def test_partial_load_shares_cpu_weights_across_devices():
    store = SharedCpuWeightsStore()
    # Two independently-initialised modules (distinct weights), as two devices would build.
    model_a = DummyModule()
    model_b = DummyModule()
    a_ptrs = _data_ptrs(model_a.state_dict())

    cached_a = CachedModelWithPartialLoad(model_a, CPU, keep_ram_copy=True, shared_store=store, cache_key="m")
    cached_b = CachedModelWithPartialLoad(model_b, CPU, keep_ram_copy=True, shared_store=store, cache_key="m")

    # Both cached models expose the SAME canonical CPU tensors.
    assert cached_a.get_cpu_state_dict() is cached_b.get_cpu_state_dict()
    assert _data_ptrs(cached_b.get_cpu_state_dict()) == a_ptrs

    # model_b's own parameters were re-pointed at the canonical tensors (b's originals are gone).
    assert _data_ptrs(model_b.state_dict()) == a_ptrs

    assert store.refcount("m") == 2
    # Counted once despite two devices holding it.
    assert store.total_bytes_in_use() == cached_a.total_bytes()


def test_full_load_shares_cpu_weights_across_devices():
    store = SharedCpuWeightsStore()
    model_a = DummyModule()
    model_b = DummyModule()
    a_ptrs = _data_ptrs(model_a.state_dict())

    cached_a = CachedModelOnlyFullLoad(
        model_a, CPU, total_bytes=100, keep_ram_copy=True, shared_store=store, cache_key="m"
    )
    cached_b = CachedModelOnlyFullLoad(
        model_b, CPU, total_bytes=100, keep_ram_copy=True, shared_store=store, cache_key="m"
    )

    assert cached_a.get_cpu_state_dict() is cached_b.get_cpu_state_dict()
    assert _data_ptrs(model_b.state_dict()) == a_ptrs
    assert store.refcount("m") == 2


def test_release_shared_weights_frees_at_last_reference():
    store = SharedCpuWeightsStore()
    cached_a = CachedModelWithPartialLoad(DummyModule(), CPU, keep_ram_copy=True, shared_store=store, cache_key="m")
    cached_b = CachedModelWithPartialLoad(DummyModule(), CPU, keep_ram_copy=True, shared_store=store, cache_key="m")
    assert store.refcount("m") == 2

    cached_a.release_shared_weights()
    assert store.refcount("m") == 1
    assert "m" in store

    cached_b.release_shared_weights()
    assert "m" not in store
    assert store.total_bytes_in_use() == 0


def test_release_shared_weights_is_idempotent():
    store = SharedCpuWeightsStore()
    cached = CachedModelWithPartialLoad(DummyModule(), CPU, keep_ram_copy=True, shared_store=store, cache_key="m")
    cached.release_shared_weights()
    cached.release_shared_weights()  # second call must not double-decrement
    assert store.refcount("m") == 0
    assert "m" not in store


def test_no_store_means_no_sharing_and_no_release_error():
    # Without a shared store, behaviour is unchanged: each model keeps its own CPU state dict.
    model = DummyModule()
    cached = CachedModelWithPartialLoad(model, CPU, keep_ram_copy=True)
    assert cached.get_cpu_state_dict() is not None
    # release is a safe no-op when nothing was shared.
    cached.release_shared_weights()


def test_keep_ram_copy_false_does_not_touch_store():
    store = SharedCpuWeightsStore()
    cached = CachedModelWithPartialLoad(DummyModule(), CPU, keep_ram_copy=False, shared_store=store, cache_key="m")
    assert cached.get_cpu_state_dict() is None
    assert "m" not in store
    assert store.refcount("m") == 0


class _RepointFailsModule(DummyModule):
    """A model whose load_state_dict raises, to simulate a re-point failure during construction."""

    def load_state_dict(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("simulated re-point failure")


def test_acquire_is_released_if_repoint_fails():
    # First device registers the canonical weights (refcount 1).
    store = SharedCpuWeightsStore()
    CachedModelWithPartialLoad(DummyModule(), CPU, keep_ram_copy=True, shared_store=store, cache_key="m")
    assert store.refcount("m") == 1

    # Second device adopts the canonical copy, but its re-point throws. The just-acquired reference
    # must be released so the store's refcount is not leaked (the wrapper never enters the cache).
    with pytest.raises(RuntimeError, match="simulated re-point failure"):
        CachedModelWithPartialLoad(_RepointFailsModule(), CPU, keep_ram_copy=True, shared_store=store, cache_key="m")

    assert store.refcount("m") == 1  # back to just the first device, not leaked at 2
