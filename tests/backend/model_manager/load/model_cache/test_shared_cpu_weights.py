import threading

import torch

from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "a": torch.ones(10, 10, dtype=torch.float32),  # 400 bytes
        "b": torch.ones(5, dtype=torch.float32),  # 20 bytes
    }


def test_first_acquire_registers_and_returns_same_object():
    store = SharedCpuWeightsStore()
    sd = _state_dict()
    canonical = store.acquire("k", sd)
    # The first acquire keeps the caller's own dict as canonical.
    assert canonical is sd
    assert store.refcount("k") == 1
    assert "k" in store


def test_second_acquire_returns_canonical_not_the_new_dict():
    store = SharedCpuWeightsStore()
    first = _state_dict()
    second = _state_dict()  # distinct tensors, same shapes
    canonical_first = store.acquire("k", first)
    canonical_second = store.acquire("k", second)

    # The second caller gets the originally-registered tensors, not its own.
    assert canonical_second is canonical_first
    assert canonical_second["a"].data_ptr() == first["a"].data_ptr()
    assert canonical_second["a"].data_ptr() != second["a"].data_ptr()
    assert store.refcount("k") == 2


def test_total_bytes_counts_each_key_once():
    store = SharedCpuWeightsStore()
    # Two devices acquire the same key -> counted once.
    store.acquire("k", _state_dict())
    store.acquire("k", _state_dict())
    assert store.total_bytes_in_use() == 420
    # A different key adds its own bytes.
    store.acquire("k2", {"x": torch.ones(100, dtype=torch.float32)})  # 400 bytes
    assert store.total_bytes_in_use() == 820


def test_release_frees_only_at_zero():
    store = SharedCpuWeightsStore()
    store.acquire("k", _state_dict())
    store.acquire("k", _state_dict())
    assert store.refcount("k") == 2

    store.release("k")
    assert store.refcount("k") == 1
    assert "k" in store
    assert store.total_bytes_in_use() == 420

    store.release("k")
    assert store.refcount("k") == 0
    assert "k" not in store
    assert store.total_bytes_in_use() == 0


def test_release_unknown_key_is_noop():
    store = SharedCpuWeightsStore()
    store.release("missing")  # must not raise
    assert store.total_bytes_in_use() == 0


def test_reacquire_after_full_release_registers_fresh():
    store = SharedCpuWeightsStore()
    first = _state_dict()
    store.acquire("k", first)
    store.release("k")
    assert "k" not in store

    second = _state_dict()
    canonical = store.acquire("k", second)
    # After a full release the next caller becomes the new canonical.
    assert canonical is second
    assert store.refcount("k") == 1


def test_concurrent_acquire_release_is_consistent():
    store = SharedCpuWeightsStore()
    sd = _state_dict()
    # Pre-register so the key exists for the whole run and the count never hits zero.
    store.acquire("k", sd)

    def worker():
        for _ in range(200):
            store.acquire("k", _state_dict())
            store.release("k")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every acquire was paired with a release, so only the pre-registration reference remains.
    assert store.refcount("k") == 1
    assert store.total_bytes_in_use() == 420


def test_invalidate_forgets_key_and_submodels():
    store = SharedCpuWeightsStore()
    store.acquire("k", _state_dict())
    store.acquire("k:unet", _state_dict())
    store.acquire("other", _state_dict())

    assert store.invalidate("k") == 2
    assert "k" not in store
    assert "k:unet" not in store
    assert store.peek("k") is None
    # Unrelated key untouched.
    assert store.refcount("other") == 1


def test_invalidated_entry_counts_against_budget_until_released():
    """Invalidation must not undercount RAM: a retired canonical still referenced by a locked
    cached model stays resident, so if a replacement is registered before the holder releases,
    BOTH copies must be counted — otherwise the budget admits models past `max_cache_ram_gb`."""
    store = SharedCpuWeightsStore()
    old = _state_dict()
    store.acquire("k", old)  # 420 bytes, held by a (conceptually locked) cached model
    assert store.total_bytes_in_use() == 420

    # A load-affecting setting change invalidates the key while the holder is still using it.
    assert store.invalidate("k") == 1
    assert "k" not in store
    # The retired tensors are still resident and must still be counted.
    assert store.total_bytes_in_use() == 420
    assert store.retired_bytes() == 420

    # Another device rebuilds the model under the same key: both copies are resident.
    new = _state_dict()
    store.acquire("k", new)
    assert store.total_bytes_in_use() == 840

    # The stale holder finally unlocks and releases its retired reference: only now does the
    # retired copy stop counting.
    store.release("k", old)
    assert store.retired_bytes() == 0
    assert store.total_bytes_in_use() == 420
    assert store.refcount("k") == 1


def test_invalidated_entry_with_multiple_holders_counts_until_last_release():
    store = SharedCpuWeightsStore()
    old = _state_dict()
    store.acquire("k", old)
    store.acquire("k", old)  # two devices hold the same canonical
    store.invalidate("k")

    store.release("k", old)
    # One holder remains: the retired copy is still resident.
    assert store.total_bytes_in_use() == 420

    store.release("k", old)
    assert store.total_bytes_in_use() == 0


def test_release_with_identity_mismatch_is_noop():
    """A holder of an invalidated canonical must not decrement a NEW canonical registered under
    the same key after the invalidation — that would free weights still aliased by live models."""
    store = SharedCpuWeightsStore()
    old = _state_dict()
    store.acquire("k", old)
    store.invalidate("k")

    new = _state_dict()
    store.acquire("k", new)
    assert store.refcount("k") == 1

    # The stale holder releases with its (retired) dict: no-op.
    store.release("k", old)
    assert store.refcount("k") == 1

    # The legitimate holder releases with the current dict: frees the entry.
    store.release("k", new)
    assert "k" not in store
