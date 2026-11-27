"""`ModelCache.locks_held`: every pin on the cache's records, those it dropped included, whichever thread took it and
of whichever kind. The working-memory probe compares it with its own thread's locks to see another thread sharing the
device."""

import logging
import threading
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


@pytest.fixture
def cache():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=None,
    )
    yield cache
    cache.shutdown()


def test_counts_device_and_ram_locks_of_every_thread(cache: ModelCache):
    cache.put("vae", torch.nn.Linear(2, 2))
    cache.put("lora", torch.nn.Linear(2, 2))
    vae, lora = cache.get("vae"), cache.get("lora")

    cache.lock(vae, None)
    cache.lock(vae, None)
    other = threading.Thread(target=cache.lock_in_ram, args=(lora,))
    other.start()
    other.join()

    assert cache.locks_held() == 3

    cache.unlock(vae)
    cache.unlock(lora)
    assert cache.locks_held() == 1
    cache.unlock(vae)
    assert cache.locks_held() == 0


def test_counts_the_locks_on_a_record_the_cache_dropped_while_it_was_locked(cache: ModelCache):
    cache.put("vae", torch.nn.Linear(2, 2))
    dropped = cache.get("vae")
    cache.lock(dropped, None)
    # The VRAM-move error paths drop a record that is still locked; reaching them takes a GPU.
    cache._delete_cache_entry(dropped)
    cache.put("vae", torch.nn.Linear(2, 2))
    readmitted = cache.get("vae")
    cache.lock_in_ram(readmitted)

    assert cache.locks_held() == 2

    cache.unlock(dropped)
    assert cache.locks_held() == 1
    cache.unlock(readmitted)
    assert cache.locks_held() == 0


def test_an_unlock_without_a_lock_is_refused_and_leaves_the_count_right(cache: ModelCache):
    cache.put("vae", torch.nn.Linear(2, 2))
    vae = cache.get("vae")
    cache.lock(vae, None)
    cache.unlock(vae)

    with pytest.raises(RuntimeError, match="unlocked more often than it was locked"):
        cache.unlock(vae)

    assert cache.locks_held() == 0
    cache.lock(vae, None)
    assert (cache.locks_held(), vae.is_locked) == (1, True)
    cache.unlock(vae)
    assert (cache.locks_held(), vae.is_locked) == (0, False)


def _failing_vram_load(cache: ModelCache, monkeypatch: pytest.MonkeyPatch, settles_first: bool):
    """A record on a GPU whose VRAM move fails: on its first pass, or on the pass after a paced first one."""
    cache.put("vae", torch.nn.Linear(2, 2))
    vae = cache.get("vae")
    # Only the device the record names: the move it would start is the stub below.
    monkeypatch.setattr(vae.cached_model, "_compute_device", torch.device("cuda:0"))
    passes = iter([False] if settles_first else [])

    def load_locked_model(cache_entry, working_mem_bytes=None, max_move_bytes=None, stream_started_at=None) -> bool:
        for settled in passes:
            return settled
        raise torch.OutOfMemoryError("no room")

    monkeypatch.setattr(cache, "_load_locked_model", load_locked_model)
    monkeypatch.setattr(cache, "_log_cache_state", lambda *args, **kwargs: None)
    return vae


def test_a_lock_whose_vram_move_fails_holds_no_lock(cache: ModelCache, monkeypatch: pytest.MonkeyPatch):
    vae = _failing_vram_load(cache, monkeypatch, settles_first=False)

    with pytest.raises(torch.OutOfMemoryError):
        cache.lock(vae, None)

    assert (cache.locks_held(), vae.is_locked) == (0, False)


def test_a_paced_lock_whose_next_pass_fails_holds_no_lock(cache: ModelCache, monkeypatch: pytest.MonkeyPatch):
    vae = _failing_vram_load(cache, monkeypatch, settles_first=True)
    assert cache.lock(vae, None, max_move_bytes=1) is False
    assert cache.locks_held() == 1

    with pytest.raises(torch.OutOfMemoryError):
        cache.continue_lock(vae, None, max_move_bytes=1)

    assert (cache.locks_held(), vae.is_locked) == (0, False)
