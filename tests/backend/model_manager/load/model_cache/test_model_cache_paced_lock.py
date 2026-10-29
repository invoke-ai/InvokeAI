"""Tests for paced (chunked) VRAM moves.

`ModelCache.lock(..., max_move_bytes=...)` / `continue_lock()` split one long RAM->VRAM stream
into bounded passes, and `LoadedModelWithoutConfig._lock_paced()` drops MODEL_LOAD_LOCK's read
lock between passes so a queued model construction (write lock) on any worker runs in the gap
instead of waiting out the entire stream.
"""

import logging
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK, ModelCache
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")


def _mock_logger() -> MagicMock:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache() -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cuda:0",
        storage_device="cpu",
        logger=_mock_logger(),
        shared_cpu_weights=None,
    )


@requires_cuda
def test_paced_lock_converges_and_pins_once():
    """A capped lock() returning False plus continue_lock() passes must end fully resident with
    the entry pinned exactly once."""
    cache = _make_cache()
    cache.put("m", DummyModule())
    record = cache.get("m")

    settled = cache.lock(record, None, max_move_bytes=1024)
    passes = 1
    while not settled:
        settled = cache.continue_lock(record, None, max_move_bytes=1024)
        passes += 1
        assert passes < 100, "paced lock did not converge"

    assert passes > 1, "cap did not actually split the move into multiple passes"
    cached_model = record.cached_model
    assert isinstance(cached_model, CachedModelWithPartialLoad)
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()

    # Exactly one pin across all passes: a single unlock releases it.
    assert record.is_locked
    cache.unlock(record)
    assert not record.is_locked


@requires_cuda
def test_paced_lock_without_cap_settles_in_one_call():
    """The legacy shape — no cap — must settle in the first lock() call."""
    cache = _make_cache()
    cache.put("m", DummyModule())
    record = cache.get("m")

    assert cache.lock(record, None) is True
    cached_model = record.cached_model
    assert isinstance(cached_model, CachedModelWithPartialLoad)
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()
    cache.unlock(record)


@requires_cuda
def test_paced_lock_failure_mid_stream_releases_pin():
    """continue_lock() failing must release the pin taken by the initial lock(), mirroring a
    failed lock() — the caller must not be left owing (or double-issuing) an unlock."""
    cache = _make_cache()
    cache.put("m", DummyModule())
    record = cache.get("m")

    settled = cache.lock(record, None, max_move_bytes=256)
    assert settled is False, "expected the tiny cap to truncate the first pass"

    def boom(*args, **kwargs):
        raise RuntimeError("stream failed mid-pass")

    record.cached_model.partial_load_to_vram_chunk = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stream failed mid-pass"):
        cache.continue_lock(record, None, max_move_bytes=256)

    assert not record.is_locked, "failed continue_lock left the entry pinned"


def test_lock_paced_yields_global_lock_to_waiting_writer():
    """A writer queued while a paced stream is mid-flight must acquire between passes.

    With the pre-pacing behavior (one read lock held across the whole stream) the writer could
    only acquire after the stream settled — which in this test never happens until the writer
    runs, so a regression shows up as the timeouts below expiring.
    """
    record = MagicMock()
    record.awaiting_first_use = False
    # Real bytes so _lock_paced's termination-backstop arithmetic works on the mock. Large enough
    # that the backstop's uncapped pass cannot be reached within this test's runtime — the loop
    # must exit via the writer, not the backstop.
    record.cached_model.total_bytes.return_value = 1 << 40
    cache = MagicMock()
    cache.release_first_use_grace = None

    in_stream = threading.Event()
    writer_ran = threading.Event()
    stream_done = threading.Event()

    cache.lock.return_value = False

    def continue_side_effect(*args, **kwargs):
        in_stream.set()
        # Honor the pacing contract: an uncapped pass always settles.
        if kwargs.get("max_move_bytes") is None:
            return True
        # Emulate a pass doing real work so the loop doesn't spin unrealistically fast.
        time.sleep(0.001)
        return writer_ran.is_set()

    cache.continue_lock.side_effect = continue_side_effect

    loaded = LoadedModelWithoutConfig(cache_record=record, cache=cache)

    def run_stream() -> None:
        loaded._lock_paced(None)
        stream_done.set()

    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()
    assert in_stream.wait(timeout=10), "stream never started"

    writer_done = threading.Event()

    def writer() -> None:
        with MODEL_LOAD_LOCK.write_lock():
            writer_ran.set()
        writer_done.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    assert writer_done.wait(timeout=10), "writer never acquired between paced passes"
    assert stream_done.wait(timeout=10), "paced stream did not settle after the writer finished"
    stream_thread.join(timeout=10)
    writer_thread.join(timeout=10)


def _paced_wrapper(claim: MagicMock, cache: MagicMock) -> LoadedModelWithoutConfig:
    record = MagicMock()
    record.awaiting_first_use = False
    record.cached_model.total_bytes.return_value = 1 << 20
    cache.release_first_use_grace = None
    cache.register_first_use_hold = None
    return LoadedModelWithoutConfig(cache_record=record, cache=cache, first_use_claim=claim)


def test_multi_pass_paced_enter_releases_first_use_claim_exactly_once():
    """The first-use claim is released after the whole paced stream settles, and only once.

    Releasing per pass would drop the eviction shield while later passes are still streaming;
    releasing again on re-entry would double-release a claim that no longer belongs to us.
    """
    cache = MagicMock()
    claim = MagicMock()
    releases_seen_by_pass: list[int] = []
    settled_by_pass = iter([False, False, True])

    def pass_runs_with_claim_held(*args, **kwargs):
        releases_seen_by_pass.append(claim.release.call_count)
        return next(settled_by_pass, True)

    cache.lock.side_effect = pass_runs_with_claim_held
    cache.continue_lock.side_effect = pass_runs_with_claim_held
    loaded = _paced_wrapper(claim, cache)

    with loaded:
        claim.release.assert_called_once()
    # Initial lock() plus two continue_lock() passes, none of which saw the claim released.
    assert releases_seen_by_pass == [0, 0, 0]
    assert cache.continue_lock.call_count == 2

    with loaded:
        pass
    claim.release.assert_called_once()


def test_paced_stream_failure_does_not_release_first_use_claim_synchronously():
    """A pass that raises behaves like a failed lock(): the cache already unpinned the entry, so
    no unlock is owed, and the claim stays with the wrapper so its shield outlives the failure
    until the wrapper itself is dropped (the claim's own finalizer releases it then)."""
    cache = MagicMock()
    cache.lock.return_value = False
    cache.continue_lock.side_effect = RuntimeError("stream failed mid-pass")
    claim = MagicMock()
    loaded = _paced_wrapper(claim, cache)

    with pytest.raises(RuntimeError, match="stream failed mid-pass"):
        with loaded:
            pass

    claim.release.assert_not_called()
    cache.unlock.assert_not_called()
