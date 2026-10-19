"""Tests for _ModelLoadReadWriteLock's bookkeeping under abnormal exits.

The lock is write-preferring: readers defer while `_writers_waiting > 0`. That makes the
waiting count load-bearing — if a writer ever leaves it incremented, every VRAM move in the
process blocks forever, because nothing will bring it back to zero.
"""

import threading

import pytest

from invokeai.backend.model_manager.load.model_cache.model_cache import _ModelLoadReadWriteLock


def test_writer_does_not_leak_waiting_count_when_wait_raises() -> None:
    """An exception out of `_cond.wait()` must not wedge every future reader."""
    lock = _ModelLoadReadWriteLock()
    reader_holding = threading.Event()
    release_reader = threading.Event()

    def reader() -> None:
        with lock.read_lock():
            reader_holding.set()
            release_reader.wait(timeout=10)

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert reader_holding.wait(timeout=10), "reader never acquired"

    # With a reader resident, the writer is forced into its wait loop. Make that wait raise
    # the way an async exception delivered to this thread would.
    original_wait = lock._cond.wait

    def exploding_wait(*args, **kwargs):
        lock._cond.wait = original_wait  # only detonate once
        raise KeyboardInterrupt("async exception during wait")

    lock._cond.wait = exploding_wait

    with pytest.raises(KeyboardInterrupt):
        with lock.write_lock():
            pass

    assert lock._writers_waiting == 0, (
        "write_lock leaked _writers_waiting after an aborted wait; every subsequent read_lock() would block forever"
    )
    assert not lock._writer_active, "aborted writer must not be recorded as active"

    release_reader.set()
    reader_thread.join(timeout=10)
    assert not reader_thread.is_alive()

    # The real proof: a fresh reader must still be able to acquire.
    acquired = threading.Event()

    def late_reader() -> None:
        with lock.read_lock():
            acquired.set()

    late = threading.Thread(target=late_reader)
    late.start()
    assert acquired.wait(timeout=10), "readers are permanently blocked after the aborted writer"
    late.join(timeout=10)


def test_readers_blocked_by_an_aborted_writer_are_woken() -> None:
    """A writer that aborts while readers defer to it must notify them.

    The readers are parked in `_cond.wait()` because `_writers_waiting > 0`. Decrementing
    the count without a notify would leave them asleep until some unrelated event.
    """
    lock = _ModelLoadReadWriteLock()
    resident_holding = threading.Event()
    release_resident = threading.Event()

    def resident_reader() -> None:
        with lock.read_lock():
            resident_holding.set()
            release_resident.wait(timeout=10)

    resident = threading.Thread(target=resident_reader)
    resident.start()
    assert resident_holding.wait(timeout=10)

    # A writer queues up behind the resident reader, so _writers_waiting becomes 1.
    writer_waiting = threading.Event()
    writer_failed = threading.Event()
    original_wait = lock._cond.wait

    def signalling_wait(*args, **kwargs):
        writer_waiting.set()
        lock._cond.wait = original_wait
        raise KeyboardInterrupt("async exception during wait")

    def writer() -> None:
        lock._cond.wait = signalling_wait
        try:
            with lock.write_lock():
                pass
        except KeyboardInterrupt:
            writer_failed.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert writer_waiting.wait(timeout=10), "writer never reached its wait loop"
    assert writer_failed.wait(timeout=10), "writer did not abort as arranged"
    writer_thread.join(timeout=10)

    assert lock._writers_waiting == 0

    release_resident.set()
    resident.join(timeout=10)

    # A reader must now be able to proceed without any further external event.
    proceeded = threading.Event()

    def deferred_reader() -> None:
        with lock.read_lock():
            proceeded.set()

    deferred = threading.Thread(target=deferred_reader)
    deferred.start()
    assert proceeded.wait(timeout=10), "reader was not woken after the writer aborted"
    deferred.join(timeout=10)


def test_normal_write_lock_still_serializes_against_readers() -> None:
    """Guard against the try/finally restructuring breaking the happy path."""
    lock = _ModelLoadReadWriteLock()
    observed: list[str] = []
    writer_done = threading.Event()

    with lock.read_lock():
        assert lock._readers == 1

        def writer() -> None:
            with lock.write_lock():
                observed.append("writer")
                assert lock._writer_active
            writer_done.set()

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        # The writer cannot enter while a reader is resident.
        assert not writer_done.wait(timeout=0.2)
        observed.append("reader-exit")

    assert writer_done.wait(timeout=10), "writer never acquired after the reader left"
    writer_thread.join(timeout=10)

    assert observed == ["reader-exit", "writer"]
    assert lock._writers_waiting == 0
    assert not lock._writer_active
    assert lock._readers == 0
