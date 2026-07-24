"""Tests for the idle generation-device arbiter used by text-encoder offload."""

import threading
import time
from collections.abc import Iterator

import pytest
import torch

from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL


@pytest.fixture(autouse=True)
def reset_pool() -> Iterator[None]:
    """The arbiter is a process-global singleton; reset it around each test."""
    GENERATION_DEVICE_POOL.reset()
    try:
        yield
    finally:
        GENERATION_DEVICE_POOL.reset()


def test_borrow_picks_lowest_other_device():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_borrow_excludes_requesting_device():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:1")) == torch.device("cuda:0")


def test_session_lock_blocks_borrow():
    """A device held by a native session cannot be borrowed."""
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:1"))
    try:
        # The only other device is busy with a session -> no borrow.
        assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None
    finally:
        GENERATION_DEVICE_POOL.release_session(torch.device("cuda:1"))
    # Released -> borrowable again.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_borrow_blocks_session_until_released():
    """A native session acquire waits for an in-flight borrow on the same device (startup race)."""
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
    assert borrowed == torch.device("cuda:1")

    acquired = threading.Event()

    def native_session():
        GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:1"))
        acquired.set()

    t = threading.Thread(target=native_session)
    t.start()
    # The session must block while the borrow holds cuda:1.
    assert not acquired.wait(timeout=0.2)
    GENERATION_DEVICE_POOL.release_borrow(torch.device("cuda:1"))
    # Now it can proceed.
    assert acquired.wait(timeout=2.0)
    t.join()
    GENERATION_DEVICE_POOL.release_session(torch.device("cuda:1"))


def test_two_borrowers_do_not_share_a_device():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    first = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
    assert first == torch.device("cuda:1")
    # A second borrower (also from cuda:0) finds the only other device already taken -> None.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None
    GENERATION_DEVICE_POOL.release_borrow(first)


def test_single_device_has_no_borrow_target():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0")])
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None


def test_deterministic_lowest_order_selection():
    GENERATION_DEVICE_POOL.set_generation_devices(
        [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    )
    # cuda:1 and cuda:2 are both free; the lowest-order one (cuda:1) is chosen, and the choice is
    # stable across calls (release then re-borrow) so a cached encoder can be reused.
    for _ in range(3):
        device = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
        assert device == torch.device("cuda:1")
        GENERATION_DEVICE_POOL.release_borrow(device)


def test_non_cuda_devices_ignored():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cpu"), torch.device("cuda:0")])
    # Only cuda:0 registered; nothing else to borrow.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None
    # A non-cuda requester never borrows, and a non-cuda session acquire is a no-op.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cpu")) is None
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cpu"))  # must not raise
    GENERATION_DEVICE_POOL.release_session(torch.device("cpu"))


def test_empty_pool_returns_none():
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None


def test_concurrent_sessions_and_borrows_never_overlap_on_a_device():
    """Regression: a GPU must never be used by a native session and a borrowed encoder at the same
    time. That overlap is exactly what corrupted a shared encoder and produced garbled images. Here
    we stress the arbiter from several threads and assert exclusive use is always honored.

    With only the busy-flag approach this used before the fix, a borrow could win against a starting
    session and both would "use" the device — which this test would catch as occupancy > 1.
    """
    device_strs = ["cuda:0", "cuda:1", "cuda:2"]
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device(d) for d in device_strs])

    occupancy = dict.fromkeys(device_strs, 0)
    occ_lock = threading.Lock()
    violations: list[str] = []

    def occupy(device_str: str) -> None:
        with occ_lock:
            occupancy[device_str] += 1
            if occupancy[device_str] > 1:
                violations.append(device_str)

    def vacate(device_str: str) -> None:
        with occ_lock:
            occupancy[device_str] -= 1

    def worker(own: str) -> None:
        own_device = torch.device(own)
        for _ in range(200):
            GENERATION_DEVICE_POOL.acquire_session(own_device)
            occupy(own)  # this thread now exclusively owns `own` (as a native session would)
            try:
                borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=own_device)
                if borrowed is not None:
                    occupy(str(borrowed))
                    try:
                        time.sleep(0.0002)  # widen the window so any overlap is observed
                    finally:
                        vacate(str(borrowed))
                        GENERATION_DEVICE_POOL.release_borrow(borrowed)
            finally:
                vacate(own)
                GENERATION_DEVICE_POOL.release_session(own_device)

    threads = [threading.Thread(target=worker, args=(d,)) for d in device_strs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not violations, f"device(s) used concurrently by a session and a borrow: {set(violations)}"
