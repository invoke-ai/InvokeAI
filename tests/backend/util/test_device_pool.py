"""Tests for the idle generation-device arbiter used by text-encoder offload and off-queue model work."""

import threading
import time
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL, idle_device_borrowed
from invokeai.backend.util.devices import TorchDevice
from tests.fixtures.device_pool import two_gpu_pool


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


def test_xpu_devices_participate_in_offload():
    """XPU devices register and lend like CUDA ones (multi-GPU Intel Arc setups)."""
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("xpu:0"), torch.device("xpu:1")])
    borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("xpu:0"))
    assert borrowed == torch.device("xpu:1")
    GENERATION_DEVICE_POOL.release_borrow(borrowed)


def test_borrow_never_crosses_device_types():
    """A mixed pool must not lend a CUDA session an XPU device (or vice versa) -- the encoder
    would land on a different backend than the session that needs its output."""
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("xpu:0")])
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("xpu:0")) is None


def test_borrow_picks_same_type_from_mixed_pool():
    GENERATION_DEVICE_POOL.set_generation_devices(
        [torch.device("cuda:0"), torch.device("xpu:0"), torch.device("cuda:1")]
    )
    borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
    assert borrowed == torch.device("cuda:1")
    GENERATION_DEVICE_POOL.release_borrow(borrowed)


def test_xpu_session_lock_blocks_borrow():
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("xpu:0"), torch.device("xpu:1")])
    GENERATION_DEVICE_POOL.acquire_session(torch.device("xpu:1"))
    try:
        assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("xpu:0")) is None
    finally:
        GENERATION_DEVICE_POOL.release_session(torch.device("xpu:1"))
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("xpu:0")) == torch.device("xpu:1")


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


class TestAnyOtherDeviceBusy:
    """any_other_device_busy() gates process-global empty_cache against peer convoys."""

    def test_idle_pool_reports_not_busy(self):
        GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
        try:
            assert not GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:0"))
            assert not GENERATION_DEVICE_POOL.any_other_device_busy(None)
        finally:
            GENERATION_DEVICE_POOL.reset()

    def test_own_session_does_not_count(self):
        GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
        try:
            GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))
            try:
                # cuda:0's own worker sees no OTHER busy device...
                assert not GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:0"))
                # ...but cuda:1's worker (and an unpinned maintenance thread) must defer to it.
                assert GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:1"))
                assert GENERATION_DEVICE_POOL.any_other_device_busy(None)
            finally:
                GENERATION_DEVICE_POOL.release_session(torch.device("cuda:0"))
            assert not GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:1"))
        finally:
            GENERATION_DEVICE_POOL.reset()

    def test_borrow_counts_as_busy(self):
        GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
        try:
            borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
            assert borrowed is not None
            try:
                assert GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:0"))
            finally:
                GENERATION_DEVICE_POOL.release_borrow(borrowed)
        finally:
            GENERATION_DEVICE_POOL.reset()

    def test_single_device_never_busy_for_own_worker(self):
        """Single-GPU installs must keep pre-multi-GPU empty_cache behavior."""
        GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0")])
        try:
            GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))
            try:
                assert not GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:0"))
            finally:
                GENERATION_DEVICE_POOL.release_session(torch.device("cuda:0"))
        finally:
            GENERATION_DEVICE_POOL.reset()

    def test_empty_registry_never_busy(self):
        """Legacy mode registers no devices; empty_cache must run as before."""
        GENERATION_DEVICE_POOL.reset()
        assert not GENERATION_DEVICE_POOL.any_other_device_busy(None)
        assert not GENERATION_DEVICE_POOL.any_other_device_busy(torch.device("cuda:0"))


# --- Borrowing from outside the session queue -------------------------------------------------


@pytest.fixture
def off_queue_thread() -> Iterator[list[torch.device]]:
    with two_gpu_pool() as torch_pins:
        yield torch_pins


def test_off_queue_borrow_takes_the_first_idle_gpu_in_registration_order() -> None:
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:1"), torch.device("cuda:0")])
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))  # a render; it still serves the queue
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda") == torch.device("cuda:1")
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("mps") is None


def test_off_queue_borrow_never_takes_the_last_gpu_left_for_the_queue() -> None:
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda") == torch.device("cuda:0")
    # cuda:1 is idle, but lending it too would leave the queue nowhere to start a session.
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda") is None
    # A running session's encoder offload is not off-queue work, and may still borrow it.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_off_queue_borrow_is_refused_on_a_single_gpu() -> None:
    """With one GPU, a borrow would only make the next session wait for the off-queue work."""
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0")])
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda") is None


def test_off_queue_borrow_follows_the_thread_device_type() -> None:
    GENERATION_DEVICE_POOL.set_generation_devices(
        [torch.device("cuda:0"), torch.device("xpu:0"), torch.device("xpu:1")]
    )
    with (
        patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("xpu")),
        patch("invokeai.backend.util.device_pool.set_torch_current_device"),
        patch("invokeai.backend.util.device_pool._torch_current_device", return_value=None),
    ):
        try:
            with idle_device_borrowed() as device:
                assert device == torch.device("xpu:0")
        finally:
            TorchDevice.clear_session_device()


def test_lent_state_and_release_listener() -> None:
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    released: list[bool] = []
    GENERATION_DEVICE_POOL.set_release_listener(lambda: released.append(True))

    borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0"))
    assert borrowed is not None and GENERATION_DEVICE_POOL.is_lent(borrowed)
    assert not GENERATION_DEVICE_POOL.is_lent(torch.device("cuda:0"))
    GENERATION_DEVICE_POOL.release_borrow(borrowed)
    assert not GENERATION_DEVICE_POOL.is_lent(borrowed)
    assert released == [True]


def test_off_queue_work_moves_to_the_idle_gpu_and_unpins_after(off_queue_thread: list[torch.device]) -> None:
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))  # a render is running on GPU 0

    with idle_device_borrowed() as device:
        assert device == torch.device("cuda:1")
        assert TorchDevice.choose_torch_device() == torch.device("cuda:1")
        assert GENERATION_DEVICE_POOL.is_lent(torch.device("cuda:1"))

    # A pooled thread must come back unpinned, or later unrelated work would follow the pin.
    assert TorchDevice.get_session_device() is None
    assert off_queue_thread == [torch.device("cuda:1"), torch.device("cuda:0")]
    assert not GENERATION_DEVICE_POOL.is_lent(torch.device("cuda:1"))


def test_off_queue_work_stays_put_when_every_gpu_is_busy(off_queue_thread: list[torch.device]) -> None:
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:1"))

    with idle_device_borrowed() as device:
        assert device is None
        assert TorchDevice.get_session_device() is None

    assert off_queue_thread == []


def test_off_queue_borrow_is_released_and_unpinned_when_the_work_raises(
    off_queue_thread: list[torch.device],
) -> None:
    with pytest.raises(RuntimeError, match="out of memory"):
        with idle_device_borrowed():
            raise RuntimeError("out of memory")

    assert TorchDevice.get_session_device() is None
    assert not GENERATION_DEVICE_POOL.is_lent(torch.device("cuda:0"))


def test_off_queue_borrow_is_a_no_op_without_a_registered_pool() -> None:
    """Legacy installs (no generation devices) register nothing, so nothing changes for them."""
    with idle_device_borrowed() as device:
        assert device is None
    assert TorchDevice.get_session_device() is None


def test_off_queue_borrow_is_held_until_the_pins_are_restored(off_queue_thread: list[torch.device]) -> None:
    """Releasing first would let a session start on the GPU while this thread still points at it."""
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:0"))
    lent_at_each_pin: list[bool] = []
    with patch(
        "invokeai.backend.util.device_pool.set_torch_current_device",
        side_effect=lambda device: lent_at_each_pin.append(GENERATION_DEVICE_POOL.is_lent(torch.device("cuda:1"))),
    ):
        with idle_device_borrowed():
            pass
    assert lent_at_each_pin == [True, True]  # the pin, then the restore
