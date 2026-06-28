"""Tests for DefaultSessionRunner._maybe_offload_to_idle_gpu (idle-GPU text-encoder offload).

These exercise the re-pinning + borrow-lock logic without needing real CUDA: the session device is
a thread-local set via TorchDevice, and the device pool only manipulates locks keyed by device
string.
"""

import logging
import threading
import time
from collections.abc import Iterator

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    GENERATION_DEVICE_POOL.reset()
    try:
        yield
    finally:
        TorchDevice.clear_session_device()
        GENERATION_DEVICE_POOL.reset()


class _FakeInvocation:
    def __init__(self, idle_gpu_offloadable: bool, type_str: str = "fake_node"):
        self.idle_gpu_offloadable = idle_gpu_offloadable
        self._type_str = type_str

    def get_type(self) -> str:
        return self._type_str


class _FakeConfig:
    def __init__(self, enabled: bool = True):
        self.offload_text_encoders_to_idle_gpus = enabled


class _FakeServices:
    def __init__(self, enabled: bool = True):
        self.configuration = _FakeConfig(enabled)
        self.logger = logging.getLogger("test-encoder-offload")


def _runner(enabled: bool = True) -> DefaultSessionRunner:
    runner = DefaultSessionRunner()
    runner._services = _FakeServices(enabled)  # type: ignore[assignment]
    return runner


def test_encoder_node_repins_to_idle_gpu_and_restores():
    runner = _runner()
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    TorchDevice.set_session_device("cuda:0")

    with runner._maybe_offload_to_idle_gpu(_FakeInvocation(True, "flux_text_encoder")):
        # Re-pinned to the borrowed idle GPU for the duration of the node...
        assert TorchDevice.get_session_device() == torch.device("cuda:1")
        # ...and that GPU is locked, so nothing else can borrow it.
        assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) is None

    # Pin restored and the borrow released.
    assert TorchDevice.get_session_device() == torch.device("cuda:0")
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_non_encoder_node_is_not_offloaded():
    runner = _runner()
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    TorchDevice.set_session_device("cuda:0")

    with runner._maybe_offload_to_idle_gpu(_FakeInvocation(False, "denoise_latents")):
        assert TorchDevice.get_session_device() == torch.device("cuda:0")
    # Idle device was never borrowed.
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_no_offload_when_target_running_a_session():
    """With both GPUs busy (the other holds a session lock), the encoder stays on its own GPU."""
    runner = _runner()
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    TorchDevice.set_session_device("cuda:0")
    GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:1"))
    try:
        with runner._maybe_offload_to_idle_gpu(_FakeInvocation(True, "flux_text_encoder")):
            assert TorchDevice.get_session_device() == torch.device("cuda:0")
    finally:
        GENERATION_DEVICE_POOL.release_session(torch.device("cuda:1"))


def test_flag_off_disables_offload():
    runner = _runner(enabled=False)
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    TorchDevice.set_session_device("cuda:0")

    with runner._maybe_offload_to_idle_gpu(_FakeInvocation(True, "flux_text_encoder")):
        assert TorchDevice.get_session_device() == torch.device("cuda:0")
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_borrow_released_on_exception():
    runner = _runner()
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
    TorchDevice.set_session_device("cuda:0")

    with pytest.raises(RuntimeError):
        with runner._maybe_offload_to_idle_gpu(_FakeInvocation(True, "flux_text_encoder")):
            raise RuntimeError("node failed")

    # The pin is restored and the borrow lock released even though the node raised.
    assert TorchDevice.get_session_device() == torch.device("cuda:0")
    assert GENERATION_DEVICE_POOL.try_borrow(exclude=torch.device("cuda:0")) == torch.device("cuda:1")


def test_single_gpu_never_offloads():
    runner = _runner()
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0")])
    TorchDevice.set_session_device("cuda:0")

    with runner._maybe_offload_to_idle_gpu(_FakeInvocation(True, "flux_text_encoder")):
        assert TorchDevice.get_session_device() == torch.device("cuda:0")


def test_concurrent_workers_never_share_a_gpu():
    """Regression for the garbled-image bug: two sessions running at once must never use the same
    GPU for an encoder concurrently. Each worker holds its own GPU's session lock (as the processor
    does) and runs encoder nodes that may borrow the other GPU through the real offload path; we
    assert no GPU is ever occupied by two workers at the same time.

    Before the fix, a startup race let one worker offload its encoder onto the other's GPU while
    that GPU also ran a native session — both touching the same cached encoder. This test exercises
    that exact interleaving and would flag it as occupancy > 1.
    """
    GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])

    occupancy = {"cuda:0": 0, "cuda:1": 0}
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
        runner = _runner()
        own_device = torch.device(own)
        encoder = _FakeInvocation(True, "flux_text_encoder")
        for _ in range(150):
            # The processor holds the device's session lock for the whole run.
            GENERATION_DEVICE_POOL.acquire_session(own_device)
            TorchDevice.set_session_device(own_device)
            occupy(own)
            try:
                with runner._maybe_offload_to_idle_gpu(encoder):
                    current = str(TorchDevice.get_session_device())
                    if current != own:
                        # The node was re-pinned to a borrowed GPU; it must be exclusively ours.
                        occupy(current)
                        try:
                            time.sleep(0.0002)
                        finally:
                            vacate(current)
                    else:
                        time.sleep(0.0001)
            finally:
                vacate(own)
                TorchDevice.clear_session_device()
                GENERATION_DEVICE_POOL.release_session(own_device)

    threads = [threading.Thread(target=worker, args=(d,)) for d in ("cuda:0", "cuda:1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not violations, f"GPU(s) used by two workers at once: {set(violations)}"


def test_real_nodes_declare_the_marker_correctly():
    """The @invocation(idle_gpu_offloadable=...) marker is wired through to the class, and is set on
    encoder nodes but not on ordinary nodes."""
    from invokeai.app.invocations.compel import CompelInvocation
    from invokeai.app.invocations.flux_text_encoder import FluxTextEncoderInvocation
    from invokeai.app.invocations.primitives import IntegerInvocation

    assert FluxTextEncoderInvocation.idle_gpu_offloadable is True
    assert CompelInvocation.idle_gpu_offloadable is True
    # A non-encoder node defaults to False (never re-pinned to a borrowed GPU).
    assert IntegerInvocation.idle_gpu_offloadable is False
