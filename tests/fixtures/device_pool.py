"""A faked two-GPU generation pool, for testing work that borrows an idle GPU on any hardware.

The pool's locks and the thread's session pin are real. Torch's own current-device calls are recorded
instead of made, an unpinned thread's default device is reported as CUDA (a CPU-only runner would
otherwise have nothing to borrow), and the dtype choice skips its device-name lookup -- the faked
``cuda:1`` does not exist on a one-GPU host, and no test here should touch a real GPU.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch

from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice

GPU0 = torch.device("cuda:0")
GPU1 = torch.device("cuda:1")


@contextmanager
def two_gpu_pool(busy: tuple[torch.device, ...] = ()) -> Iterator[list[torch.device]]:
    """Register GPU0 and GPU1, mark ``busy`` as running a session, and yield torch's recorded current-device pins.

    Resets the pool and clears the calling thread's session pin on exit.
    """
    GENERATION_DEVICE_POOL.reset()
    GENERATION_DEVICE_POOL.set_generation_devices([GPU0, GPU1])
    for device in busy:
        GENERATION_DEVICE_POOL.acquire_session(device)
    torch_pins: list[torch.device] = []
    with (
        patch("invokeai.backend.util.device_pool.set_torch_current_device", side_effect=torch_pins.append),
        patch("invokeai.backend.util.device_pool._torch_current_device", return_value=GPU0),
        # Only the unpinned case is faked: a pinned thread still resolves to its pin.
        patch.object(
            TorchDevice,
            "choose_torch_device",
            side_effect=lambda: TorchDevice.get_session_device() or torch.device("cuda"),
        ),
        patch.object(TorchDevice, "choose_torch_dtype", return_value=torch.bfloat16),
    ):
        try:
            yield torch_pins
        finally:
            TorchDevice.clear_session_device()
            GENERATION_DEVICE_POOL.reset()
