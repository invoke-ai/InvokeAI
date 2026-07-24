"""Process-global arbiter that lends idle generation GPUs for text-encoder offload.

In multi-GPU mode (see ``generation_devices``) the session processor runs one generation worker
per GPU. When fewer sessions are running than there are GPUs, some GPUs sit idle. This arbiter lets
a busy worker temporarily *borrow* an idle GPU to host a text encoder, instead of churning the busy
GPU's denoise model in and out of VRAM.

Correctness hinges on one rule: **a borrowed GPU must never run an encoder at the same time as a
native generation session on that same GPU.** They share that device's single ``ModelCache``, and a
model's forward pass (including in-place LoRA patching) runs with no cache lock held — so two
threads touching the same cached encoder concurrently corrupts it (garbled output).

To enforce the rule, each generation device has one lock used for *both* roles:

- A native session holds its device's lock for the entire run (blocking acquire).
- A borrower *try*-acquires another device's lock for the duration of one encoder node; if the lock
  is already held (that GPU is running, or just started, a session) the borrow simply fails and the
  encoder runs on the worker's own GPU instead.

Because borrows are non-blocking try-acquires and a session only ever blocking-acquires its *own*
device lock, there is no lock-ordering cycle — the design is deadlock-free. The only cost is that,
in the startup race where a borrow wins the lock a moment before the lent GPU's own session starts,
that session waits out the (short) encoder node before beginning.
"""

import threading
from typing import Optional

import torch

from invokeai.backend.util.devices import TorchDevice


class _GenerationDevicePool:
    """Arbitrates exclusive use of each generation device between native sessions and borrowers."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        # Registration order is preserved so borrow selection is deterministic (and therefore sticky
        # across repeated single-session generations, letting a cached encoder be reused). Maps
        # normalized device string -> that device's exclusive-use lock.
        self._device_locks: dict[str, threading.Lock] = {}
        self._order: list[str] = []

    def set_generation_devices(self, devices: list[torch.device]) -> None:
        """Register the full set of generation devices (called once at processor startup).

        Only CUDA devices participate in idle-offload; others are ignored.
        """
        with self._registry_lock:
            self._device_locks = {}
            self._order = []
            for device in devices:
                if device.type != "cuda":
                    continue
                key = str(TorchDevice.normalize(device))
                if key not in self._device_locks:
                    self._device_locks[key] = threading.Lock()
                    self._order.append(key)

    def _get_lock(self, device: torch.device) -> Optional[threading.Lock]:
        key = str(TorchDevice.normalize(device))
        with self._registry_lock:
            return self._device_locks.get(key)

    def acquire_session(self, device: Optional[torch.device]) -> None:
        """Take exclusive use of ``device`` for a native generation session (blocking).

        Waits out any in-flight borrow that won the lock first, guaranteeing the session never runs
        concurrently with a borrowed encoder on the same GPU. No-op for non-CUDA / unregistered
        devices (e.g. legacy single-device mode).
        """
        if device is None or device.type != "cuda":
            return
        lock = self._get_lock(device)
        if lock is not None:
            lock.acquire()

    def release_session(self, device: Optional[torch.device]) -> None:
        """Release the exclusive use taken by :meth:`acquire_session`."""
        if device is None or device.type != "cuda":
            return
        lock = self._get_lock(device)
        if lock is not None:
            lock.release()

    def try_borrow(self, exclude: torch.device) -> Optional[torch.device]:
        """Try to take exclusive use of an idle CUDA device other than ``exclude`` (non-blocking).

        Returns the borrowed device (whose lock the caller now holds and must release via
        :meth:`release_borrow`), or ``None`` if no other registered device is currently free.
        Selection is deterministic (lowest registration order) so repeated borrows reuse the same
        GPU and the encoder cached there.
        """
        if exclude.type != "cuda":
            return None
        exclude_key = str(TorchDevice.normalize(exclude))
        with self._registry_lock:
            candidates = [(key, self._device_locks[key]) for key in self._order if key != exclude_key]
        for key, lock in candidates:
            if lock.acquire(blocking=False):
                return torch.device(key)
        return None

    def release_borrow(self, device: torch.device) -> None:
        """Release a device taken by :meth:`try_borrow`."""
        lock = self._get_lock(device)
        if lock is not None:
            lock.release()

    def reset(self) -> None:
        """Clear all registered devices (used by tests)."""
        with self._registry_lock:
            self._device_locks = {}
            self._order = []


# Process-global singleton.
GENERATION_DEVICE_POOL = _GenerationDevicePool()
