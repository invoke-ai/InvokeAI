"""Process-global arbiter that lends idle generation GPUs.

In multi-GPU mode (see ``generation_devices``) the session processor runs one generation worker
per GPU. When fewer sessions are running than there are GPUs, some GPUs sit idle. This arbiter lets
a busy worker temporarily *borrow* an idle GPU to host a text encoder, instead of churning the busy
GPU's denoise model in and out of VRAM. Model work that runs outside the session queue (Expand
Prompt, Image to Prompt, image-index embedding) borrows the same way, via :func:`idle_device_borrowed`,
so it lands on an idle GPU rather than beside a running session; when it cannot borrow, it runs
unlocked on the default device as it always did.

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
device lock, there is no lock-ordering cycle — the design is deadlock-free. A worker whose GPU is
lent does not claim queue items (see :meth:`_GenerationDevicePool.is_lent`), so they go to a free GPU
instead, and every release wakes the workers. What remains is the startup race where a borrow wins the
lock between a worker's check and its claim: that session waits out the borrow before beginning.

Note that "the whole borrowed node" includes the encoder's *model load*. Caches are per-device, so
the first borrow of a given GPU always cold-loads the encoder into that GPU's cache — seconds, not
milliseconds. Subsequent borrows of the same GPU hit that cache (borrow selection is sticky for this
reason), so the stall amortizes. Keep that in mind before marking a node ``idle_gpu_offloadable``:
the cost is bounded by how long the node runs, and a node that does substantial work *per execution*
— an autoregressive ``generate()`` loop, say — makes the stall recur on every generation instead of
amortizing away. When a node has both kinds of work, splitting it is the way out:
``ernie_image_prompt_enhancer`` is a separate, deliberately un-offloadable node for this reason, so
that ``ernie_image_text_encoder`` can stay offloadable.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Optional

import torch

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

# Device types that can lend/borrow for idle offload. MPS is excluded: it is always a single
# shared device, so there is never another GPU to borrow.
_OFFLOAD_DEVICE_TYPES = ("cuda", "xpu")


class _GenerationDevicePool:
    """Arbitrates exclusive use of each generation device between native sessions and borrowers."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        # Registration order is preserved so borrow selection is deterministic (and therefore sticky
        # across repeated single-session generations, letting a cached encoder be reused). Maps
        # normalized device string -> that device's exclusive-use lock.
        self._device_locks: dict[str, threading.Lock] = {}
        self._order: list[str] = []
        # Devices currently lent to a borrower, and the subset lent to work outside the session queue.
        # Guarded by _registry_lock; a device joins these only while its lock is held by the borrow.
        self._lent: set[str] = set()
        self._lent_off_queue: set[str] = set()
        # Called after every borrow is released, so a worker that stood aside can claim work at once.
        self._on_release: Optional[Callable[[], None]] = None

    def set_release_listener(self, listener: Optional[Callable[[], None]]) -> None:
        """Call ``listener`` (no arguments, any thread) after each borrow is released."""
        with self._registry_lock:
            self._on_release = listener

    def set_generation_devices(self, devices: list[torch.device]) -> None:
        """Register the full set of generation devices (called once at processor startup).

        Only GPU devices participate in idle-offload; others are ignored.
        """
        with self._registry_lock:
            self._device_locks = {}
            self._order = []
            self._lent = set()
            self._lent_off_queue = set()
            for device in devices:
                if device.type not in _OFFLOAD_DEVICE_TYPES:
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
        concurrently with a borrowed encoder on the same GPU. No-op for non-GPU / unregistered
        devices (e.g. legacy single-device mode).
        """
        if device is None or device.type not in _OFFLOAD_DEVICE_TYPES:
            return
        lock = self._get_lock(device)
        if lock is not None:
            lock.acquire()

    def release_session(self, device: Optional[torch.device]) -> None:
        """Release the exclusive use taken by :meth:`acquire_session`."""
        if device is None or device.type not in _OFFLOAD_DEVICE_TYPES:
            return
        lock = self._get_lock(device)
        if lock is not None:
            lock.release()

    def try_borrow(self, exclude: torch.device) -> Optional[torch.device]:
        """Try to take exclusive use of an idle GPU other than ``exclude`` (non-blocking).

        Returns the borrowed device (whose lock the caller now holds and must release via
        :meth:`release_borrow`), or ``None`` if no other registered device is currently free.
        Selection is deterministic (lowest registration order) so repeated borrows reuse the same
        GPU and the encoder cached there.

        Candidates are restricted to the excluded device's own type. The pool can hold more than
        one accelerator type (``generation_devices`` accepts e.g. ``["cuda:0", "xpu:0"]``), and
        handing a CUDA session an XPU device would load the encoder onto a different backend
        than the session it belongs to.
        """
        if exclude.type not in _OFFLOAD_DEVICE_TYPES:
            return None
        return self._try_acquire(exclude.type, exclude_key=str(TorchDevice.normalize(exclude)), off_queue=False)

    def try_borrow_off_queue(self, device_type: str) -> Optional[torch.device]:
        """Try to take an idle GPU of ``device_type`` for work outside the session queue (non-blocking).

        Same contract and selection order as :meth:`try_borrow`, with one extra refusal that keeps the
        queue moving, since such work (a prompt rewrite, say) can hold a GPU for a minute: never take
        a GPU if that would leave every GPU of the type lent to off-queue work, so the queue always
        has one to start on. A single-GPU install therefore never borrows, and keeps running
        off-queue work unlocked beside sessions as it always has, rather than making the next session
        wait for it.
        """
        if device_type not in _OFFLOAD_DEVICE_TYPES:
            return None
        return self._try_acquire(device_type, exclude_key=None, off_queue=True)

    def _try_acquire(self, device_type: str, exclude_key: Optional[str], off_queue: bool) -> Optional[torch.device]:
        # The whole selection runs under the registry lock so the off-queue refusals are checked and the
        # claim recorded atomically. Every acquire here is non-blocking, and acquire_session never holds
        # the registry lock while it blocks, so this cannot deadlock.
        with self._registry_lock:
            same_type = [key for key in self._order if torch.device(key).type == device_type]
            if off_queue and len(self._lent_off_queue.intersection(same_type)) + 1 >= len(same_type):
                return None
            for key in same_type:
                if key != exclude_key and self._device_locks[key].acquire(blocking=False):
                    self._lent.add(key)
                    if off_queue:
                        self._lent_off_queue.add(key)
                    return torch.device(key)
        return None

    def is_lent(self, device: Optional[torch.device]) -> bool:
        """True while ``device`` is lent to a borrower.

        A worker checks this before claiming a queue item: its session would only wait for the borrow
        to end, while another worker's GPU may be free to run the item now.
        """
        if device is None:
            return False
        with self._registry_lock:
            return str(TorchDevice.normalize(device)) in self._lent

    def release_borrow(self, device: torch.device) -> None:
        """Release a device taken by :meth:`try_borrow` or :meth:`try_borrow_off_queue`."""
        key = str(TorchDevice.normalize(device))
        with self._registry_lock:
            lock = self._device_locks.get(key)
            self._lent.discard(key)
            self._lent_off_queue.discard(key)
            listener = self._on_release
        if lock is not None:
            lock.release()
        if listener is not None:
            listener()

    def any_other_device_busy(self, device: Optional[torch.device]) -> bool:
        """True when any registered generation device OTHER than ``device`` is in use.

        "In use" means its exclusive-use lock is held — a native session is running on it, or
        a borrower is running an encoder there. Used to decide whether a process-global,
        peer-convoying operation (``TorchDevice.empty_cache``) is safe to run right now.
        ``device=None`` (a caller with no pinned session device, e.g. a maintenance thread)
        treats EVERY busy device as "other". Single-device and legacy installs have at most
        one registered lock, so a device's own worker always gets False — pre-multi-GPU
        behavior is unchanged there.
        """
        exclude_key = None
        if device is not None and device.type in _OFFLOAD_DEVICE_TYPES:
            exclude_key = str(TorchDevice.normalize(device))
        with self._registry_lock:
            return any(lock.locked() for key, lock in self._device_locks.items() if key != exclude_key)

    def reset(self) -> None:
        """Clear all registered devices (used by tests)."""
        with self._registry_lock:
            self._device_locks = {}
            self._order = []
            self._lent = set()
            self._lent_off_queue = set()
            self._on_release = None


# Process-global singleton.
GENERATION_DEVICE_POOL = _GenerationDevicePool()


def set_torch_current_device(device: torch.device) -> None:
    """Mirror a session-device pin onto torch's per-thread current device.

    CUDA and XPU both track a current device per thread, and index-less allocations
    (e.g. ``torch.zeros(2, device="xpu")``) resolve through it. Setting only the
    session device would leave such allocations on whichever GPU the thread was last
    pinned to -- for a borrowed idle GPU, that is the busy device the borrow exists to protect.

    Availability is checked first, mirroring TorchDevice.normalize: generation devices
    can be configured (or, in tests, faked) for a backend this process cannot actually
    initialise, and set_device would then fail or block on backend init.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(device)


def _torch_current_device(device_type: str) -> Optional[torch.device]:
    """The calling thread's current torch device of ``device_type``, or None when that backend is unavailable."""
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", torch.xpu.current_device())
    return None


@contextmanager
def idle_device_borrowed(
    exclude: Optional[torch.device] = None, purpose: Optional[str] = None
) -> Iterator[Optional[torch.device]]:
    """Pin the calling thread to an idle generation GPU for the duration of the block.

    Inside the block every device-selecting call (``TorchDevice.choose_torch_device``, the model
    loader's per-device cache, index-less torch allocations) resolves to the borrowed GPU, and that
    GPU's lock keeps a session from starting on it until the block exits. Load *and* use the model
    inside the block: a model loaded here lives in the borrowed GPU's cache.

    Yields the borrowed device, or None when nothing could be borrowed -- a CPU/MPS device, a
    legacy install, every GPU busy, or a refusal described below. The thread is then left exactly as
    it was, and the caller runs where it would have run anyway.

    With ``exclude`` (a session worker's own GPU) only other GPUs are candidates. Without it (a
    thread outside the session queue) any idle GPU of the thread's device type is, in
    ``generation_devices`` order -- except that such work never borrows on a single-GPU install and
    never holds every GPU at once (see :meth:`_GenerationDevicePool.try_borrow_off_queue`).

    ``purpose`` names the work for a debug log of where it ran (the session processor logs its own).

    On exit the thread's previous pins are restored -- or cleared, for a thread that had none. That
    matters for pooled threads (``asyncio.to_thread``): a pin left behind would silently steer
    unrelated later work on that thread to this GPU.
    """
    if exclude is not None:
        borrowed = GENERATION_DEVICE_POOL.try_borrow(exclude=exclude)
    else:
        borrowed = GENERATION_DEVICE_POOL.try_borrow_off_queue(TorchDevice.choose_torch_device().type)
    if purpose is not None:
        InvokeAILogger.get_logger(__name__).debug(
            f"{purpose} on idle device {borrowed}"
            if borrowed is not None
            else f"{purpose}: no idle device to borrow, running on {TorchDevice.choose_torch_device()}"
        )
    if borrowed is None:
        yield None
        return

    # Everything after the borrow succeeds is inside the try: if pinning raises, the borrow still has
    # to be released, or this GPU stays locked for the life of the process.
    try:
        previous_session_device = TorchDevice.get_session_device()
        previous_torch_device = _torch_current_device(borrowed.type)
        try:
            TorchDevice.set_session_device(borrowed)
            set_torch_current_device(borrowed)
            yield borrowed
        finally:
            if previous_session_device is None:
                TorchDevice.clear_session_device()
            else:
                TorchDevice.set_session_device(previous_session_device)
            if previous_torch_device is not None:
                set_torch_current_device(previous_torch_device)
    finally:
        GENERATION_DEVICE_POOL.release_borrow(borrowed)
