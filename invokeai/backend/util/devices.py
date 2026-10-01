import threading
from collections import Counter, defaultdict
from functools import wraps
from typing import Dict, Literal, Optional, Union

import torch
from deprecated import deprecated

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.level_zero import xpu_device_is_integrated, xpu_memory_info
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.wddm import video_memory_budget

# legacy APIs
TorchPrecisionNames = Literal["float32", "float16", "bfloat16"]
CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")
XPU_DEVICE = torch.device("xpu")
MPS_DEVICE = torch.device("mps")

# Devices for which the blind free-VRAM estimate has already been reported, so the
# warning is emitted once per device rather than on every cache query.
_XPU_MEM_FALLBACK_WARNED: set[str] = set()


def _xpu_is_available() -> bool:
    """Return True if a torch XPU (Intel GPU) device is available."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@deprecated("Use TorchDevice.choose_torch_dtype() instead.")  # type: ignore
def choose_precision(device: torch.device) -> TorchPrecisionNames:
    """Return the string representation of the recommended torch device."""
    torch_dtype = TorchDevice.choose_torch_dtype(device)
    return PRECISION_TO_NAME[torch_dtype]


@deprecated("Use TorchDevice.choose_torch_device() instead.")  # type: ignore
def choose_torch_device() -> torch.device:
    """Return the torch.device to use for accelerated inference."""
    return TorchDevice.choose_torch_device()


@deprecated("Use TorchDevice.choose_torch_dtype() instead.")  # type: ignore
def torch_dtype(device: torch.device) -> torch.dtype:
    """Return the torch precision for the recommended torch device."""
    return TorchDevice.choose_torch_dtype(device)


NAME_TO_PRECISION: Dict[TorchPrecisionNames, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
PRECISION_TO_NAME: Dict[torch.dtype, TorchPrecisionNames] = {v: k for k, v in NAME_TO_PRECISION.items()}


class TorchDevice:
    """Abstraction layer for torch devices."""

    CPU_DEVICE = torch.device("cpu")
    CUDA_DEVICE = torch.device("cuda")
    XPU_DEVICE = torch.device("xpu")
    MPS_DEVICE = torch.device("mps")

    # Per-thread execution device. When set (by a session-processor worker thread bound to a
    # specific GPU), `choose_torch_device()` returns it instead of consulting the global config.
    # This is the lynchpin that makes the ~79 `choose_torch_device()` call sites (nodes, model
    # patcher, etc.) resolve to the calling worker's GPU without per-call-site changes.
    _session_device = threading.local()

    # Set when a peer-aware empty_cache (this class's or the installed torch.cuda wrapper) skipped
    # the real call because another generation device was mid-session. The skipped release is
    # not lost: `flush_deferred_empty_cache()` performs it at the next quiet moment — a busy
    # worker's step boundary or a session's end (see `flush_deferred_empty_cache`).
    _empty_cache_deferred = threading.Event()

    @classmethod
    def set_session_device(cls, device: Union[str, torch.device]) -> None:
        """Pin the calling thread's execution device. Used by multi-GPU session workers."""
        cls._session_device.device = cls.normalize(device)

    @classmethod
    def get_session_device(cls) -> Optional[torch.device]:
        """Return the calling thread's pinned execution device, or None if unset."""
        return getattr(cls._session_device, "device", None)

    @classmethod
    def clear_session_device(cls) -> None:
        """Remove the calling thread's pinned execution device, reverting to global config."""
        if hasattr(cls._session_device, "device"):
            del cls._session_device.device

    @classmethod
    def get_session_device_index(cls) -> Optional[int]:
        """Return the CUDA index of the calling thread's effective device, or None if not on CUDA.

        Resolves the thread-local session device when a worker has pinned one (multi-GPU), otherwise
        falls back to the globally-configured device. Used to annotate logs/progress with the GPU
        number so concurrent sessions can be told apart.
        """
        device = cls.get_session_device() or cls.choose_torch_device()
        return device.index if device.type in ("cuda", "xpu") else None

    @classmethod
    def get_session_device_label(cls) -> str:
        """Return a ``" (#N)"`` suffix for the calling thread's CUDA device, or ``""`` when not on CUDA."""
        index = cls.get_session_device_index()
        return f" (#{index})" if index is not None else ""

    @classmethod
    def choose_torch_device(cls) -> torch.device:
        """Return the torch.device to use for accelerated inference."""
        # A worker thread pinned to a specific GPU takes precedence over the global config.
        session_device = cls.get_session_device()
        if session_device is not None:
            return session_device
        app_config = get_config()
        if app_config.device != "auto":
            device = torch.device(app_config.device)
        elif torch.cuda.is_available():
            device = CUDA_DEVICE
        elif _xpu_is_available():
            device = XPU_DEVICE
        elif torch.backends.mps.is_available():
            device = MPS_DEVICE
        else:
            device = CPU_DEVICE
        return cls.normalize(device)

    @classmethod
    def choose_torch_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
        """Return the precision to use for accelerated inference."""
        device = device or cls.choose_torch_device()
        config = get_config()
        if device.type == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(device)
            if "GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name:
                # These GPUs have limited support for float16
                return cls._to_dtype("float32")
            elif config.precision == "auto":
                # Default to float16 for CUDA devices
                return cls._to_dtype("float16")
            else:
                # Use the user-defined precision
                return cls._to_dtype(config.precision)

        elif device.type == "xpu" and _xpu_is_available():
            if config.precision == "auto":
                # Default to float16 for XPU (Intel GPU) devices
                return cls._to_dtype("float16")
            else:
                # Use the user-defined precision
                return cls._to_dtype(config.precision)

        elif device.type == "mps" and torch.backends.mps.is_available():
            if config.precision == "auto":
                # Default to float16 for MPS devices
                return cls._to_dtype("float16")
            else:
                # Use the user-defined precision
                return cls._to_dtype(config.precision)
        # CPU / safe fallback
        return cls._to_dtype("float32")

    @classmethod
    def get_device_name(cls, device: torch.device) -> str:
        """Return the human-readable name for a torch device (e.g. 'AMD Radeon PRO W7900', 'CPU').

        Falls back to the device's own string form when the backend cannot name it. This is used
        only for labelling and logging, so a backend that cannot answer must not take the caller
        down with it -- notably an XPU device on a torch build without XPU, where ``_lazy_init``
        raises ``AssertionError``. The raised type varies between torch releases, so the except is
        deliberately broad rather than enumerating types that move.
        """
        try:
            if device.type == "cuda":
                return torch.cuda.get_device_name(device)
            if device.type == "xpu":
                return torch.xpu.get_device_name(device)
        except Exception:
            return str(device)
        return device.type.upper()

    @classmethod
    def get_torch_device_name(cls) -> str:
        """Return the device name for the current torch device."""
        return cls.get_device_name(cls.choose_torch_device())

    @classmethod
    def get_generation_devices_summary(cls, generation_devices: Union[str, list[str], None]) -> str:
        """Build a human-readable summary of the devices that will be used for generation.

        For a single device, returns just its name (e.g. ``'AMD Radeon PRO W7900'`` or ``'CPU'``). For
        multiple devices, returns a bracketed list annotating each with its GPU number and device id,
        e.g. ``'[AMD Radeon PRO W7900 #1 (cuda:0), AMD Radeon PRO W7900 #2 (cuda:1)]'``. Identically
        named GPUs get a 1-based ``#N`` suffix so they can be told apart; a uniquely named device gets
        no suffix.

        The ``#N`` suffix is tied to each device's position in the full set of available devices, not
        to its position in the (possibly filtered) `generation_devices` list. This keeps a device's
        label stable when other devices are disabled — e.g. disabling ``cuda:1`` leaves ``cuda:2`` as
        ``#3`` rather than renumbering it to ``#2`` — and keeps the backend log consistent with the
        frontend, which labels devices over the full available set.
        """
        devices = cls.get_generation_devices(generation_devices)
        if not devices:
            # Empty resolution (e.g. `generation_devices` set to an empty list) falls back to the
            # single globally-configured device.
            devices = [cls.choose_torch_device()]

        if len(devices) == 1:
            return cls.get_device_name(devices[0])

        labels = cls._get_device_labels()
        parts = [f"{labels.get(str(device)) or cls.get_device_name(device)} ({device})" for device in devices]
        return "[" + ", ".join(parts) + "]"

    @classmethod
    def _get_device_labels(cls) -> dict[str, str]:
        """Map each available device id (e.g. ``'cuda:0'``) to its disambiguated human-readable name.

        Identically-named devices get a stable 1-based ``#N`` suffix tied to their order in the full
        available-device enumeration, so the suffix does not shift when devices are filtered out of
        `generation_devices`. Mirrors the frontend's device labeling.
        """
        all_devices = cls._all_available_devices()
        names = [cls.get_device_name(device) for device in all_devices]
        name_counts = Counter(names)
        ordinals: dict[str, int] = defaultdict(int)
        labels: dict[str, str] = {}
        for device, name in zip(all_devices, names, strict=True):
            ordinals[name] += 1
            labels[str(device)] = f"{name} #{ordinals[name]}" if name_counts[name] > 1 else name
        return labels

    @classmethod
    def _all_available_devices(cls) -> list[torch.device]:
        """Every device generation could run on: all visible CUDA (or XPU) devices, or the single
        best available device (mps/cpu) when neither is available. Ignores configuration — used for
        enumeration/labeling, where filtered-out devices must still be listed."""
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
        if _xpu_is_available():
            return [torch.device(f"xpu:{index}") for index in range(torch.xpu.device_count())]
        return [cls.choose_torch_device()]

    @classmethod
    def _auto_generation_devices(cls) -> list[torch.device]:
        """The device list `generation_devices: auto` expands to.

        Unlike CUDA, Level Zero enumerates the CPU's integrated GPU alongside any discrete card,
        so on the mainstream Arc configuration (iGPU + discrete Arc) `auto` would dispatch half
        the queue to the iGPU and make it a text-encoder borrow target. Drop integrated GPUs here.

        Two deliberate limits: a device whose type cannot be determined is kept (the Level Zero
        probe returns None, and narrowing on a guess is worse than the status quo), and a machine
        whose only GPU is integrated keeps it -- otherwise there would be nothing to generate on.
        An explicit `generation_devices` list is unaffected, so an iGPU can still be opted into.
        """
        integrated: list[torch.device] = []
        remaining: list[torch.device] = []
        for device in cls._all_available_devices():
            (integrated if xpu_device_is_integrated(device) is True else remaining).append(device)
        if not integrated or not remaining:
            return integrated + remaining
        InvokeAILogger.get_logger(__name__).info(
            f"Excluding integrated GPU(s) {[str(d) for d in integrated]} from `generation_devices: auto`. "
            "List them explicitly in `generation_devices` to use them for generation."
        )
        return remaining

    @classmethod
    def get_generation_devices(cls, generation_devices: Union[str, list[str], None]) -> list[torch.device]:
        """Resolve the configured `generation_devices` into a concrete, deduplicated device list.

        - ``"auto"`` (the default) defers to an explicitly pinned legacy ``device:`` setting (an
          upgraded install that pinned e.g. ``device: cuda:1`` to avoid its display GPU must not
          silently start generating on every GPU); otherwise it expands to every visible CUDA
          device, or the single best available device (mps/cpu) when CUDA is unavailable.
        - An explicit list is normalized and deduplicated, with order preserved, and overrides the
          legacy ``device:`` setting.
        - ``None`` or an empty list yields an empty list; the caller decides the single-device fallback.
        """
        if generation_devices == "auto":
            legacy_device = get_config().device
            if legacy_device != "auto":
                device_strs: list[str] = [legacy_device]
            else:
                device_strs = [str(device) for device in cls._auto_generation_devices()]
        elif not generation_devices:
            return []
        else:
            device_strs = list(generation_devices)

        devices: list[torch.device] = []
        seen: set[str] = set()
        for device_str in device_strs:
            device = cls.normalize(device_str)
            # Fail fast on a CUDA device that doesn't exist, rather than starting a worker pinned to
            # it that only errors cryptically at the first tensor allocation. ("auto" only generates
            # valid indices, so this just validates explicitly-configured devices.)
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError(f"generation_devices requested '{device_str}', but no CUDA device is available.")
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise ValueError(
                        f"generation_devices requested '{device_str}', but only {torch.cuda.device_count()} "
                        f"CUDA device(s) are available (valid indices 0-{torch.cuda.device_count() - 1})."
                    )
            elif device.type == "xpu":
                if not _xpu_is_available():
                    raise ValueError(f"generation_devices requested '{device_str}', but no XPU device is available.")
                if device.index is not None and device.index >= torch.xpu.device_count():
                    raise ValueError(
                        f"generation_devices requested '{device_str}', but only {torch.xpu.device_count()} "
                        f"XPU device(s) are available (valid indices 0-{torch.xpu.device_count() - 1})."
                    )
            elif device.type == "mps" and not torch.backends.mps.is_available():
                raise ValueError(f"generation_devices requested '{device_str}', but MPS is not available.")
            if str(device) not in seen:
                seen.add(str(device))
                devices.append(device)
        return devices

    @classmethod
    def normalize(cls, device: Union[str, torch.device]) -> torch.device:
        """Add the device index to CUDA and XPU devices."""
        device = torch.device(device)
        if device.index is None and device.type == "cuda" and torch.cuda.is_available():
            device = torch.device(device.type, torch.cuda.current_device())
        elif device.index is None and device.type == "xpu" and _xpu_is_available():
            device = torch.device(device.type, torch.xpu.current_device())
        return device

    @classmethod
    def empty_cache(cls) -> bool:
        """Clear the GPU device cache — unless another generation device is mid-session. Says whether it ran.

        ``torch.cuda.empty_cache()`` is process-global: it takes EVERY device's
        caching-allocator mutex and cudaFree/hipFrees their cached blocks, and a free on a
        device with a long kernel in flight blocks until that kernel completes with the mutex
        held. On a multi-GPU box this freezes the busy worker — it cannot allocate or even
        deallocate a tensor — for the remainder of its current step (observed via py-spy on a
        dual-GPU ROCm rig: one worker spinning in HIP ``release_block`` inside ``emptyCache``,
        the other parked on the allocator mutex; a video step is 40-100 s).

        empty_cache is advisory — torch reuses its own cached blocks whether or not they are
        returned to the driver — so when any OTHER registered generation device is running a
        session, skip it rather than convoy. Cost of skipping: driver-level free-memory
        queries (``mem_get_info``) count the still-cached blocks as used, so the model cache's
        VRAM accounting turns conservative until a quiet-moment call runs. On single-GPU
        installs a worker never sees another busy device, so its own calls run as before; only
        a thread with no session device (the cache keep-alive timer, the cache's background
        worker) defers while the worker is mid-session, and that release now lands at the
        worker's next step boundary instead of being dropped.

        A skipped call is recorded as deferred rather than dropped: the memory it would have
        returned (a canceled session's working set, a timed-out cache's weights) stays cached in
        the allocator — invisible to the caller's own accounting, but counted as used by the
        driver and every other process — until `flush_deferred_empty_cache()` runs it from a
        quiet moment. Without that, VRAM freed on one GPU stayed resident until the peer's
        whole render finished and something else happened to call empty_cache.
        """
        if cls._another_generation_device_busy():
            cls._empty_cache_deferred.set()
            InvokeAILogger.get_logger(cls.__name__).debug(
                "Deferring empty_cache: another generation device is mid-session."
            )
            return False
        # Clear before running: a skip that races in after this point re-sets the flag, so a
        # request is never lost, only (harmlessly) repeated.
        cls._empty_cache_deferred.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _xpu_is_available():
            torch.xpu.empty_cache()
        return True

    @classmethod
    def flush_deferred_empty_cache(cls) -> None:
        """Run an empty_cache that a peer-aware skip deferred, if the pool is quiet now.

        Call this from points where the calling worker's own device is at a natural sync point
        and a driver-level free is cheap — a denoise step boundary (the progress path calls it)
        or the end of a session. It is a flag test when nothing is pending, so it is safe to call
        often. When a device other than the caller's is still mid-session the flush is skipped
        again (and stays pending): on a two-GPU box the requester's session has ended, so the
        busy worker's own step boundary is the first quiet moment and the release lands there —
        within one of its steps instead of at the end of its render. The cost to the busy worker
        is re-allocating its own cached working blocks on the next step, once per request.
        """
        if not cls._empty_cache_deferred.is_set():
            return
        cls.empty_cache()

    @classmethod
    def _another_generation_device_busy(cls) -> bool:
        """True when a generation device OTHER than this thread's session device is running.

        Imported lazily: the device pool module imports this one at top level.
        """
        from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL

        return GENERATION_DEVICE_POOL.any_other_device_busy(cls.get_session_device())

    @classmethod
    def xpu_mem_get_info(cls, device: torch.device) -> tuple[int, int]:
        """Return ``(free, total)`` VRAM in bytes for an XPU (Intel GPU) device.

        Three sources are tried in order, most accurate first:

        1. ``torch.xpu.mem_get_info()`` -- driver-global, but needs the SYCL
           ``ext_intel_free_memory`` aspect, which is missing on some driver/kernel
           combinations (notably GPU passthrough VMs and WSL2).
        2. Level Zero Sysman ``zesMemoryGetState`` -- the same driver-global figure by
           another route, frequently available when the SYCL aspect is not.
        3. ``total_memory`` minus this process's reserved bytes -- a last-resort estimate.

        The failure type is torch-version dependent -- ``RuntimeError`` from the
        missing aspect, ``AssertionError`` from ``torch.xpu._lazy_init()`` on a
        build without XPU -- so the probe catches broadly rather than enumerating
        types that move between releases.

        Only source 3 is blind to VRAM held by other processes; callers get a driver-global
        answer whenever the platform can give one. A failure to read ``total_memory`` is
        deliberately *not* swallowed: a device whose properties cannot be read is not usable
        as an execution device, and returning ``(0, 0)`` would collapse the model cache's
        available-VRAM arithmetic to a constant negative budget for the lifetime of the
        process.
        """
        try:
            return torch.xpu.mem_get_info(device)
        except Exception:
            pass

        sysman_info = xpu_memory_info(device)
        if sysman_info is not None:
            return sysman_info

        # total_memory does not depend on the unavailable free-memory aspect. Allowed to
        # raise -- see the docstring.
        total_bytes = int(torch.xpu.get_device_properties(device).total_memory)
        try:
            reserved_bytes = torch.xpu.memory_reserved(device)
        except Exception:
            reserved_bytes = 0

        device_key = str(device)
        if device_key not in _XPU_MEM_FALLBACK_WARNED:
            _XPU_MEM_FALLBACK_WARNED.add(device_key)
            InvokeAILogger.get_logger(__name__).warning(
                f"Neither torch.xpu.mem_get_info() nor Level Zero Sysman could report free VRAM "
                f"for {device}; estimating it as total memory minus this process's reserved bytes. "
                "The estimate ignores VRAM held by other processes, so the model cache may "
                "over-commit on a shared GPU. Set max_cache_vram_gb explicitly if you hit OOMs."
            )

        return (max(total_bytes - reserved_bytes, 0), total_bytes)

    @classmethod
    def cuda_mem_get_info(cls, device: torch.device) -> tuple[int, int]:
        """Return ``(free, total)`` VRAM in bytes for a CUDA or ROCm device.

        ``torch.cuda.mem_get_info``, except that on a ROCm build under Windows the free figure is capped by what the
        Windows video-memory budget still allows this process (`wddm.video_memory_budget`). There, torch's figure is
        the device total minus this process's own live allocations: it ignores other processes, and Windows pages
        allocations into shared system memory well before it is exhausted instead of failing them.
        """
        free, total = torch.cuda.mem_get_info(device)
        budget = video_memory_budget(device)
        if budget is not None:
            free = min(free, max(budget - (total - free), 0))
        return free, total

    @classmethod
    def _to_dtype(cls, precision_name: TorchPrecisionNames) -> torch.dtype:
        return NAME_TO_PRECISION[precision_name]

    @classmethod
    def choose_bfloat16_safe_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
        """Return bfloat16 if supported on the device, else fallback to float16/float32.

        This is useful for models that require bfloat16 precision (e.g., Z-Image, Flux)
        but need to run on hardware that may not support bfloat16.

        Asking costs no device memory. A device that cannot be reached at all still raises, as
        it did before: that is a broken device, not a verdict on dtypes, and it belongs to the
        caller.

        Args:
            device: The target device. If None, uses choose_torch_device().

        Returns:
            torch.bfloat16 if supported, torch.float16 for CUDA that rejects bfloat16, or
            torch.float32 for any other device that rejects it (CPU, MPS, XPU).
        """
        device = device or cls.choose_torch_device()
        try:
            # Zero elements, because this is a question about the dtype and not about memory. A
            # backend rejects an unsupported dtype from a size-independent check it runs before
            # it reaches its allocator, and an allocator is asked for nothing at all: torch's
            # `empty_mps` gates bfloat16 on the macOS version above the allocate() call, and the
            # MPS allocator skips the Metal buffer entirely for zero bytes. Probing with a real
            # element instead put an allocation between a caller and its dtype -- 256 bytes that
            # a full GPU can refuse, failing a run where no memory had been asked for yet.
            torch.empty(0, dtype=torch.bfloat16, device=device)
        except TypeError:
            # bfloat16 not supported - fallback based on device type
            if device.type == "cuda":
                return torch.float16
            return torch.float32
        return torch.bfloat16

    @classmethod
    def choose_anima_inference_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
        """Choose the inference dtype for Anima models, honoring config.precision.

        When precision is 'auto', delegates to choose_bfloat16_safe_dtype (current
        behavior). When precision is set to a specific value (float16, bfloat16,
        float32), returns that dtype directly without hardware probing.
        """
        device = device or cls.choose_torch_device()
        config = get_config()
        if config.precision == "auto":
            return cls.choose_bfloat16_safe_dtype(device)
        return NAME_TO_PRECISION[config.precision]


_PEER_AWARE_SENTINEL = "_invokeai_peer_aware"


def install_peer_aware_empty_cache() -> None:
    """Rebind ``torch.cuda.empty_cache`` itself with the peer-aware guard (idempotent).

    ``TorchDevice.empty_cache`` already skips while another generation device is mid-session,
    but third-party libraries call ``torch.cuda.empty_cache`` directly and convoy the peer all
    the same — py-spy caught diffusers doing it from inside every model materialization
    (``from_pretrained`` -> ``_load_pretrained_model`` -> ``empty_device_cache``,
    ``modeling_utils.py``; likewise ``from_single_file``), freezing the other GPU's in-flight
    denoise for the remainder of its step. Wrapping the torch entry point makes every Python
    caller inherit the policy. (Torch-internal C++ callers — e.g. MIOpen's chooseAlgorithm
    workspace-OOM fallback — bypass Python entirely and remain out of reach.)

    Installed once at startup via ``apply_monkeypatches``. Single-GPU installs never have an
    "other" busy device, so the wrapper is a pass-through there.
    """
    if getattr(torch.cuda.empty_cache, _PEER_AWARE_SENTINEL, False):
        return
    original_empty_cache = torch.cuda.empty_cache

    @wraps(original_empty_cache)
    def peer_aware_empty_cache() -> None:
        if TorchDevice._another_generation_device_busy():
            # Deferred, not dropped — see TorchDevice.empty_cache / flush_deferred_empty_cache.
            TorchDevice._empty_cache_deferred.set()
            InvokeAILogger.get_logger("TorchDevice").debug(
                "Deferring torch.cuda.empty_cache: another generation device is mid-session."
            )
            return
        TorchDevice._empty_cache_deferred.clear()
        original_empty_cache()

    setattr(peer_aware_empty_cache, _PEER_AWARE_SENTINEL, True)
    torch.cuda.empty_cache = peer_aware_empty_cache


def disable_conv_benchmark_empty_cache() -> None:
    """Stop torch's conv algorithm search from calling global emptyCache() after each find.

    When torch searches for a convolution algorithm (cuDNN benchmark mode; MIOpen on ROCm on
    every algo-cache miss, benchmark flag or not), ``findAlgorithm`` ends the search with a
    process-global ``CUDACachingAllocator::emptyCache()`` to release benchmarking workspace
    (``aten/src/ATen/native/miopen/Conv_miopen.cpp``, likewise the cuDNN path). That call takes
    every CUDA/HIP device's allocator mutex and frees their cached blocks — freezing a peer
    GPU's worker mid-step exactly like the Python-level ``torch.cuda.empty_cache`` calls
    handled by ``install_peer_aware_empty_cache``, but from C++, out of reach of that wrapper
    (observed via py-spy on a dual-GPU ROCm rig: one worker inside
    ``chooseAlgorithm -> emptyCache -> hipFree`` waiting out the other worker's 40-100 s
    denoise step; each new conv shape in the process re-triggers it).

    The call is gated on ``_cudnn_get_conv_benchmark_empty_cache()``, which torch exposes a
    setter for. Disabling it leaves the benchmarking workspace blocks cached in the allocator
    for reuse instead of returning them to the driver — the same trade the peer-aware skips
    already make everywhere else. Called only on multi-GPU installs (see
    ``DefaultSessionProcessor.start``); single-GPU installs keep torch's default behavior.

    No-op on torch builds that lack the flag (e.g. CPU-only builds).
    """
    setter = getattr(torch._C, "_cudnn_set_conv_benchmark_empty_cache", None)
    if setter is None:
        return
    setter(False)
    InvokeAILogger.get_logger("TorchDevice").debug(
        "Disabled torch's post-conv-algorithm-search global emptyCache (multi-GPU install)."
    )
