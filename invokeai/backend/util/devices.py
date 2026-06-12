import threading
from typing import Dict, Literal, Optional, Union

import torch
from deprecated import deprecated

from invokeai.app.services.config.config_default import get_config

# legacy APIs
TorchPrecisionNames = Literal["float32", "float16", "bfloat16"]
CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")
MPS_DEVICE = torch.device("mps")


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
    MPS_DEVICE = torch.device("mps")

    # Per-thread execution device. When set (by a session-processor worker thread bound to a
    # specific GPU), `choose_torch_device()` returns it instead of consulting the global config.
    # This is the lynchpin that makes the ~79 `choose_torch_device()` call sites (nodes, model
    # patcher, etc.) resolve to the calling worker's GPU without per-call-site changes.
    _session_device = threading.local()

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
        return device.index if device.type == "cuda" else None

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
    def get_torch_device_name(cls) -> str:
        """Return the device name for the current torch device."""
        device = cls.choose_torch_device()
        return torch.cuda.get_device_name(device) if device.type == "cuda" else device.type.upper()

    @classmethod
    def get_generation_devices(cls, generation_devices: Union[str, list[str], None]) -> list[torch.device]:
        """Resolve the configured `generation_devices` into a concrete, deduplicated device list.

        - ``"auto"`` (the default) expands to every visible CUDA device, or the single best available
          device (mps/cpu) when CUDA is unavailable.
        - An explicit list is normalized and deduplicated, with order preserved.
        - ``None`` or an empty list yields an empty list; the caller decides the single-device fallback.
        """
        if generation_devices == "auto":
            if torch.cuda.is_available():
                device_strs: list[str] = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
            else:
                device_strs = [str(cls.choose_torch_device())]
        elif not generation_devices:
            return []
        else:
            device_strs = list(generation_devices)

        devices: list[torch.device] = []
        seen: set[str] = set()
        for device_str in device_strs:
            device = cls.normalize(device_str)
            if str(device) not in seen:
                seen.add(str(device))
                devices.append(device)
        return devices

    @classmethod
    def normalize(cls, device: Union[str, torch.device]) -> torch.device:
        """Add the device index to CUDA devices."""
        device = torch.device(device)
        if device.index is None and device.type == "cuda" and torch.cuda.is_available():
            device = torch.device(device.type, torch.cuda.current_device())
        return device

    @classmethod
    def empty_cache(cls) -> None:
        """Clear the GPU device cache."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def _to_dtype(cls, precision_name: TorchPrecisionNames) -> torch.dtype:
        return NAME_TO_PRECISION[precision_name]

    @classmethod
    def choose_bfloat16_safe_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
        """Return bfloat16 if supported on the device, else fallback to float16/float32.

        This is useful for models that require bfloat16 precision (e.g., Z-Image, Flux)
        but need to run on hardware that may not support bfloat16.

        Args:
            device: The target device. If None, uses choose_torch_device().

        Returns:
            torch.bfloat16 if supported, torch.float16 for CUDA without bfloat16 support,
            or torch.float32 for CPU/MPS.
        """
        device = device or cls.choose_torch_device()
        try:
            # Test if bfloat16 is supported on this device
            torch.tensor([1.0], dtype=torch.bfloat16, device=device)
            return torch.bfloat16
        except TypeError:
            # bfloat16 not supported - fallback based on device type
            if device.type == "cuda":
                return torch.float16
            return torch.float32

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
