"""Torch Device class provides torch device selection services."""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import torch
from deprecated import deprecated

from invokeai.app.services.config.config_default import get_config

if TYPE_CHECKING:
    from invokeai.backend.model_manager.config import AnyModel
    from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase

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

    _model_cache: Optional["ModelCacheBase[AnyModel]"] = None

    @classmethod
    def set_model_cache(cls, cache: "ModelCacheBase[AnyModel]"):
        """Set the current model cache."""
        cls._model_cache = cache

    @classmethod
    def choose_torch_device(cls) -> torch.device:
        """Return the torch.device to use for accelerated inference."""
        if cls._model_cache:
            return cls._model_cache.get_execution_device()
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
    def execution_devices(cls) -> List[torch.device]:
        """Return a list of torch.devices that can be used for accelerated inference."""
        if cls._model_cache:
            return cls._model_cache.execution_devices
        else:
            return [cls.choose_torch_device]

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
