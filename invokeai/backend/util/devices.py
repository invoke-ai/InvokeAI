from typing import TYPE_CHECKING, Optional, Union

import torch

from invokeai.app.services.config.config_default import get_config

if TYPE_CHECKING:
    from invokeai.app.services.shared.invocation_context import InvocationContext

NAME_TO_PRECISION = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": torch.float32,
    "autocast": torch.float32,
}


class TorchDeviceSelect:
    """Abstraction layer for torch devices."""

    @classmethod
    def choose_torch_device(cls, context: Optional["InvocationContext"] = None) -> torch.device:
        """Return the torch.device to use for accelerated inference."""
        # A future version of the model manager will have methods that mediate
        # among multiple GPUs to balance load. This provides a forward hook
        # to support those methods.
        if context:
            model_mgr = context.models
            app_config = context.config.get()
            try:
                device = model_mgr.get_free_device()
                return cls.normalize(device)
            except NotImplementedError:
                pass

        app_config = get_config()
        if app_config.device != "auto":
            device = torch.device(app_config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return cls.normalize(device)

    @classmethod
    def choose_torch_dtype(
        cls, device: Optional[torch.device] = None, context: Optional["InvocationContext"] = None
    ) -> torch.dtype:
        """Return the precision to use for accelerated inference."""
        device = device or cls.choose_torch_device(context)
        config = context.config.get() if context else get_config()
        if device.type == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(device)
            if "GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name:
                # These GPUs have limited support for float16
                return cls._to_dtype("float32")
            elif config.precision in ["auto", "autocast"]:
                # Default to float16 for CUDA devices
                return cls._to_dtype("float16")
            else:
                # Use the user-defined precision
                return cls._to_dtype(config.precision)

        elif device.type == "mps" and torch.backends.mps.is_available():
            if config.precision == "auto" or config.precision == "autocast":
                # Default to float16 for MPS devices
                return cls._to_dtype("float16")
            else:
                # Use the user-defined precision
                return cls._to_dtype(config.precision)
        # CPU / safe fallback
        return cls._to_dtype("float32")

    @classmethod
    def get_torch_device_name(cls, context: Optional[torch.device] = None) -> str:
        """Return the device name for the current torch device."""
        device = cls.choose_torch_device(context)
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
    def _to_dtype(cls, precision_name: str) -> torch.dtype:
        return NAME_TO_PRECISION[precision_name]

    @classmethod
    def _device_from_model_manager(cls) -> torch.device:
        context = cls.context
        assert context is not None
        return context.models.get_free_device()
