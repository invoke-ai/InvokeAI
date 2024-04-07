from contextlib import AbstractContextManager, nullcontext
from typing import Optional, Union

import torch
from torch import autocast

from invokeai.app.services.config.config_default import InvokeAIAppConfig, get_config
from invokeai.app.services.shared.invocation_context import InvocationContext

NAME_TO_PRECISION = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": torch.float32,
    "autocast": torch.float32
}


class TorchDeviceSelect():
    """Abstraction layer for torch devices."""

    def __init__(self, context: Optional[InvocationContext] = None):
        if context:
            self._app_config = context.config.get()
            self._model_mgr = context.models
        else:
            self._app_config = get_config()
            self._model_mgr = None

    @property
    def app_config(self) -> InvokeAIAppConfig:
        """Return the InvokeAIAppConfig."""
        return self._app_config

    @property
    def context(self) -> Optional[InvocationContext]:
        """Return the InvocationContext, if any."""
        return self._context

    def choose_torch_device(self) -> torch.device:
        """Return the torch.device to use for accelerated inference."""

        # A future version of the model manager will have methods that mediate
        # among multiple GPUs to balance load. This provides a forward hook
        # to support those methods.
        if self._model_mgr:
            try:
                device = self._model_mgr.get_free_device()
                return self.normalize(device)
            except NotImplementedError:
                pass

        if self.app_config.device != "auto":
            device = torch.device(self.app_config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device('cpu')
        return self.normalize(device)

    def choose_torch_dtype(self, device: Optional[torch.device]=None) -> torch.dtype:
        """Return the precision to use for accelerated inference."""
        device = device or self.choose_torch_device()
        config = self.app_config
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
            if "GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name:
                # These GPUs have limited support for float16
                return self._to_dtype("float32")
            elif config.precision in ["auto", "autocast"]:
                # Default to float16 for CUDA devices
                return self._to_dtype("float16")
            else:
                # Use the user-defined precision
                return self._to_dtype(config.precision)

        elif device.type == "mps":
            if config.precision == "auto" or config.precision == "autocast":
                # Default to float16 for MPS devices
                return self._to_dtype("float16")
            else:
                # Use the user-defined precision
                return self._to_dtype(config.precision)
        # CPU / safe fallback
        return self._to_dtype("float32")

    def choose_autocast(self) -> AbstractContextManager:
        """Return an autocast context or nullcontext for the given precision string."""
        # float16 currently requires autocast to avoid errors like:
        # 'expected scalar type Half but found Float'
        precision = self.app_config.precision
        if precision == "autocast" or precision == "float16":
            return autocast
        return nullcontext

    def normalize(self, device: Union[str, torch.device]) -> torch.device:
        """Add the device index to CUDA devices."""
        device = torch.device(device)
        if device.index is None and device.type == "cuda":
            device = torch.device(device.type, torch.cuda.current_device())
        return device

    @classmethod
    def empty_cache(self) -> None:
        """Clear the GPU device cache."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.backends.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_dtype(self, precision_name: str) -> torch.dtype:
        return NAME_TO_PRECISION[precision_name]

    def _device_from_model_manager(self) -> torch.device:
        context = self.context
        assert context is not None
        return context.models.get_free_device()
