from typing import Any, Callable

import torch
from torch.overrides import TorchFunctionMode


def add_autocast_to_module_forward(m: torch.nn.Module, to_device: torch.device):
    """Monkey-patch m.forward(...) with a new forward(...) method that activates device autocasting for its duration."""
    old_forward = m.forward

    def new_forward(*args: Any, **kwargs: Any):
        with TorchFunctionAutocastDeviceContext(to_device):
            return old_forward(*args, **kwargs)

    m.forward = new_forward


def _cast_to_device_and_run(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], to_device: torch.device
):
    args_on_device = [a.to(to_device) if isinstance(a, torch.Tensor) else a for a in args]
    kwargs_on_device = {k: v.to(to_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return func(*args_on_device, **kwargs_on_device)


class TorchFunctionAutocastDeviceContext(TorchFunctionMode):
    def __init__(self, to_device: torch.device):
        self._to_device = to_device

    def __torch_function__(
        self, func: Callable[..., Any], types, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ):
        return _cast_to_device_and_run(func, args, kwargs or {}, self._to_device)
