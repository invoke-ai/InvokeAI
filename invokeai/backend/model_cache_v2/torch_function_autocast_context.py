import torch
from torch.overrides import TorchFunctionMode


def cast_to_device_and_run(func, args, kwargs, to_device: torch.device):
    args_on_device = [a.to(to_device) if isinstance(a, torch.Tensor) else a for a in args]
    kwargs_on_device = {k: v.to(to_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return func(*args_on_device, **kwargs_on_device)


class TorchFunctionAutocastContext(TorchFunctionMode):
    def __init__(self, to_device: torch.device):
        self._to_device = to_device

    def __torch_function__(self, func, types, args, kwargs=None):
        return cast_to_device_and_run(func, args, kwargs or {}, self._to_device)
