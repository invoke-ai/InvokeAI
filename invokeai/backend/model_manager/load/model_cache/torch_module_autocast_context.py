from typing import Any, Iterator

import torch


def _add_autocast_to_module(m: torch.nn.Module, to_device: torch.device):
    def forward_pre_hook(module: torch.nn.Module, args: tuple[Any, ...]):
        # Backup shallow copies of the existing parameters and buffers.
        module._parameters_backup = {k: v for k, v in module._parameters.items()}
        module._buffers_backup = {k: v for k, v in module._buffers.items()}

        # Replace the parameters and buffers with their device-casted versions.
        for key, param in module._parameters.items():
            if param is not None and param.device.type != to_device.type:
                out_param = torch.nn.Parameter(param.to(to_device, copy=True), requires_grad=param.requires_grad)
                module._parameters[key] = out_param

        for key, buffer in module._buffers.items():
            if buffer is not None and buffer.device.type != to_device.type:
                out_buffer = buffer.to(to_device, copy=True)
                module._buffers[key] = out_buffer

    def forward_post_hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any):
        # Restore the original parameters and buffers.
        if hasattr(module, "_parameters_backup"):
            module._parameters = module._parameters_backup
            del module._parameters_backup
        if hasattr(module, "_buffers_backup"):
            module._buffers = module._buffers_backup
            del module._buffers_backup

    m.register_forward_pre_hook(forward_pre_hook)
    m.register_forward_hook(forward_post_hook, always_call=True)


def _add_autocast_to_module_forward(m: torch.nn.Module, to_device: torch.device):
    m.forward = _cast_to_device_and_run(m.forward, to_device)


def _is_leaf_module(m: torch.nn.Module) -> bool:
    for _ in m.children():
        # If the the m.children() generator returns a value, then m is not a leaf module.
        return False
    # If we get here then the m.children() generator returned an empty generator, so m is a leaf module.
    return True


def _named_leaf_modules(m: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
    """An iterator over all leaf modules in the module hierarchy."""
    for name, module in m.named_modules():
        if _is_leaf_module(module):
            yield name, module


def add_autocast_to_all_leaf_modules(m: torch.nn.Module, to_device: torch.device):
    for name, module in _named_leaf_modules(m):
        _add_autocast_to_module(module, to_device)


def add_autocast_to_modules(m: torch.nn.Module, to_device: torch.device):
    for name, module in m.named_modules():
        if isinstance(
            module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.GroupNorm, torch.nn.Embedding)
        ):
            _add_autocast_to_module(module, to_device)


# def _cast_to_device_and_run(
#     func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], to_device: torch.device
# ):
#     args_on_device = [a.to(to_device) if isinstance(a, torch.Tensor) else a for a in args]
#     kwargs_on_device = {k: v.to(to_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
#     return func(*args_on_device, **kwargs_on_device)

# - Fastest option is if we know exactly which params need to be cast.
#     - i.e. patch at module level
# - Inheritance vs composition?
#     - Inheritance means that the module looks slightly closer to the original module in case other layers want to
#       patch it.
#     - Composition means that the module looks
