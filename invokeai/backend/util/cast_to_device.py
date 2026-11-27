from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None, torch.Tensor | None)


def cast_to_device(t: T, to_device: torch.device) -> T:
    """Helper function to cast an optional tensor to a target device."""
    if t is None:
        return t

    if t.device.type != to_device.type:
        if t.device.type == "cpu" and to_device.type == "cuda":
            # Deferred: a module-level import runs model_manager.load's __init__ and recreates the #10063 cycle.
            from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.weight_streaming import (
                stream_to_device,
            )

            streamed = stream_to_device(t, to_device)
            if streamed is not None:
                return streamed
        return t.to(to_device)
    return t
