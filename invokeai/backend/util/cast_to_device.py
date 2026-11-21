from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None, torch.Tensor | None)


def cast_to_device(t: T, to_device: torch.device) -> T:
    """Helper function to cast an optional tensor to a target device."""
    if t is None:
        return t

    if t.device.type != to_device.type:
        return t.to(to_device)
    return t
