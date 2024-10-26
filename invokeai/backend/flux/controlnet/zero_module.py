from typing import TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


def zero_module(module: T) -> T:
    """Initialize the parameters of a module to zero."""
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module
