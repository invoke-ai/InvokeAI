from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None, torch.Tensor | None)


def cast_to_dtype(t: T, to_dtype: torch.dtype) -> T:
    """Helper function to cast an optional tensor to a target dtype."""
    if t is None:
        return t

    if t.dtype != to_dtype:
        return t.to(dtype=to_dtype)
    return t
