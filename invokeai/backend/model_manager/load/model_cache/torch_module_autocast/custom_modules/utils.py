from typing import overload

import torch


@overload
def add_nullable_tensors(a: None, b: None) -> None: ...


@overload
def add_nullable_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


@overload
def add_nullable_tensors(a: torch.Tensor, b: None) -> torch.Tensor: ...


@overload
def add_nullable_tensors(a: None, b: torch.Tensor) -> torch.Tensor: ...


def add_nullable_tensors(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return a + b
