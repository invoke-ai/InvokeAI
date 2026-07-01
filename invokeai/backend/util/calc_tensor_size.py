import torch


def calc_tensor_size(t: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes."""
    return t.nelement() * t.element_size()


def calc_tensors_size(tensors: list[torch.Tensor | None]) -> int:
    """Calculate the size of a list of tensors in bytes."""
    return sum(calc_tensor_size(t) for t in tensors if t is not None)
