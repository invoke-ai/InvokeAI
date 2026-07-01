import torch


def pad_with_zeros(orig_weight: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Pad a weight tensor with zeros to match the target shape."""
    expanded_weight = torch.zeros(target_shape, dtype=orig_weight.dtype, device=orig_weight.device)
    slices = tuple(slice(0, dim) for dim in orig_weight.shape)
    expanded_weight[slices] = orig_weight
    return expanded_weight
