import torch


def to_standard_mask_dim(mask: torch.Tensor) -> torch.Tensor:
    """Standardize the dimensions of a mask tensor.

    Args:
        mask (torch.Tensor): A mask tensor. The shape can be (1, h, w) or (h, w).

    Returns:
        torch.Tensor: The output mask tensor. The shape is (1, h, w).
    """
    # Get the mask height and width.
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}. Expected (1, h, w) or (h, w).")

    return mask


def to_standard_float_mask(mask: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Standardize the format of a mask tensor.

    Args:
        mask (torch.Tensor): A mask tensor. The dtype can be any bool, float, or int type. The shape must be (1, h, w)
            or (h, w).

        out_dtype (torch.dtype): The dtype of the output mask tensor. Must be a float type.

    Returns:
        torch.Tensor: The output mask tensor. The dtype is out_dtype. The shape is (1, h, w). All values are either 0.0
            or 1.0.
    """

    if not out_dtype.is_floating_point:
        raise ValueError(f"out_dtype must be a float type, but got {out_dtype}")

    mask = to_standard_mask_dim(mask)
    mask = mask.to(out_dtype)

    # Set masked regions to 1.0.
    if mask.dtype == torch.bool:
        mask = mask.to(out_dtype)
    else:
        mask = mask.to(out_dtype)
        mask_region = mask > 0.5
        mask[mask_region] = 1.0
        mask[~mask_region] = 0.0

    return mask
