from dataclasses import dataclass

import torch


@dataclass
class TBLR:
    top: int
    bottom: int
    left: int
    right: int


def crop(image: torch.Tensor, crop_box: TBLR) -> torch.Tensor:
    """Extract a cropped region from an image.

    Args:
        image (torch.Tensor): The image to crop.
        crop_box (TBLR): The box to crop to.

    Returns:
        torch.Tensor: The cropped region. A copy is returned, rather than a view into the original tensor.
    """
    return image[..., crop_box.top : crop_box.bottom, crop_box.left : crop_box.right].clone()
