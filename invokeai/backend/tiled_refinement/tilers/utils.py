from dataclasses import dataclass
from typing import Optional

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


def paste(dst_image: torch.Tensor, src_image: torch.Tensor, box: TBLR, mask: Optional[torch.Tensor] = None):
    """Paste a source image into a destination image.

    Args:
        dst_image (torch.Tensor): The destination image to paste into.
        src_image (torch.Tensor): The source image to paste.
        box (TBLR): Box defining the region in the dst_image where src_image will be pasted.
        mask (Optional[torch.Tensor]): A mask that defines the blending between src_image and dst_image.
            range: [0.0, 1.0], shape: (H, W). The output is calculate per-pixel according to
            `src * mask + dst * (1 - mask)`.
    """

    if mask is None:
        dst_image[..., box.top : box.bottom, box.left : box.right] = src_image
    else:
        raise NotImplementedError()
        # dst_image_box = dst_image[..., box.top : box.bottom, box.left : box.right]
        # dst_image[..., box.top : box.bottom, box.left : box.right] = src_image * mask + dst_image_box * (1.0 - mask)
