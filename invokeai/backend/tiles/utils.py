from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class TBLR(BaseModel):
    top: int
    bottom: int
    left: int
    right: int


class Tile(BaseModel):
    coords: TBLR = Field(description="The coordinates of this tile relative to its parent image.")
    overlap: TBLR = Field(description="The amount of overlap with adjacent tiles on each side of this tile.")


def paste(dst_image: np.ndarray, src_image: np.ndarray, box: TBLR, mask: Optional[np.ndarray] = None):
    """Paste a source image into a destination image.

    Args:
        dst_image (torch.Tensor): The destination image to paste into. Shape: (H, W, C).
        src_image (torch.Tensor): The source image to paste. Shape: (H, W, C). H and W must be compatible with 'box'.
        box (TBLR): Box defining the region in the 'dst_image' where 'src_image' will be pasted.
        mask (Optional[torch.Tensor]): A mask that defines the blending between 'src_image' and 'dst_image'.
            Range: [0.0, 1.0], Shape: (H, W). The output is calculate per-pixel according to
            `src * mask + dst * (1 - mask)`.
    """

    if mask is None:
        dst_image[box.top : box.bottom, box.left : box.right] = src_image
    else:
        mask = np.expand_dims(mask, -1)
        dst_image_box = dst_image[box.top : box.bottom, box.left : box.right]
        dst_image[box.top : box.bottom, box.left : box.right] = src_image * mask + dst_image_box * (1.0 - mask)
