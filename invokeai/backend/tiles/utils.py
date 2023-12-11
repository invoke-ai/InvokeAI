import math
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field


class TBLR(BaseModel):
    top: int
    bottom: int
    left: int
    right: int

    def __eq__(self, other):
        return (
            self.top == other.top
            and self.bottom == other.bottom
            and self.left == other.left
            and self.right == other.right
        )


class Tile(BaseModel):
    coords: TBLR = Field(description="The coordinates of this tile relative to its parent image.")
    overlap: TBLR = Field(description="The amount of overlap with adjacent tiles on each side of this tile.")

    def __eq__(self, other):
        return self.coords == other.coords and self.overlap == other.overlap


def paste(dst_image: np.ndarray, src_image: np.ndarray, box: TBLR, mask: Optional[np.ndarray] = None):
    """Paste a source image into a destination image.

    Args:
        dst_image (np.array): The destination image to paste into. Shape: (H, W, C).
        src_image (np.array): The source image to paste. Shape: (H, W, C). H and W must be compatible with 'box'.
        box (TBLR): Box defining the region in the 'dst_image' where 'src_image' will be pasted.
        mask (Optional[np.array]): A mask that defines the blending between 'src_image' and 'dst_image'.
            Range: [0.0, 1.0], Shape: (H, W). The output is calculate per-pixel according to
            `src * mask + dst * (1 - mask)`.
    """

    if mask is None:
        dst_image[box.top : box.bottom, box.left : box.right] = src_image
    else:
        mask = np.expand_dims(mask, -1)
        dst_image_box = dst_image[box.top : box.bottom, box.left : box.right]
        dst_image[box.top : box.bottom, box.left : box.right] = src_image * mask + dst_image_box * (1.0 - mask)


def seam_blend(ia1: np.ndarray, ia2: np.ndarray, blend_amount: int, x_seam: bool) -> np.ndarray:
    """Blend two overlapping tile sections using a seams to find a path.

    It is assumed that input images will be RGB np arrays and are the same size.

    Args:
        ia1 (np.array): Image array 1 Shape: (H, W, C).
        ia2 (np.array): Image array 2 Shape: (H, W, C).
        x_seam (bool): If the images should be blended on the x axis or not.
        blend_amount (int): The size of the blur to use on the seam. Half of this value will be used to avoid the edges of the image.
    """
    assert ia1.shape == ia2.shape
    assert ia2.size == ia2.size

    def shift(arr, num, fill_value=255.0):
        result = np.full_like(arr, fill_value)
        if num > 0:
            result[num:] = arr[:-num]
        elif num < 0:
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

    # Assume RGB and convert to grey
    # Could offer other options for the luminance conversion
    # BT.709 [0.2126, 0.7152, 0.0722], BT.2020 [0.2627, 0.6780, 0.0593])
    # it might not have a huge impact due to the blur that is applied over the seam
    iag1 = np.dot(ia1, [0.2989, 0.5870, 0.1140])  # BT.601 perceived brightness
    iag2 = np.dot(ia2, [0.2989, 0.5870, 0.1140])

    # Calc Difference between the images
    ia = iag2 - iag1

    # If the seam is on the X-axis rotate the array so we can treat it like a vertical seam
    if x_seam:
        ia = np.rot90(ia, 1)

    # Calc max and min X & Y limits
    # gutter is used to avoid the blur hitting the edge of the image
    gutter = math.ceil(blend_amount / 2) if blend_amount > 0 else 0
    max_y, max_x = ia.shape
    max_x -= gutter
    min_x = gutter

    # Calc the energy in the difference
    # Could offer different energy calculations e.g. Sobel or Scharr
    energy = np.abs(np.gradient(ia, axis=0)) + np.abs(np.gradient(ia, axis=1))

    # Find the starting position of the seam
    res = np.copy(energy)
    for y in range(1, max_y):
        row = res[y, :]
        rowl = shift(row, -1)
        rowr = shift(row, 1)
        res[y, :] = res[y - 1, :] + np.min([row, rowl, rowr], axis=0)

    # create an array max_y long
    lowest_energy_line = np.empty([max_y], dtype="uint16")
    lowest_energy_line[max_y - 1] = np.argmin(res[max_y - 1, min_x : max_x - 1])

    # Calc the path of the seam
    # could offer options for larger search than just 1 pixel by adjusting lpos and rpos
    for ypos in range(max_y - 2, -1, -1):
        lowest_pos = lowest_energy_line[ypos + 1]
        lpos = lowest_pos - 1
        rpos = lowest_pos + 1
        lpos = np.clip(lpos, min_x, max_x - 1)
        rpos = np.clip(rpos, min_x, max_x - 1)
        lowest_energy_line[ypos] = np.argmin(energy[ypos, lpos : rpos + 1]) + lpos

    # Draw the mask
    mask = np.zeros_like(ia)
    for ypos in range(0, max_y):
        to_fill = lowest_energy_line[ypos]
        mask[ypos, :to_fill] = 1

    # If the seam is on the X-axis rotate the array back
    if x_seam:
        mask = np.rot90(mask, 3)

    # blur the seam mask if required
    if blend_amount > 0:
        mask = cv2.blur(mask, (blend_amount, blend_amount))

    # for visual debugging
    # from PIL import Image
    # m_image = Image.fromarray((mask * 255.0).astype("uint8"))

    # copy ia2 over ia1 while applying the seam mask
    mask = np.expand_dims(mask, -1)
    blended_image = ia1 * mask + ia2 * (1.0 - mask)

    # for visual debugging
    # i1 = Image.fromarray(ia1.astype("uint8"))
    # i2 = Image.fromarray(ia2.astype("uint8"))
    # b_image = Image.fromarray(blended_image.astype("uint8"))
    # print(f"{ia1.shape}, {ia2.shape}, {mask.shape}, {blended_image.shape}")
    # print(f"{i1.size}, {i2.size}, {m_image.size}, {b_image.size}")

    return blended_image
