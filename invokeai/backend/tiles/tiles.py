import math
from typing import Union

import numpy as np

from invokeai.backend.tiles.utils import TBLR, Tile, paste


def calc_tiles_with_overlap(
    image_height: int, image_width: int, tile_height: int, tile_width: int, overlap: int = 0
) -> list[Tile]:
    """Calculate the tile coordinates for a given image shape under a simple tiling scheme with overlaps.

    Args:
        image_height (int): The image height in px.
        image_width (int): The image width in px.
        tile_height (int): The tile height in px. All tiles will have this height.
        tile_width (int): The tile width in px. All tiles will have this width.
        overlap (int, optional): The target overlap between adjacent tiles. If the tiles do not evenly cover the image
            shape, then the last row/column of tiles will overlap more than this. Defaults to 0.

    Returns:
        list[Tile]: A list of tiles that cover the image shape. Ordered from left-to-right, top-to-bottom.
    """
    assert image_height >= tile_height
    assert image_width >= tile_width
    assert overlap < tile_height
    assert overlap < tile_width

    non_overlap_per_tile_height = tile_height - overlap
    non_overlap_per_tile_width = tile_width - overlap

    num_tiles_y = math.ceil((image_height - overlap) / non_overlap_per_tile_height)
    num_tiles_x = math.ceil((image_width - overlap) / non_overlap_per_tile_width)

    # tiles[y * num_tiles_x + x] is the tile for the y'th row, x'th column.
    tiles: list[Tile] = []

    # Calculate tile coordinates. (Ignore overlap values for now.)
    for tile_idx_y in range(num_tiles_y):
        for tile_idx_x in range(num_tiles_x):
            tile = Tile(
                coords=TBLR(
                    top=tile_idx_y * non_overlap_per_tile_height,
                    bottom=tile_idx_y * non_overlap_per_tile_height + tile_height,
                    left=tile_idx_x * non_overlap_per_tile_width,
                    right=tile_idx_x * non_overlap_per_tile_width + tile_width,
                ),
                overlap=TBLR(top=0, bottom=0, left=0, right=0),
            )

            if tile.coords.bottom > image_height:
                # If this tile would go off the bottom of the image, shift it so that it is aligned with the bottom
                # of the image.
                tile.coords.bottom = image_height
                tile.coords.top = image_height - tile_height

            if tile.coords.right > image_width:
                # If this tile would go off the right edge of the image, shift it so that it is aligned with the
                # right edge of the image.
                tile.coords.right = image_width
                tile.coords.left = image_width - tile_width

            tiles.append(tile)

    def get_tile_or_none(idx_y: int, idx_x: int) -> Union[Tile, None]:
        if idx_y < 0 or idx_y > num_tiles_y or idx_x < 0 or idx_x > num_tiles_x:
            return None
        return tiles[idx_y * num_tiles_x + idx_x]

    # Iterate over tiles again and calculate overlaps.
    for tile_idx_y in range(num_tiles_y):
        for tile_idx_x in range(num_tiles_x):
            cur_tile = get_tile_or_none(tile_idx_y, tile_idx_x)
            top_neighbor_tile = get_tile_or_none(tile_idx_y - 1, tile_idx_x)
            left_neighbor_tile = get_tile_or_none(tile_idx_y, tile_idx_x - 1)

            assert cur_tile is not None

            # Update cur_tile top-overlap and corresponding top-neighbor bottom-overlap.
            if top_neighbor_tile is not None:
                cur_tile.overlap.top = max(0, top_neighbor_tile.coords.bottom - cur_tile.coords.top)
                top_neighbor_tile.overlap.bottom = cur_tile.overlap.top

            # Update cur_tile left-overlap and corresponding left-neighbor right-overlap.
            if left_neighbor_tile is not None:
                cur_tile.overlap.left = max(0, left_neighbor_tile.coords.right - cur_tile.coords.left)
                left_neighbor_tile.overlap.right = cur_tile.overlap.left

    return tiles


def merge_tiles_with_linear_blending(
    dst_image: np.ndarray, tiles: list[Tile], tile_images: list[np.ndarray], blend_amount: int
):
    """Merge a set of image tiles into `dst_image` with linear blending between the tiles.

    We expect every tile edge to either:
    1) have an overlap of 0, because it is aligned with the image edge, or
    2) have an overlap >= blend_amount.
    If neither of these conditions are satisfied, we raise an exception.

    The linear blending is centered at the halfway point of the overlap between adjacent tiles.

    Args:
        dst_image (np.ndarray): The destination image. Shape: (H, W, C).
        tiles (list[Tile]): The list of tiles describing the locations of the respective `tile_images`.
        tile_images (list[np.ndarray]): The tile images to merge into `dst_image`.
        blend_amount (int): The amount of blending (in px) between adjacent overlapping tiles.
    """
    # Sort tiles and images first by left x coordinate, then by top y coordinate. During tile processing, we want to
    # iterate over tiles left-to-right, top-to-bottom.
    tiles_and_images = list(zip(tiles, tile_images, strict=True))
    tiles_and_images = sorted(tiles_and_images, key=lambda x: x[0].coords.left)
    tiles_and_images = sorted(tiles_and_images, key=lambda x: x[0].coords.top)

    # Prepare 1D linear gradients for blending.
    gradient_left_x = np.linspace(start=0.0, stop=1.0, num=blend_amount)
    gradient_top_y = np.linspace(start=0.0, stop=1.0, num=blend_amount)
    # Convert shape: (blend_amount, ) -> (blend_amount, 1). The extra dimension enables the gradient to be applied
    # to a 2D image via broadcasting. Note that no additional dimension is needed on gradient_left_x for
    # broadcasting to work correctly.
    gradient_top_y = np.expand_dims(gradient_top_y, axis=1)

    for tile, tile_image in tiles_and_images:
        # We expect tiles to be written left-to-right, top-to-bottom. We construct a mask that applies linear blending
        # to the top and to the left of the current tile. The inverse linear blending is automatically applied to the
        # bottom/right of the tiles that have already been pasted by the paste(...) operation.
        tile_height, tile_width, _ = tile_image.shape
        mask = np.ones(shape=(tile_height, tile_width), dtype=np.float64)
        # Top blending:
        if tile.overlap.top > 0:
            assert tile.overlap.top >= blend_amount
            # Center the blending gradient in the middle of the overlap.
            blend_start_top = tile.overlap.top // 2 - blend_amount // 2
            # The region above the blending region is masked completely.
            mask[:blend_start_top, :] = 0.0
            # Apply the blend gradient to the mask. Note that we use `*=` rather than `=` to achieve more natural
            # behavior on the corners where vertical and horizontal blending gradients overlap.
            mask[blend_start_top : blend_start_top + blend_amount, :] *= gradient_top_y
            # For visual debugging:
            # tile_image[blend_start_top : blend_start_top + blend_amount, :] = 0

        # Left blending:
        # (See comments under 'top blending' for an explanation of the logic.)
        if tile.overlap.left > 0:
            assert tile.overlap.left >= blend_amount
            blend_start_left = tile.overlap.left // 2 - blend_amount // 2
            mask[:, :blend_start_left] = 0.0
            mask[:, blend_start_left : blend_start_left + blend_amount] *= gradient_left_x
            # For visual debugging:
            # tile_image[:, blend_start_left : blend_start_left + blend_amount] = 0

        paste(dst_image=dst_image, src_image=tile_image, box=tile.coords, mask=mask)
