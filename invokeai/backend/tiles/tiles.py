import math
from typing import Union

import numpy as np

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.tiles.utils import TBLR, Tile, paste, seam_blend


def calc_overlap(tiles: list[Tile], num_tiles_x: int, num_tiles_y: int) -> list[Tile]:
    """Calculate and update the overlap of a list of tiles.

    Args:
        tiles (list[Tile]): The list of tiles describing the locations of the respective `tile_images`.
        num_tiles_x: the number of tiles on the x axis.
        num_tiles_y: the number of tiles on the y axis.
    """

    def get_tile_or_none(idx_y: int, idx_x: int) -> Union[Tile, None]:
        if idx_y < 0 or idx_y > num_tiles_y or idx_x < 0 or idx_x > num_tiles_x:
            return None
        return tiles[idx_y * num_tiles_x + idx_x]

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

    return calc_overlap(tiles, num_tiles_x, num_tiles_y)


def calc_tiles_even_split(
    image_height: int, image_width: int, num_tiles_x: int, num_tiles_y: int, overlap: int = 0
) -> list[Tile]:
    """Calculate the tile coordinates for a given image shape with the number of tiles requested.

    Args:
        image_height (int): The image height in px.
        image_width (int): The image width in px.
        num_x_tiles (int): The number of tile to split the image into on the X-axis.
        num_y_tiles (int): The number of tile to split the image into on the Y-axis.
        overlap (int, optional): The overlap between adjacent tiles in pixels. Defaults to 0.

    Returns:
        list[Tile]: A list of tiles that cover the image shape. Ordered from left-to-right, top-to-bottom.
    """
    # Ensure the image is divisible by LATENT_SCALE_FACTOR
    if image_width % LATENT_SCALE_FACTOR != 0 or image_height % LATENT_SCALE_FACTOR != 0:
        raise ValueError(f"image size (({image_width}, {image_height})) must be divisible by {LATENT_SCALE_FACTOR}")

    # Calculate the tile size based on the number of tiles and overlap, and ensure it's divisible by 8 (rounding down)
    if num_tiles_x > 1:
        # ensure the overlap is not more than the maximum overlap if we only have 1 tile then we dont care about overlap
        assert overlap <= image_width - (LATENT_SCALE_FACTOR * (num_tiles_x - 1))
        tile_size_x = LATENT_SCALE_FACTOR * math.floor(
            ((image_width + overlap * (num_tiles_x - 1)) // num_tiles_x) / LATENT_SCALE_FACTOR
        )
        assert overlap < tile_size_x
    else:
        tile_size_x = image_width

    if num_tiles_y > 1:
        # ensure the overlap is not more than the maximum overlap if we only have 1 tile then we dont care about overlap
        assert overlap <= image_height - (LATENT_SCALE_FACTOR * (num_tiles_y - 1))
        tile_size_y = LATENT_SCALE_FACTOR * math.floor(
            ((image_height + overlap * (num_tiles_y - 1)) // num_tiles_y) / LATENT_SCALE_FACTOR
        )
        assert overlap < tile_size_y
    else:
        tile_size_y = image_height

    # tiles[y * num_tiles_x + x] is the tile for the y'th row, x'th column.
    tiles: list[Tile] = []

    # Calculate tile coordinates. (Ignore overlap values for now.)
    for tile_idx_y in range(num_tiles_y):
        # Calculate the top and bottom of the row
        top = tile_idx_y * (tile_size_y - overlap)
        bottom = min(top + tile_size_y, image_height)
        # For the last row adjust bottom to be the height of the image
        if tile_idx_y == num_tiles_y - 1:
            bottom = image_height

        for tile_idx_x in range(num_tiles_x):
            # Calculate the left & right coordinate of each tile
            left = tile_idx_x * (tile_size_x - overlap)
            right = min(left + tile_size_x, image_width)
            # For the last tile in the row adjust right to be the width of the image
            if tile_idx_x == num_tiles_x - 1:
                right = image_width

            tile = Tile(
                coords=TBLR(top=top, bottom=bottom, left=left, right=right),
                overlap=TBLR(top=0, bottom=0, left=0, right=0),
            )

            tiles.append(tile)

    return calc_overlap(tiles, num_tiles_x, num_tiles_y)


def calc_tiles_min_overlap(
    image_height: int,
    image_width: int,
    tile_height: int,
    tile_width: int,
    min_overlap: int = 0,
) -> list[Tile]:
    """Calculate the tile coordinates for a given image shape under a simple tiling scheme with overlaps.

    Args:
        image_height (int): The image height in px.
        image_width (int): The image width in px.
        tile_height (int): The tile height in px. All tiles will have this height.
        tile_width (int): The tile width in px. All tiles will have this width.
        min_overlap (int): The target minimum overlap between adjacent tiles. If the tiles do not evenly cover the image
            shape, then the overlap will be spread between the tiles.

    Returns:
        list[Tile]: A list of tiles that cover the image shape. Ordered from left-to-right, top-to-bottom.
    """

    assert min_overlap < tile_height
    assert min_overlap < tile_width

    # catches the cases when the tile size is larger than the images size and adjusts the tile size
    if image_width < tile_width:
        tile_width = image_width

    if image_height < tile_height:
        tile_height = image_height

    num_tiles_x = math.ceil((image_width - min_overlap) / (tile_width - min_overlap))
    num_tiles_y = math.ceil((image_height - min_overlap) / (tile_height - min_overlap))

    # tiles[y * num_tiles_x + x] is the tile for the y'th row, x'th column.
    tiles: list[Tile] = []

    # Calculate tile coordinates. (Ignore overlap values for now.)
    for tile_idx_y in range(num_tiles_y):
        top = (tile_idx_y * (image_height - tile_height)) // (num_tiles_y - 1) if num_tiles_y > 1 else 0
        bottom = top + tile_height

        for tile_idx_x in range(num_tiles_x):
            left = (tile_idx_x * (image_width - tile_width)) // (num_tiles_x - 1) if num_tiles_x > 1 else 0
            right = left + tile_width

            tile = Tile(
                coords=TBLR(top=top, bottom=bottom, left=left, right=right),
                overlap=TBLR(top=0, bottom=0, left=0, right=0),
            )

            tiles.append(tile)

    return calc_overlap(tiles, num_tiles_x, num_tiles_y)


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

    # Organize tiles into rows.
    tile_and_image_rows: list[list[tuple[Tile, np.ndarray]]] = []
    cur_tile_and_image_row: list[tuple[Tile, np.ndarray]] = []
    first_tile_in_cur_row, _ = tiles_and_images[0]
    for tile_and_image in tiles_and_images:
        tile, _ = tile_and_image
        if not (
            tile.coords.top == first_tile_in_cur_row.coords.top
            and tile.coords.bottom == first_tile_in_cur_row.coords.bottom
        ):
            # Store the previous row, and start a new one.
            tile_and_image_rows.append(cur_tile_and_image_row)
            cur_tile_and_image_row = []
            first_tile_in_cur_row, _ = tile_and_image

        cur_tile_and_image_row.append(tile_and_image)
    tile_and_image_rows.append(cur_tile_and_image_row)

    # Prepare 1D linear gradients for blending.
    gradient_left_x = np.linspace(start=0.0, stop=1.0, num=blend_amount)
    gradient_top_y = np.linspace(start=0.0, stop=1.0, num=blend_amount)
    # Convert shape: (blend_amount, ) -> (blend_amount, 1). The extra dimension enables the gradient to be applied
    # to a 2D image via broadcasting. Note that no additional dimension is needed on gradient_left_x for
    # broadcasting to work correctly.
    gradient_top_y = np.expand_dims(gradient_top_y, axis=1)

    for tile_and_image_row in tile_and_image_rows:
        first_tile_in_row, _ = tile_and_image_row[0]
        row_height = first_tile_in_row.coords.bottom - first_tile_in_row.coords.top
        row_image = np.zeros((row_height, dst_image.shape[1], dst_image.shape[2]), dtype=dst_image.dtype)

        # Blend the tiles in the row horizontally.
        for tile, tile_image in tile_and_image_row:
            # We expect the tiles to be ordered left-to-right. For each tile, we construct a mask that applies linear
            # blending to the left of the current tile. The inverse linear blending is automatically applied to the
            # right of the tiles that have already been pasted by the paste(...) operation.
            tile_height, tile_width, _ = tile_image.shape
            mask = np.ones(shape=(tile_height, tile_width), dtype=np.float64)

            # Left blending:
            if tile.overlap.left > 0:
                assert tile.overlap.left >= blend_amount
                # Center the blending gradient in the middle of the overlap.
                blend_start_left = tile.overlap.left // 2 - blend_amount // 2
                # The region left of the blending region is masked completely.
                mask[:, :blend_start_left] = 0.0
                # Apply the blend gradient to the mask.
                mask[:, blend_start_left : blend_start_left + blend_amount] = gradient_left_x
                # For visual debugging:
                # tile_image[:, blend_start_left : blend_start_left + blend_amount] = 0

            paste(
                dst_image=row_image,
                src_image=tile_image,
                box=TBLR(
                    top=0, bottom=tile.coords.bottom - tile.coords.top, left=tile.coords.left, right=tile.coords.right
                ),
                mask=mask,
            )

        # Blend the row into the dst_image vertically.
        # We construct a mask that applies linear blending to the top of the current row. The inverse linear blending is
        # automatically applied to the bottom of the tiles that have already been pasted by the paste(...) operation.
        mask = np.ones(shape=(row_image.shape[0], row_image.shape[1]), dtype=np.float64)
        # Top blending:
        # (See comments under 'Left blending' for an explanation of the logic.)
        # We assume that the entire row has the same vertical overlaps as the first_tile_in_row.
        if first_tile_in_row.overlap.top > 0:
            assert first_tile_in_row.overlap.top >= blend_amount
            blend_start_top = first_tile_in_row.overlap.top // 2 - blend_amount // 2
            mask[:blend_start_top, :] = 0.0
            mask[blend_start_top : blend_start_top + blend_amount, :] = gradient_top_y
            # For visual debugging:
            # row_image[blend_start_top : blend_start_top + blend_amount, :] = 0
        paste(
            dst_image=dst_image,
            src_image=row_image,
            box=TBLR(
                top=first_tile_in_row.coords.top,
                bottom=first_tile_in_row.coords.bottom,
                left=0,
                right=row_image.shape[1],
            ),
            mask=mask,
        )


def merge_tiles_with_seam_blending(
    dst_image: np.ndarray, tiles: list[Tile], tile_images: list[np.ndarray], blend_amount: int
):
    """Merge a set of image tiles into `dst_image` with seam blending between the tiles.

    We expect every tile edge to either:
    1) have an overlap of 0, because it is aligned with the image edge, or
    2) have an overlap >= blend_amount.
    If neither of these conditions are satisfied, we raise an exception.

    The seam blending is centered on a seam of least energy of the overlap between adjacent tiles.

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

    # Organize tiles into rows.
    tile_and_image_rows: list[list[tuple[Tile, np.ndarray]]] = []
    cur_tile_and_image_row: list[tuple[Tile, np.ndarray]] = []
    first_tile_in_cur_row, _ = tiles_and_images[0]
    for tile_and_image in tiles_and_images:
        tile, _ = tile_and_image
        if not (
            tile.coords.top == first_tile_in_cur_row.coords.top
            and tile.coords.bottom == first_tile_in_cur_row.coords.bottom
        ):
            # Store the previous row, and start a new one.
            tile_and_image_rows.append(cur_tile_and_image_row)
            cur_tile_and_image_row = []
            first_tile_in_cur_row, _ = tile_and_image

        cur_tile_and_image_row.append(tile_and_image)
    tile_and_image_rows.append(cur_tile_and_image_row)

    for tile_and_image_row in tile_and_image_rows:
        first_tile_in_row, _ = tile_and_image_row[0]
        row_height = first_tile_in_row.coords.bottom - first_tile_in_row.coords.top
        row_image = np.zeros((row_height, dst_image.shape[1], dst_image.shape[2]), dtype=dst_image.dtype)

        # Blend the tiles in the row horizontally.
        for tile, tile_image in tile_and_image_row:
            # We expect the tiles to be ordered left-to-right.
            # For each tile:
            # - extract the overlap regions and pass to seam_blend()
            # - apply blended region to the row_image
            # - apply the un-blended region to the row_image
            tile_height, tile_width, _ = tile_image.shape
            overlap_size = tile.overlap.left
            # Left blending:
            if overlap_size > 0:
                assert overlap_size >= blend_amount

                overlap_coord_right = tile.coords.left + overlap_size
                src_overlap = row_image[:, tile.coords.left : overlap_coord_right]
                dst_overlap = tile_image[:, :overlap_size]
                blended_overlap = seam_blend(src_overlap, dst_overlap, blend_amount, x_seam=False)
                row_image[:, tile.coords.left : overlap_coord_right] = blended_overlap
                row_image[:, overlap_coord_right : tile.coords.right] = tile_image[:, overlap_size:]
            else:
                # no overlap just paste the tile
                row_image[:, tile.coords.left : tile.coords.right] = tile_image

        # Blend the row into the dst_image
        # We assume that the entire row has the same vertical overlaps as the first_tile_in_row.
        # Rows are processed in the same way as tiles (extract overlap, blend, apply)
        row_overlap_size = first_tile_in_row.overlap.top
        if row_overlap_size > 0:
            assert row_overlap_size >= blend_amount

            overlap_coords_bottom = first_tile_in_row.coords.top + row_overlap_size
            src_overlap = dst_image[first_tile_in_row.coords.top : overlap_coords_bottom, :]
            dst_overlap = row_image[:row_overlap_size, :]
            blended_overlap = seam_blend(src_overlap, dst_overlap, blend_amount, x_seam=True)
            dst_image[first_tile_in_row.coords.top : overlap_coords_bottom, :] = blended_overlap
            dst_image[overlap_coords_bottom : first_tile_in_row.coords.bottom, :] = row_image[row_overlap_size:, :]
        else:
            # no overlap just paste the row
            dst_image[first_tile_in_row.coords.top : first_tile_in_row.coords.bottom, :] = row_image
