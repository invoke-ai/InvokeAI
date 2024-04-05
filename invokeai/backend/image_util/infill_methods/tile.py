from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


def create_tile_pool(img_array: np.ndarray, tile_size: tuple[int, int]) -> list[np.ndarray]:
    """
    Create a pool of tiles from non-transparent areas of the image by systematically walking through the image.

    Args:
        img_array: numpy array of the image.
        tile_size: tuple (tile_width, tile_height) specifying the size of each tile.

    Returns:
        A list of numpy arrays, each representing a tile.
    """
    tiles: list[np.ndarray] = []
    rows, cols = img_array.shape[:2]
    tile_width, tile_height = tile_size

    for y in range(0, rows - tile_height + 1, tile_height):
        for x in range(0, cols - tile_width + 1, tile_width):
            tile = img_array[y : y + tile_height, x : x + tile_width]
            # Check if the image has an alpha channel and the tile is completely opaque
            if img_array.shape[2] == 4 and np.all(tile[:, :, 3] == 255):
                tiles.append(tile)
            elif img_array.shape[2] == 3:  # If no alpha channel, append the tile
                tiles.append(tile)

    if not tiles:
        raise ValueError(
            "Not enough opaque pixels to generate any tiles. Use a smaller tile size or a different image."
        )

    return tiles


def create_filled_image(
    img_array: np.ndarray, tile_pool: list[np.ndarray], tile_size: tuple[int, int], seed: int
) -> np.ndarray:
    """
    Create an image of the same dimensions as the original, filled entirely with tiles from the pool.

    Args:
        img_array: numpy array of the original image.
        tile_pool: A list of numpy arrays, each representing a tile.
        tile_size: tuple (tile_width, tile_height) specifying the size of each tile.

    Returns:
        A numpy array representing the filled image.
    """

    rows, cols, _ = img_array.shape
    tile_width, tile_height = tile_size

    # Prep an empty RGB image
    filled_img_array = np.zeros((rows, cols, 3), dtype=img_array.dtype)

    # Make the random tile selection reproducible
    rng = np.random.default_rng(seed)

    for y in range(0, rows, tile_height):
        for x in range(0, cols, tile_width):
            # Pick a random tile from the pool
            tile = tile_pool[rng.integers(len(tile_pool))]

            # Calculate the space available (may be less than tile size near the edges)
            space_y = min(tile_height, rows - y)
            space_x = min(tile_width, cols - x)

            # Crop the tile if necessary to fit into the available space
            cropped_tile = tile[:space_y, :space_x, :3]

            # Fill the available space with the (possibly cropped) tile
            filled_img_array[y : y + space_y, x : x + space_x, :3] = cropped_tile

    return filled_img_array


@dataclass
class InfillTileOutput:
    infilled: Image.Image
    tile_image: Optional[Image.Image] = None


def infill_tile(image_to_infill: Image.Image, seed: int, tile_size: int) -> InfillTileOutput:
    """Infills an image with random tiles from the image itself.

    If the image is not an RGBA image, it is returned untouched.

    Args:
        image: The image to infill.
        tile_size: The size of the tiles to use for infilling.

    Raises:
        ValueError: If there are not enough opaque pixels to generate any tiles.
    """

    if image_to_infill.mode != "RGBA":
        return InfillTileOutput(infilled=image_to_infill)

    # Internally, we want a tuple of (tile_width, tile_height). In the future, the tile size can be any rectangle.
    _tile_size = (tile_size, tile_size)
    np_image = np.array(image_to_infill, dtype=np.uint8)

    # Create the pool of tiles that we will use to infill
    tile_pool = create_tile_pool(np_image, _tile_size)

    # Create an image from the tiles, same size as the original
    tile_np_image = create_filled_image(np_image, tile_pool, _tile_size, seed)

    # Paste the OG image over the tile image, effectively infilling the area
    tile_image = Image.fromarray(tile_np_image, "RGB")
    infilled = tile_image.copy()
    infilled.paste(image_to_infill, (0, 0), image_to_infill.split()[-1])

    # I think we want this to be "RGBA"?
    infilled.convert("RGBA")

    return InfillTileOutput(infilled=infilled, tile_image=tile_image)
