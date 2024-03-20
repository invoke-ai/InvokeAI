from typing import Tuple

import numpy as np
from PIL import Image


def infill_mosaic(
    image: Image.Image,
    tile_shape: Tuple[int, int] = (64, 16),
    min_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    max_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
) -> Image.Image:
    """
    image:PIL - A PIL Image
    tile_shape: Tuple[int,int] - Tile width & Tile Height
    min_color: Tuple[int,int,int] - RGB values for the lowest color to clip to (0-255)
    max_color: Tuple[int,int,int] - RGB values for the highest color to clip to (0-255)
    """

    np_image = np.array(image)  # Convert image to np array
    alpha = np_image[:, :, 3]  # Get the mask from the alpha channel of the image
    non_transparent_pixels = np_image[alpha != 0, :3]  # List of non-transparent pixels

    # Create color tiles to paste in the empty areas of the image
    tile_width, tile_height = tile_shape

    # Clip the range of colors in the image to a particular spectrum only
    r_min, g_min, b_min, _ = min_color
    r_max, g_max, b_max, _ = max_color
    non_transparent_pixels[:, 0] = np.clip(non_transparent_pixels[:, 0], r_min, r_max)
    non_transparent_pixels[:, 1] = np.clip(non_transparent_pixels[:, 1], g_min, g_max)
    non_transparent_pixels[:, 2] = np.clip(non_transparent_pixels[:, 2], b_min, b_max)

    tiles = []
    for _ in range(256):
        color = non_transparent_pixels[np.random.randint(len(non_transparent_pixels))]

        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        tile[:, :] = color
        tiles.append(tile)

    # Fill the transparent area with tiles
    filled_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for x in range(image.width):
        for y in range(image.height):
            tile = tiles[np.random.randint(len(tiles))]
            filled_image[
                y - (y % tile_height) : y - (y % tile_height) + tile_height,
                x - (x % tile_width) : x - (x % tile_width) + tile_width,
            ] = tile

    filled_image = Image.fromarray(filled_image)  # Convert the filled tiles image to PIL
    image = Image.composite(
        image, filled_image, image.split()[-1]
    )  # Composite the original image on top of the filled tiles
    return image
