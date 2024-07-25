import math

import numpy as np
from PIL import Image


def map_seamless_tiles(seamless_tile: Image.Image, target_hw: tuple[int, int], num_repeats_h: float) -> Image.Image:
    """Map seamless tiles to a target size with a given number of repeats along the height dimension."""
    # TODO(ryand): Add option to flip odd rows and columns if the tile is not seamless.
    # - May also want the option to decide on a per-axis basis.

    target_h, target_w = target_hw

    # Calculate the height of the tile that is necessary to achieve the desired number of repeats.
    # Take the ceiling so that the last tile overflows the target height.
    target_tile_h = math.ceil(target_h / num_repeats_h)

    # Resize the tile to the target height.
    # Determine the target_tile_w that preserves the original aspect ratio.
    target_tile_w = int(target_tile_h / seamless_tile.height * seamless_tile.width)
    resized_tile = seamless_tile.resize((target_tile_w, target_tile_h))

    # Repeat the tile along the height and width dimensions.
    num_repeats_h_int = math.ceil(num_repeats_h)
    num_repeats_w_int = math.ceil(target_w / target_tile_w)
    seamless_tiles_np = np.array(resized_tile)
    repeated_tiles_np = np.tile(seamless_tiles_np, (num_repeats_h_int, num_repeats_w_int, 1))

    # Crop the repeated tiles to the target size.
    output_pattern = Image.fromarray(repeated_tiles_np[:target_h, :target_w])
    return output_pattern
