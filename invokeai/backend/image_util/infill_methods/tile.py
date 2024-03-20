import math
from typing import Optional

import numpy as np
from PIL import Image


def get_tile_images(image: np.ndarray, width: int = 8, height: int = 8):
    _nrows, _ncols, depth = image.shape
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False,
    )


def infill_tile(im: Image.Image, tile_size: int = 16, seed: Optional[int] = None) -> Image.Image:
    # Only fill if there's an alpha layer
    if im.mode != "RGBA":
        return im

    a = np.asarray(im, dtype=np.uint8)

    tile_size_tuple = (tile_size, tile_size)

    # Get the image as tiles of a specified size
    tiles = get_tile_images(a, *tile_size_tuple).copy()

    # Get the mask as tiles
    tiles_mask = tiles[:, :, :, :, 3]

    # Find any mask tiles with any fully transparent pixels (we will be replacing these later)
    tmask_shape = tiles_mask.shape
    tiles_mask = tiles_mask.reshape(math.prod(tiles_mask.shape))
    n, ny = (math.prod(tmask_shape[0:2])), math.prod(tmask_shape[2:])
    tiles_mask = tiles_mask > 0
    tiles_mask = tiles_mask.reshape((n, ny)).all(axis=1)

    # Get RGB tiles in single array and filter by the mask
    tshape = tiles.shape
    tiles_all = tiles.reshape((math.prod(tiles.shape[0:2]), *tiles.shape[2:]))
    filtered_tiles = tiles_all[tiles_mask]

    if len(filtered_tiles) == 0:
        return im

    # Find all invalid tiles and replace with a random valid tile
    replace_count = (tiles_mask == False).sum()  # noqa: E712
    rng = np.random.default_rng(seed=seed)
    tiles_all[np.logical_not(tiles_mask)] = filtered_tiles[rng.choice(filtered_tiles.shape[0], replace_count), :, :, :]

    # Convert back to an image
    tiles_all = tiles_all.reshape(tshape)
    tiles_all = tiles_all.swapaxes(1, 2)
    st = tiles_all.reshape(
        (
            math.prod(tiles_all.shape[0:2]),
            math.prod(tiles_all.shape[2:4]),
            tiles_all.shape[4],
        )
    )
    si = Image.fromarray(st, mode="RGBA")

    return si
