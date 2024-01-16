import numpy as np
import pytest

from invokeai.backend.tiles.tiles import (
    calc_tiles_even_split,
    calc_tiles_min_overlap,
    calc_tiles_with_overlap,
    merge_tiles_with_linear_blending,
)
from invokeai.backend.tiles.utils import TBLR, Tile

####################################
# Test calc_tiles_with_overlap(...)
####################################


def test_calc_tiles_with_overlap_single_tile():
    """Test calc_tiles_with_overlap() behavior when a single tile covers the image."""
    tiles = calc_tiles_with_overlap(image_height=512, image_width=1024, tile_height=512, tile_width=1024, overlap=64)

    expected_tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=1024),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        )
    ]

    assert tiles == expected_tiles


def test_calc_tiles_with_overlap_evenly_divisible():
    """Test calc_tiles_with_overlap() behavior when the image is evenly covered by multiple tiles."""
    # Parameters chosen so that image is evenly covered by 2 rows, 3 columns of tiles.
    tiles = calc_tiles_with_overlap(image_height=576, image_width=1600, tile_height=320, tile_width=576, overlap=64)

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=320, left=0, right=576),
            overlap=TBLR(top=0, bottom=64, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=512, right=1088),
            overlap=TBLR(top=0, bottom=64, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=1024, right=1600),
            overlap=TBLR(top=0, bottom=64, left=64, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=256, bottom=576, left=0, right=576),
            overlap=TBLR(top=64, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=512, right=1088),
            overlap=TBLR(top=64, bottom=0, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=1024, right=1600),
            overlap=TBLR(top=64, bottom=0, left=64, right=0),
        ),
    ]

    assert tiles == expected_tiles


def test_calc_tiles_with_overlap_not_evenly_divisible():
    """Test calc_tiles_with_overlap() behavior when the image requires 'uneven' overlaps to achieve proper coverage."""
    # Parameters chosen so that image is covered by 2 rows and 3 columns of tiles, with uneven overlaps.
    tiles = calc_tiles_with_overlap(image_height=400, image_width=1200, tile_height=256, tile_width=512, overlap=64)

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=256, left=0, right=512),
            overlap=TBLR(top=0, bottom=112, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=256, left=448, right=960),
            overlap=TBLR(top=0, bottom=112, left=64, right=272),
        ),
        Tile(
            coords=TBLR(top=0, bottom=256, left=688, right=1200),
            overlap=TBLR(top=0, bottom=112, left=272, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=144, bottom=400, left=0, right=512),
            overlap=TBLR(top=112, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=144, bottom=400, left=448, right=960),
            overlap=TBLR(top=112, bottom=0, left=64, right=272),
        ),
        Tile(
            coords=TBLR(top=144, bottom=400, left=688, right=1200),
            overlap=TBLR(top=112, bottom=0, left=272, right=0),
        ),
    ]

    assert tiles == expected_tiles


@pytest.mark.parametrize(
    ["image_height", "image_width", "tile_height", "tile_width", "overlap", "raises"],
    [
        (128, 128, 128, 128, 127, False),  # OK
        (128, 128, 128, 128, 0, False),  # OK
        (128, 128, 64, 64, 0, False),  # OK
        (128, 128, 129, 128, 0, True),  # tile_height exceeds image_height.
        (128, 128, 128, 129, 0, True),  # tile_width exceeds image_width.
        (128, 128, 64, 128, 64, True),  # overlap equals tile_height.
        (128, 128, 128, 64, 64, True),  # overlap equals tile_width.
    ],
)
def test_calc_tiles_with_overlap_input_validation(
    image_height: int,
    image_width: int,
    tile_height: int,
    tile_width: int,
    overlap: int,
    raises: bool,
):
    """Test that calc_tiles_with_overlap() raises an exception if the inputs are invalid."""
    if raises:
        with pytest.raises(AssertionError):
            calc_tiles_with_overlap(image_height, image_width, tile_height, tile_width, overlap)
    else:
        calc_tiles_with_overlap(image_height, image_width, tile_height, tile_width, overlap)


####################################
# Test calc_tiles_min_overlap(...)
####################################


def test_calc_tiles_min_overlap_single_tile():
    """Test calc_tiles_min_overlap() behavior when a single tile covers the image."""
    tiles = calc_tiles_min_overlap(
        image_height=512,
        image_width=1024,
        tile_height=512,
        tile_width=1024,
        min_overlap=64,
    )

    expected_tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=1024),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        )
    ]

    assert tiles == expected_tiles


def test_calc_tiles_min_overlap_evenly_divisible():
    """Test calc_tiles_min_overlap() behavior when the image is evenly covered by multiple tiles."""
    # Parameters mimic roughly the same output as the original tile generations of the same test name
    tiles = calc_tiles_min_overlap(
        image_height=576,
        image_width=1600,
        tile_height=320,
        tile_width=576,
        min_overlap=64,
    )

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=320, left=0, right=576),
            overlap=TBLR(top=0, bottom=64, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=512, right=1088),
            overlap=TBLR(top=0, bottom=64, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=1024, right=1600),
            overlap=TBLR(top=0, bottom=64, left=64, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=256, bottom=576, left=0, right=576),
            overlap=TBLR(top=64, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=512, right=1088),
            overlap=TBLR(top=64, bottom=0, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=1024, right=1600),
            overlap=TBLR(top=64, bottom=0, left=64, right=0),
        ),
    ]

    assert tiles == expected_tiles


def test_calc_tiles_min_overlap_not_evenly_divisible():
    """Test calc_tiles_min_overlap() behavior when the image requires 'uneven' overlaps to achieve proper coverage."""
    # Parameters mimic roughly the same output as the original tile generations of the same test name
    tiles = calc_tiles_min_overlap(
        image_height=400,
        image_width=1200,
        tile_height=256,
        tile_width=512,
        min_overlap=64,
    )

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=256, left=0, right=512),
            overlap=TBLR(top=0, bottom=112, left=0, right=168),
        ),
        Tile(
            coords=TBLR(top=0, bottom=256, left=344, right=856),
            overlap=TBLR(top=0, bottom=112, left=168, right=168),
        ),
        Tile(
            coords=TBLR(top=0, bottom=256, left=688, right=1200),
            overlap=TBLR(top=0, bottom=112, left=168, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=144, bottom=400, left=0, right=512),
            overlap=TBLR(top=112, bottom=0, left=0, right=168),
        ),
        Tile(
            coords=TBLR(top=144, bottom=400, left=344, right=856),
            overlap=TBLR(top=112, bottom=0, left=168, right=168),
        ),
        Tile(
            coords=TBLR(top=144, bottom=400, left=688, right=1200),
            overlap=TBLR(top=112, bottom=0, left=168, right=0),
        ),
    ]

    assert tiles == expected_tiles


def test_calc_tiles_min_overlap_tile_bigger_than_image():
    """Test calc_tiles_min_overlap() behavior when the tile is nigger than the image"""
    # Parameters mimic roughly the same output as the original tile generations of the same test name
    tiles = calc_tiles_min_overlap(
        image_height=1024,
        image_width=1024,
        tile_height=1408,
        tile_width=1408,
        min_overlap=128,
    )

    expected_tiles = [
        # single tile
        Tile(
            coords=TBLR(top=0, bottom=1024, left=0, right=1024),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        ),
    ]

    assert tiles == expected_tiles


@pytest.mark.parametrize(
    [
        "image_height",
        "image_width",
        "tile_height",
        "tile_width",
        "min_overlap",
        "raises",
    ],
    [
        (128, 128, 128, 128, 127, False),  # OK
        (128, 128, 128, 128, 0, False),  # OK
        (128, 128, 64, 64, 0, False),  # OK
        (128, 128, 129, 128, 0, False),  # tile_height exceeds image_height defaults to 1 tile.
        (128, 128, 128, 129, 0, False),  # tile_width exceeds image_width defaults to 1 tile.
        (128, 128, 64, 128, 64, True),  # overlap equals tile_height.
        (128, 128, 128, 64, 64, True),  # overlap equals tile_width.
    ],
)
def test_calc_tiles_min_overlap_input_validation(
    image_height: int,
    image_width: int,
    tile_height: int,
    tile_width: int,
    min_overlap: int,
    raises: bool,
):
    """Test that calc_tiles_min_overlap() raises an exception if the inputs are invalid."""
    if raises:
        with pytest.raises(AssertionError):
            calc_tiles_min_overlap(image_height, image_width, tile_height, tile_width, min_overlap)
    else:
        calc_tiles_min_overlap(image_height, image_width, tile_height, tile_width, min_overlap)


####################################
# Test calc_tiles_even_split(...)
####################################


def test_calc_tiles_even_split_single_tile():
    """Test calc_tiles_even_split() behavior when a single tile covers the image."""
    tiles = calc_tiles_even_split(image_height=512, image_width=1024, num_tiles_x=1, num_tiles_y=1, overlap=64)

    expected_tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=1024),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        )
    ]

    assert tiles == expected_tiles


def test_calc_tiles_even_split_evenly_divisible():
    """Test calc_tiles_even_split() behavior when the image is evenly covered by multiple tiles."""
    # Parameters mimic roughly the same output as the original tile generations of the same test name
    tiles = calc_tiles_even_split(image_height=576, image_width=1600, num_tiles_x=3, num_tiles_y=2, overlap=64)

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=320, left=0, right=576),
            overlap=TBLR(top=0, bottom=64, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=512, right=1088),
            overlap=TBLR(top=0, bottom=64, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=320, left=1024, right=1600),
            overlap=TBLR(top=0, bottom=64, left=64, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=256, bottom=576, left=0, right=576),
            overlap=TBLR(top=64, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=512, right=1088),
            overlap=TBLR(top=64, bottom=0, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=256, bottom=576, left=1024, right=1600),
            overlap=TBLR(top=64, bottom=0, left=64, right=0),
        ),
    ]
    assert tiles == expected_tiles


def test_calc_tiles_even_split_not_evenly_divisible():
    """Test calc_tiles_even_split() behavior when the image requires 'uneven' overlaps to achieve proper coverage."""
    # Parameters mimic roughly the same output as the original tile generations of the same test name
    tiles = calc_tiles_even_split(image_height=400, image_width=1200, num_tiles_x=3, num_tiles_y=2, overlap=64)

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=232, left=0, right=440),
            overlap=TBLR(top=0, bottom=64, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=232, left=376, right=816),
            overlap=TBLR(top=0, bottom=64, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=232, left=752, right=1200),
            overlap=TBLR(top=0, bottom=64, left=64, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=168, bottom=400, left=0, right=440),
            overlap=TBLR(top=64, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=168, bottom=400, left=376, right=816),
            overlap=TBLR(top=64, bottom=0, left=64, right=64),
        ),
        Tile(
            coords=TBLR(top=168, bottom=400, left=752, right=1200),
            overlap=TBLR(top=64, bottom=0, left=64, right=0),
        ),
    ]

    assert tiles == expected_tiles


def test_calc_tiles_even_split_difficult_size():
    """Test calc_tiles_even_split() behavior when the image is a difficult size to spilt evenly and keep div8."""
    # Parameters are a difficult size for other tile gen routines to calculate
    tiles = calc_tiles_even_split(image_height=1000, image_width=1000, num_tiles_x=2, num_tiles_y=2, overlap=64)

    expected_tiles = [
        # Row 0
        Tile(
            coords=TBLR(top=0, bottom=528, left=0, right=528),
            overlap=TBLR(top=0, bottom=64, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=528, left=464, right=1000),
            overlap=TBLR(top=0, bottom=64, left=64, right=0),
        ),
        # Row 1
        Tile(
            coords=TBLR(top=464, bottom=1000, left=0, right=528),
            overlap=TBLR(top=64, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=464, bottom=1000, left=464, right=1000),
            overlap=TBLR(top=64, bottom=0, left=64, right=0),
        ),
    ]

    assert tiles == expected_tiles


@pytest.mark.parametrize(
    ["image_height", "image_width", "num_tiles_x", "num_tiles_y", "overlap", "raises"],
    [
        (128, 128, 1, 1, 127, False),  # OK
        (128, 128, 1, 1, 0, False),  # OK
        (128, 128, 2, 2, 0, False),  # OK
        (128, 128, 2, 1, 120, True),  # overlap equals tile_height.
        (128, 128, 1, 2, 120, True),  # overlap equals tile_width.
        (127, 127, 1, 1, 0, True),  # image size must be dividable by 8
    ],
)
def test_calc_tiles_even_split_input_validation(
    image_height: int,
    image_width: int,
    num_tiles_x: int,
    num_tiles_y: int,
    overlap: int,
    raises: bool,
):
    """Test that calc_tiles_even_split() raises an exception if the inputs are invalid."""
    if raises:
        with pytest.raises((AssertionError, ValueError)):
            calc_tiles_even_split(image_height, image_width, num_tiles_x, num_tiles_y, overlap)
    else:
        calc_tiles_even_split(image_height, image_width, num_tiles_x, num_tiles_y, overlap)


#############################################
# Test merge_tiles_with_linear_blending(...)
#############################################


@pytest.mark.parametrize("blend_amount", [0, 32])
def test_merge_tiles_with_linear_blending_horizontal(blend_amount: int):
    """Test merge_tiles_with_linear_blending(...) behavior when merging horizontally."""
    # Initialize 2 tiles side-by-side.
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=512, left=448, right=960),
            overlap=TBLR(top=0, bottom=0, left=64, right=0),
        ),
    ]

    dst_image = np.zeros((512, 960, 3), dtype=np.uint8)

    # Prepare tile_images that match tiles. Pixel values are set based on the tile index.
    tile_images = [
        np.zeros((512, 512, 3)) + 64,
        np.zeros((512, 512, 3)) + 128,
    ]

    # Calculate expected output.
    expected_output = np.zeros((512, 960, 3), dtype=np.uint8)
    expected_output[:, : 480 - (blend_amount // 2), :] = 64
    if blend_amount > 0:
        gradient = np.linspace(start=64, stop=128, num=blend_amount, dtype=np.uint8).reshape((1, blend_amount, 1))
        expected_output[:, 480 - (blend_amount // 2) : 480 + (blend_amount // 2), :] = gradient
    expected_output[:, 480 + (blend_amount // 2) :, :] = 128

    merge_tiles_with_linear_blending(
        dst_image=dst_image,
        tiles=tiles,
        tile_images=tile_images,
        blend_amount=blend_amount,
    )

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


@pytest.mark.parametrize("blend_amount", [0, 32])
def test_merge_tiles_with_linear_blending_vertical(blend_amount: int):
    """Test merge_tiles_with_linear_blending(...) behavior when merging vertically."""
    # Initialize 2 tiles stacked vertically.
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=64, left=0, right=0),
        ),
        Tile(
            coords=TBLR(top=448, bottom=960, left=0, right=512),
            overlap=TBLR(top=64, bottom=0, left=0, right=0),
        ),
    ]

    dst_image = np.zeros((960, 512, 3), dtype=np.uint8)

    # Prepare tile_images that match tiles. Pixel values are set based on the tile index.
    tile_images = [
        np.zeros((512, 512, 3)) + 64,
        np.zeros((512, 512, 3)) + 128,
    ]

    # Calculate expected output.
    expected_output = np.zeros((960, 512, 3), dtype=np.uint8)
    expected_output[: 480 - (blend_amount // 2), :, :] = 64
    if blend_amount > 0:
        gradient = np.linspace(start=64, stop=128, num=blend_amount, dtype=np.uint8).reshape((blend_amount, 1, 1))
        expected_output[480 - (blend_amount // 2) : 480 + (blend_amount // 2), :, :] = gradient
    expected_output[480 + (blend_amount // 2) :, :, :] = 128

    merge_tiles_with_linear_blending(
        dst_image=dst_image,
        tiles=tiles,
        tile_images=tile_images,
        blend_amount=blend_amount,
    )

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


def test_merge_tiles_with_linear_blending_blend_amount_exceeds_vertical_overlap():
    """Test that merge_tiles_with_linear_blending(...) raises an exception if 'blend_amount' exceeds the overlap between
    any vertically adjacent tiles.
    """
    # Initialize 2 tiles stacked vertically.
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=64, left=0, right=0),
        ),
        Tile(
            coords=TBLR(top=448, bottom=960, left=0, right=512),
            overlap=TBLR(top=64, bottom=0, left=0, right=0),
        ),
    ]

    dst_image = np.zeros((960, 512, 3), dtype=np.uint8)

    # Prepare tile_images that match tiles.
    tile_images = [np.zeros((512, 512, 3)), np.zeros((512, 512, 3))]

    # blend_amount=128 exceeds overlap of 64, so should raise exception.
    with pytest.raises(AssertionError):
        merge_tiles_with_linear_blending(dst_image=dst_image, tiles=tiles, tile_images=tile_images, blend_amount=128)


def test_merge_tiles_with_linear_blending_blend_amount_exceeds_horizontal_overlap():
    """Test that merge_tiles_with_linear_blending(...) raises an exception if 'blend_amount' exceeds the overlap between
    any horizontally adjacent tiles.
    """
    # Initialize 2 tiles side-by-side.
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=0, left=0, right=64),
        ),
        Tile(
            coords=TBLR(top=0, bottom=512, left=448, right=960),
            overlap=TBLR(top=0, bottom=0, left=64, right=0),
        ),
    ]

    dst_image = np.zeros((512, 960, 3), dtype=np.uint8)

    # Prepare tile_images that match tiles.
    tile_images = [np.zeros((512, 512, 3)), np.zeros((512, 512, 3))]

    # blend_amount=128 exceeds overlap of 64, so should raise exception.
    with pytest.raises(AssertionError):
        merge_tiles_with_linear_blending(dst_image=dst_image, tiles=tiles, tile_images=tile_images, blend_amount=128)


def test_merge_tiles_with_linear_blending_tiles_overflow_dst_image():
    """Test that merge_tiles_with_linear_blending(...) raises an exception if any of the tiles overflows the
    dst_image.
    """
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        )
    ]

    dst_image = np.zeros((256, 512, 3), dtype=np.uint8)

    # Prepare tile_images that match tiles.
    tile_images = [np.zeros((512, 512, 3))]

    with pytest.raises(ValueError):
        merge_tiles_with_linear_blending(dst_image=dst_image, tiles=tiles, tile_images=tile_images, blend_amount=0)


def test_merge_tiles_with_linear_blending_mismatched_list_lengths():
    """Test that merge_tiles_with_linear_blending(...) raises an exception if the lengths of 'tiles' and 'tile_images'
    do not match.
    """
    tiles = [
        Tile(
            coords=TBLR(top=0, bottom=512, left=0, right=512),
            overlap=TBLR(top=0, bottom=0, left=0, right=0),
        )
    ]

    dst_image = np.zeros((256, 512, 3), dtype=np.uint8)

    # tile_images is longer than tiles, so should cause an exception.
    tile_images = [np.zeros((512, 512, 3)), np.zeros((512, 512, 3))]

    with pytest.raises(ValueError):
        merge_tiles_with_linear_blending(dst_image=dst_image, tiles=tiles, tile_images=tile_images, blend_amount=0)
