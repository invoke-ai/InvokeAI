import itertools

import numpy as np
import pytest
from PIL import Image

from invokeai.backend.image_util.raster import AlphaMode, RasterChannels
from invokeai.backend.image_util.raster.composite import (
    alpha_composite,
    composite_layer,
    convert_channels,
    paste_blend,
    unit_srgb_raster,
)

MODES = {"L": RasterChannels.L, "LA": RasterChannels.LA, "RGB": RasterChannels.RGB, "RGBA": RasterChannels.RGBA}


def _eight_bit(mode: str, size: tuple[int, int], seed: int) -> np.ndarray:
    pixels = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], len(mode)), dtype=np.uint8)
    return pixels


def _pil(pixels: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(pixels[:, :, 0] if pixels.shape[2] == 1 else pixels, mode)


def _unit(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float32) / np.float32(255)


@pytest.mark.parametrize(("source", "target"), list(itertools.product(MODES, MODES)))
def test_convert_channels_matches_pillows_convert(source, target) -> None:
    pixels = _eight_bit(source, (5, 4), seed=1)

    result = convert_channels(_unit(pixels), MODES[source], MODES[target])

    expected = np.asarray(_pil(pixels, source).convert(target)).reshape(result.shape)
    assert np.abs(result * 255 - expected).max() <= 1.0  # Pillow rounds its integer luma


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("offset", [(0, 0), (2, 1), (-2, -1), (5, 4), (-9, 0)])
@pytest.mark.parametrize("masked", [False, True])
def test_paste_blend_matches_pillow_paste_including_clipping(mode, offset, masked) -> None:
    destination, source = _eight_bit(mode, (7, 6), seed=2), _eight_bit(mode, (4, 3), seed=3)
    coverage = _eight_bit("L", (4, 3), seed=4)[:, :, 0]
    expected = _pil(destination, mode)
    expected.paste(_pil(source, mode), offset, _pil(coverage[:, :, None], "L") if masked else None)

    result = paste_blend(_unit(destination), _unit(source), *offset, _unit(coverage) if masked else None)

    assert np.abs(result * 255 - np.asarray(expected).reshape(result.shape)).max() <= 1.0


def test_paste_blend_leaves_the_destination_alone_and_keeps_values_beyond_unit() -> None:
    destination = np.full((2, 2, 3), 2.0, np.float32)
    source = np.full((2, 2, 3), -1.0, np.float32)

    full = paste_blend(destination, source, 0, 0, np.ones((2, 2), np.float32))
    none = paste_blend(destination, source, 0, 0, np.zeros((2, 2), np.float32))
    half = paste_blend(destination, source, 0, 0, np.full((2, 2), 0.5, np.float32))

    np.testing.assert_array_equal(full, source)
    np.testing.assert_array_equal(none, destination)
    np.testing.assert_array_equal(half, np.full((2, 2, 3), 0.5, np.float32))
    assert (destination == 2.0).all()  # the input was not modified


@pytest.mark.parametrize("seed", [5, 6])
def test_alpha_composite_matches_pillow_within_its_rounding(seed) -> None:
    base, top = _eight_bit("RGBA", (6, 5), seed), _eight_bit("RGBA", (6, 5), seed + 10)
    top[0, :, 3] = 0  # fully transparent top
    base[1, :, 3] = 0  # fully transparent base
    expected = np.asarray(Image.alpha_composite(_pil(base, "RGBA"), _pil(top, "RGBA")))

    result = alpha_composite(_unit(base), _unit(top))

    np.testing.assert_allclose(result[:, :, 3] * 255, expected[:, :, 3], atol=1.0)
    # Color is compared where it is visible: Pillow's integer weights are coarse at very small alpha.
    visible = expected[:, :, 3] > 64
    assert np.abs(result[:, :, :3] * 255 - expected[:, :, :3])[visible].max() <= 2.0


def test_alpha_composite_keeps_the_base_where_the_top_is_transparent_and_overshoot_elsewhere() -> None:
    base = np.zeros((1, 3, 4), np.float32)
    base[0, 0] = (0.9, 0.8, 0.7, 0.0)  # hidden color
    base[0, 1] = (0.1, 0.2, 0.3, 1.0)
    top = np.zeros((1, 3, 4), np.float32)
    top[0, 1] = (1.5, -0.5, 0.0, 1.0)
    top[0, 2] = (1.5, 1.5, 1.5, 0.5)

    result = alpha_composite(base, top)

    np.testing.assert_array_equal(result[0, 0], base[0, 0])
    np.testing.assert_array_equal(result[0, 1], top[0, 1])
    np.testing.assert_allclose(result[0, 2], (1.5, 1.5, 1.5, 0.5))  # over a transparent base: the top as it is


def test_unit_srgb_raster_describes_the_channels() -> None:
    for count, channels in enumerate(MODES.values(), start=1):
        raster = unit_srgb_raster(np.zeros((2, 3, count), np.float32))

        assert raster.descriptor.channels is channels
        assert raster.descriptor.alpha is (AlphaMode.STRAIGHT if "A" in channels.value else AlphaMode.NONE)
        assert (raster.width, raster.height) == (3, 2)


def test_unit_srgb_raster_adopts_fresh_arrays_and_copies_views() -> None:
    fresh = np.zeros((3, 3, 4), np.float32)
    canvas = np.zeros((4, 4, 4), np.float32)

    adopted = unit_srgb_raster(fresh)
    cropped = unit_srgb_raster(canvas[1:3, 1:3])

    assert np.shares_memory(adopted.view(), fresh)
    assert not np.shares_memory(cropped.view(), canvas)
    assert canvas.flags.writeable  # a view of the canvas was not frozen


def test_paste_blend_in_place_writes_the_destination_and_matches_the_copying_form() -> None:
    destination = np.random.default_rng(7).random((9, 11, 4), dtype=np.float32)
    source = np.random.default_rng(8).random((4, 5, 4), dtype=np.float32)
    coverage = np.random.default_rng(9).random((4, 5), dtype=np.float32)
    expected = paste_blend(destination, source, -1, 3, coverage)

    result = paste_blend(destination, source, -1, 3, coverage, in_place=True)

    assert result is destination
    np.testing.assert_array_equal(destination, expected)


def test_composite_layer_matches_pasting_a_scaled_layer_then_alpha_compositing() -> None:
    base = np.random.default_rng(10).random((9, 11, 4), dtype=np.float32)
    layer = np.random.default_rng(11).random((4, 5, 4), dtype=np.float32)
    coverage = np.random.default_rng(12).random((4, 5), dtype=np.float32)
    # Row blocks of one row, so the blocking is exercised on a small image.
    expected = alpha_composite(base, paste_blend(np.zeros_like(base), layer, 2, 3, coverage))

    canvas = base.copy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("invokeai.backend.image_util.raster.composite._BLOCK_SAMPLES", 1)
        composite_layer(canvas, layer, 2, 3, coverage)

    np.testing.assert_allclose(canvas, expected, atol=1e-6)


def test_a_sub_visible_top_alpha_keeps_the_base_including_its_hidden_color() -> None:
    base = np.zeros((1, 2, 4), np.float32)
    base[:, :, :3] = 0.9  # hidden under alpha 0
    top = np.full((1, 2, 4), (0.1, 0.2, 0.3, 0.001), np.float32)  # alpha below half an 8-bit step

    np.testing.assert_array_equal(alpha_composite(base, top), base)
