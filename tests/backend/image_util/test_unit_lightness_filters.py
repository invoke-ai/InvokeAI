"""Clarity, texture and contrast dehaze on float32 unit pixels (the path precise rasters take).

The 8-bit PIL functions are the reference: on pixels an 8-bit image can hold, the float path must land within the
legacy quantization of their result. What only the float path can do (keep sub-8-bit detail, values beyond 0-1,
exact alpha) is tested against what the data say, not against the legacy output.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from invokeai.backend.image_util.clarity import clarity, clarity_unit
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze, contrast_dehaze_unit
from invokeai.backend.image_util.texture import texture, texture_unit

FILTERS = {
    "clarity": (clarity, clarity_unit, {"amount": 100, "size": 12}),
    "texture": (texture, texture_unit, {"amount": 100, "size": 8}),
    "contrast_dehaze": (contrast_dehaze, contrast_dehaze_unit, {"amount": 100}),
}
MODES = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}


@pytest.fixture(params=sorted(FILTERS))
def filter_pair(request):
    legacy, unit, kwargs = FILTERS[request.param]
    return (lambda image: legacy(image, **kwargs)), (lambda pixels: unit(pixels, **kwargs))


def _textured_levels(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Smooth ramp plus fine and coarse waves plus noise, as 8-bit levels in float (not yet rounded)."""
    y, x = np.indices(shape)
    noise = np.random.default_rng(seed).normal(0, 4, shape)
    wave = 18 * np.sin(2 * np.pi * x / 10) + 12 * np.sin(2 * np.pi * y / 37)
    return np.clip(40 + 170 * x / shape[1] + wave + noise, 0, 255)


def _eight_bit_pixels(mode: str, seed: int = 0) -> np.ndarray:
    """uint8 HxWxC pixels of a textured image; RGBA/LA get a partly transparent corner and varied alpha."""
    shape = (72, 80)
    count = MODES[mode]
    color = np.stack([_textured_levels(shape, seed + c) * (1 - 0.2 * c) for c in range(min(count, 3))], axis=-1)
    if count <= 2:
        color = color[:, :, :1]
    pixels = np.rint(color).astype(np.uint8)
    if count in (2, 4):
        alpha = np.full(shape, 255, np.uint8)
        alpha[:, :20] = 120
        alpha[:12, :12] = 0
        pixels = np.concatenate([pixels, alpha[:, :, None]], axis=-1)
    return pixels


def _pil(pixels: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(pixels[:, :, 0] if pixels.shape[2] == 1 else pixels, mode=mode)


def _as_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image)
    return array[:, :, None] if array.ndim == 2 else array


@pytest.mark.parametrize("mode", sorted(MODES))
def test_float_path_lands_within_the_legacy_quantization(filter_pair, mode: str) -> None:
    legacy, unit = filter_pair
    pixels = _eight_bit_pixels(mode)

    expected = _as_array(legacy(_pil(pixels, mode))).astype(np.float64)
    result = unit(pixels.astype(np.float32) / 255).astype(np.float64) * 255

    assert result.shape == expected.shape
    # The legacy result is the float result rounded, so the two differ by at most half a level (plus the float32 vs
    # float64 input noise that can flip a rounding tie) and the rounded float result is the legacy one almost always.
    assert np.abs(result - expected).max() < 0.51
    assert np.mean(np.rint(result) == expected) > 0.99
    assert np.abs(expected - pixels).max() > 3  # the filter really changed the image


@pytest.mark.parametrize("name", ["clarity", "texture"])
def test_constant_images_and_gamut_corners_stay_put(name: str) -> None:
    """Detail filters have nothing to change on a flat image. (Contrast dehaze moves flat tones by definition.)"""
    _, unit, kwargs = FILTERS[name]
    for rgb in [(0.3, 0.6, 0.9), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]:
        pixels = np.full((40, 48, 3), rgb, np.float32)

        np.testing.assert_allclose(unit(pixels, **kwargs), pixels, atol=2e-5)


def test_amount_zero_returns_the_pixels_unchanged_including_overshoot() -> None:
    pixels = np.random.default_rng(1).uniform(-0.5, 1.7, (20, 24, 3)).astype(np.float32)
    for _, unit, kwargs in FILTERS.values():
        result = unit(pixels, **{**kwargs, "amount": 0})

        assert result is not pixels
        np.testing.assert_array_equal(result, pixels)


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
def test_alpha_and_hidden_pixels_pass_through_exactly(filter_pair, mode: str) -> None:
    _, unit = filter_pair
    pixels = _eight_bit_pixels(mode).astype(np.float32) / 255
    alpha = np.random.default_rng(2).uniform(0.05, 1.0, pixels.shape[:2]).astype(np.float32)
    alpha[:12, :12] = 0.0
    pixels[:, :, -1] = alpha
    pixels[:12, :12, :-1] = 1.75  # hidden color beyond 0-1 must come back untouched
    before = pixels.copy()

    result = unit(pixels)

    np.testing.assert_array_equal(pixels, before)  # the input is not modified
    np.testing.assert_array_equal(result[:, :, -1], alpha)
    np.testing.assert_array_equal(result[:12, :12], before[:12, :12])
    assert np.abs(result[12:, 20:, :-1] - before[12:, 20:, :-1]).max() > 0.01


def test_hidden_pixels_do_not_bleed_into_visible_neighbours() -> None:
    """A hidden black block inside a visible gray field must not change the filter's view of that field."""
    opaque = np.full((60, 60, 4), 0.5, np.float32)
    with_hole = opaque.copy()
    with_hole[20:40, 20:40] = (0.0, 0.0, 0.0, 0.0)
    visible = with_hole[:, :, 3] > 0

    for _, unit, kwargs in FILTERS.values():
        np.testing.assert_allclose(
            unit(with_hole, **kwargs)[visible][:, :3], unit(opaque, **kwargs)[visible][:, :3], atol=2e-5
        )


def test_changed_pixels_are_gamut_mapped_and_finite_for_overshooting_input(filter_pair) -> None:
    _, unit = filter_pair
    pixels = np.clip(_textured_levels((64, 64), 5) / 255, 0, 1)[:, :, None].repeat(3, axis=2).astype(np.float32)
    pixels *= 1.6
    pixels[5, 5] = (-0.3, 0.2, 1.4)

    result = unit(pixels)

    assert np.isfinite(result).all()
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_detail_below_one_8_bit_level_survives_in_the_float_path() -> None:
    _, x = np.indices((96, 96))
    wave = (0.0008 * np.sin(2 * np.pi * x / 12)).astype(np.float32)  # 0.2 of an 8-bit level
    pixels = (128 / 255 + wave)[:, :, None].repeat(3, axis=2).astype(np.float32)
    inner = (slice(24, -24), slice(24, -24))

    result = clarity_unit(pixels, amount=100, size=12)

    assert np.abs(result * 255 - np.rint(result * 255)).max() > 0.1  # values lie between 8-bit levels
    assert result[inner].std() > 1.15 * pixels[inner].std()
    # the legacy path cannot see this detail: the 8-bit image is flat
    assert np.unique(np.rint(pixels * 255)).size == 1


def test_gray_stays_gray_and_matches_rgb_gray(filter_pair) -> None:
    _, unit = filter_pair
    gray = (_textured_levels((64, 64), 7) / 255).astype(np.float32)

    as_gray = unit(gray[:, :, None])
    as_rgb = unit(np.repeat(gray[:, :, None], 3, axis=2))

    np.testing.assert_allclose(as_gray[:, :, 0], as_rgb.mean(axis=2), atol=1e-4)
    assert (as_rgb.max(axis=2) - as_rgb.min(axis=2)).max() < 1e-4


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_pixels_are_rejected(filter_pair, bad: float) -> None:
    _, unit = filter_pair
    pixels = np.full((16, 16, 3), 0.5, np.float32)
    pixels[3, 4, 1] = bad

    with pytest.raises(ValueError):
        unit(pixels)


def test_unit_functions_validate_arguments_and_layout() -> None:
    good = np.full((8, 8, 3), 0.5, np.float32)
    with pytest.raises(ValueError):
        clarity_unit(good, amount=201)
    with pytest.raises(ValueError):
        texture_unit(good, amount=10, size=0.5)
    with pytest.raises(ValueError):
        contrast_dehaze_unit(good, amount=float("nan"))
    with pytest.raises(ValueError):
        clarity_unit(good.astype(np.float64), amount=10)
    with pytest.raises(ValueError):
        clarity_unit(good[:, :, 0], amount=10)
    with pytest.raises(ValueError):
        clarity_unit(np.zeros((8, 8, 5), np.float32), amount=10)


def test_alpha_below_half_an_8_bit_step_counts_as_transparent(filter_pair) -> None:
    """Precise resizes leave alpha ~1e-4 with color 0 at cutout edges; that color must not spread into the fill."""
    _, unit = filter_pair
    pixels = _eight_bit_pixels("RGBA").astype(np.float32) / 255
    pixels[:, :, 3] = 1.0
    fringe = np.zeros(pixels.shape[:2], bool)
    fringe[30:40, 30:40] = True
    pixels[fringe] = (0.0, 0.0, 0.0, 1e-4)
    zeroed = pixels.copy()
    zeroed[fringe, 3] = 0.0

    result = unit(pixels)
    reference = unit(zeroed)

    np.testing.assert_array_equal(result[~fringe][:, :3], reference[~fringe][:, :3])
    np.testing.assert_array_equal(result[fringe], pixels[fringe])
