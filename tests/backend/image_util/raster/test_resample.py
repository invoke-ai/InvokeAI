import numpy as np
import pytest
from PIL import Image

from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterConversionError,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.resample import MIN_VISIBLE_ALPHA, resize_raster

# The filters the image nodes offer (PIL_RESAMPLING_MAP).
FILTERS = [
    Image.Resampling.NEAREST,
    Image.Resampling.BOX,
    Image.Resampling.BILINEAR,
    Image.Resampling.HAMMING,
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
]
CHANNELS = {"RGB": RasterChannels.RGB, "RGBA": RasterChannels.RGBA, "L": RasterChannels.L, "LA": RasterChannels.LA}


def _descriptor(channels: RasterChannels, **overrides) -> RasterDescriptor:
    has_alpha = channels in (RasterChannels.LA, RasterChannels.RGBA)
    fields = {
        "channels": channels,
        "color": ColorEncoding.SRGB,
        "value_range": ValueRange.UNIT,
        "alpha": AlphaMode.STRAIGHT if has_alpha else AlphaMode.NONE,
        "channel_count": len(channels.value),
    }
    fields.update(overrides)
    return RasterDescriptor(**fields)


def _eight_bit(mode: str, size: tuple[int, int], seed: int) -> np.ndarray:
    """Random 8-bit pixels away from 0 and 255, where Pillow clips its 8-bit intermediate pass and the float result,
    which keeps overshoot, legitimately differs. Alpha is a smooth ramp: a noisy alpha rings, and Pillow's 8-bit clip of
    that ringing (which the float result does not repeat) would dominate the error."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(64, 193, size=(size[1], size[0], len(mode)), dtype=np.uint8)
    if "A" in mode:
        pixels[:, :, -1] = np.linspace(190, 240, size[0]).astype(np.uint8)[np.newaxis, :]
    return pixels


@pytest.mark.parametrize("resample", FILTERS, ids=lambda resample: resample.name)
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("source_size, target_size", [((7, 5), (19, 11)), ((13, 9), (6, 4)), ((5, 7), (5, 3))])
def test_matches_pillows_eight_bit_resize_within_its_rounding_bound(mode, resample, source_size, target_size) -> None:
    pixels = _eight_bit(mode, source_size, seed=11)
    raster = RasterImage.from_array((pixels / np.float32(255)).astype(np.float32), _descriptor(CHANNELS[mode]))

    result = resize_raster(raster, *target_size, resample).view()

    pil = Image.fromarray(pixels[:, :, 0] if len(mode) == 1 else pixels, mode)
    expected = np.asarray(pil.resize(target_size, resample), dtype=np.float32).reshape(result.shape)
    # Pillow rounds its horizontal pass to 8 bits before the vertical one (and with alpha also the premultiplied color),
    # so its own result is only that close to the exact one.
    tolerance = 2.5 if "A" in mode else 1.5
    assert np.abs(np.clip(result, 0, 1) * 255 - expected).max() <= tolerance


@pytest.mark.parametrize("resample", FILTERS, ids=lambda resample: resample.name)
def test_a_same_size_resize_returns_the_input(resample) -> None:
    pixels = np.random.default_rng(1).random((4, 6, 4), dtype=np.float32)
    raster = RasterImage.from_array(pixels, _descriptor(RasterChannels.RGBA))

    assert resize_raster(raster, 6, 4, resample) is raster


@pytest.mark.parametrize("resample", FILTERS, ids=lambda resample: resample.name)
def test_values_beyond_the_unit_range_survive(resample) -> None:
    constant = np.full((6, 8, 3), 1.75, np.float32)
    constant[:, :, 1] = -0.5
    result = resize_raster(RasterImage.from_array(constant, _descriptor(RasterChannels.RGB)), 11, 5, resample).view()

    np.testing.assert_allclose(result[:, :, 0], 1.75, atol=1e-5)
    np.testing.assert_allclose(result[:, :, 1], -0.5, atol=1e-5)

    # A hard black/white edge makes the bicubic and Lanczos kernels ring beyond the source range.
    edge = np.zeros((4, 8, 1), np.float32)
    edge[:, 4:] = 1.0
    rung = resize_raster(RasterImage.from_array(edge, _descriptor(RasterChannels.L)), 16, 4, Image.Resampling.BICUBIC)
    assert rung.view().max() > 1.0 and rung.view().min() < 0.0


@pytest.mark.parametrize("resample", [r for r in FILTERS if r != Image.Resampling.NEAREST], ids=lambda r: r.name)
@pytest.mark.parametrize("size", [(16, 6), (4, 2), (9, 7)])
def test_hidden_color_under_transparency_does_not_bleed(resample, size) -> None:
    pixels = np.zeros((6, 8, 4), np.float32)
    pixels[:, :4] = (1.0, 0.0, 0.0, 1.0)  # opaque red on the left
    pixels[:, 4:] = (0.0, 1.0, 1.0, 0.0)  # the right is transparent but carries cyan

    result = resize_raster(RasterImage.from_array(pixels, _descriptor(RasterChannels.RGBA)), *size, resample).view()

    color, alpha = result[:, :, :3], result[:, :, 3]
    assert color[:, :, 1:].max() == 0.0  # no cyan anywhere
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0
    assert color.min() >= -1e-5 and color.max() <= 1.0 + 1e-5
    assert (color[alpha < MIN_VISIBLE_ALPHA] == 0).all()
    # Red stays full strength wherever there is any coverage, instead of being darkened by the transparent side.
    np.testing.assert_allclose(color[:, :, 0][alpha >= MIN_VISIBLE_ALPHA], 1.0, atol=1e-5)
    assert (alpha < MIN_VISIBLE_ALPHA).any() and (alpha > 0.99).any()


def test_a_fully_transparent_image_stays_transparent_black() -> None:
    pixels = np.zeros((4, 4, 4), np.float32)
    pixels[:, :, :3] = 0.8

    result = resize_raster(
        RasterImage.from_array(pixels, _descriptor(RasterChannels.RGBA)), 9, 9, Image.Resampling.LANCZOS
    ).view()

    np.testing.assert_array_equal(result, np.zeros((9, 9, 4), np.float32))


def test_nearest_ignores_alpha_like_pillow() -> None:
    pixels = np.zeros((2, 2, 4), np.float32)
    pixels[:, :, :3] = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]] * 2
    pixels[:, :, 3] = 0.0

    result = resize_raster(
        RasterImage.from_array(pixels, _descriptor(RasterChannels.RGBA)), 4, 4, Image.Resampling.NEAREST
    ).view()

    np.testing.assert_array_equal(result[0], np.repeat(pixels[0], 2, axis=0))


def test_a_non_contiguous_borrowed_raster_resizes_like_its_copy() -> None:
    base = np.random.default_rng(5).random((8, 12, 3), dtype=np.float32)
    strided = base[::2, ::-1]  # neither row- nor column-contiguous
    assert not strided.flags.c_contiguous
    descriptor = _descriptor(RasterChannels.RGB)

    from_view = resize_raster(RasterImage.borrow(strided, descriptor), 7, 5, Image.Resampling.LANCZOS)
    from_copy = resize_raster(RasterImage.from_array(strided, descriptor), 7, 5, Image.Resampling.LANCZOS)

    np.testing.assert_array_equal(from_view.view(), from_copy.view())


def test_the_result_keeps_the_descriptor_including_the_icc_profile() -> None:
    descriptor = _descriptor(RasterChannels.RGB, icc_profile=b"profile")
    raster = RasterImage.from_array(np.zeros((3, 3, 3), np.float32), descriptor)

    assert resize_raster(raster, 5, 5, Image.Resampling.BILINEAR).descriptor == descriptor


@pytest.mark.parametrize(
    "raster",
    [
        RasterImage.from_array(np.zeros((3, 3, 3), np.uint16), _descriptor(RasterChannels.RGB)),
        RasterImage.from_array(np.zeros((3, 3, 3), np.float16), _descriptor(RasterChannels.RGB)),
        RasterImage.from_array(
            np.zeros((3, 3, 4), np.float32), _descriptor(RasterChannels.RGBA, alpha=AlphaMode.PREMULTIPLIED)
        ),
        RasterImage.from_array(
            np.zeros((3, 3, 3), np.float32), _descriptor(RasterChannels.RGB, value_range=ValueRange.UNBOUNDED)
        ),
    ],
    ids=["uint16", "float16", "premultiplied", "unbounded"],
)
def test_rasters_it_cannot_resample_correctly_are_refused(raster) -> None:
    with pytest.raises(RasterConversionError):
        resize_raster(raster, 6, 6, Image.Resampling.BICUBIC)


@pytest.mark.parametrize(
    ("resample", "size"),
    [(Image.Resampling.LANCZOS, (16, 4)), (Image.Resampling.LANCZOS, (60, 4)), (Image.Resampling.BICUBIC, (60, 4))],
    ids=["lanczos-down", "lanczos-up", "bicubic-up"],
)
def test_a_soft_low_alpha_fringe_stays_bounded_and_below_the_visibility_threshold_is_black(resample, size) -> None:
    pixels = np.zeros((4, 24, 4), np.float32)
    pixels[:, :12, :3] = (np.arange(12) % 2)[np.newaxis, :, np.newaxis]  # opaque, alternating black and white columns
    pixels[:, :12, 3] = 1.0
    pixels[:, 12:15, :3] = 0.5  # a nearly invisible fringe
    pixels[:, 12:15, 3] = 0.02
    pixels[:, 15:, :3] = 0.5

    result = resize_raster(RasterImage.from_array(pixels, _descriptor(RasterChannels.RGBA)), *size, resample).view()

    color, alpha = result[:, :, :3], result[:, :, 3]
    # The source color spans 0-1, so ringing may reach one span beyond it on either side, and no further.
    assert color.min() >= -1.0 and color.max() <= 2.0
    barely_visible = (alpha > 0) & (alpha < MIN_VISIBLE_ALPHA)
    assert barely_visible.any()
    assert (color[barely_visible] == 0).all()
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0


@pytest.mark.parametrize("resample", [Image.Resampling.LANCZOS, Image.Resampling.BICUBIC], ids=lambda r: r.name)
@pytest.mark.parametrize(("with_alpha", "without_alpha"), [("RGBA", "RGB"), ("LA", "L")])
def test_a_fully_opaque_image_resizes_like_its_color_alone_with_alpha_staying_one(
    resample, with_alpha, without_alpha
) -> None:
    color_count = len(without_alpha)
    pixels = np.random.default_rng(9).random((6, 8, color_count + 1), dtype=np.float32)  # noisy, so the filters ring
    pixels[:, :, -1] = 1.0

    result = resize_raster(RasterImage.from_array(pixels, _descriptor(CHANNELS[with_alpha])), 13, 9, resample).view()
    color_only = resize_raster(
        RasterImage.from_array(pixels[:, :, :color_count], _descriptor(CHANNELS[without_alpha])), 13, 9, resample
    ).view()

    np.testing.assert_array_equal(result[:, :, -1], np.ones((9, 13), np.float32))
    np.testing.assert_array_equal(result[:, :, :color_count], color_only)
    assert color_only.max() > 1.0 or color_only.min() < 0.0  # the filter really rang
