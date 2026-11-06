from fractions import Fraction

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterConversionError,
    RasterDescriptor,
    RasterImage,
    ValueRange,
    quantize_unit,
    raster_from_pil,
    raster_from_torch,
    raster_to_pil,
    to_unit_float,
)
from invokeai.backend.image_util.raster.convert import pil_refusal

CHANNELS = {1: RasterChannels.L, 2: RasterChannels.LA, 3: RasterChannels.RGB, 4: RasterChannels.RGBA}


def descriptor(
    channel_count=3,
    value_range=ValueRange.UNIT,
    color=ColorEncoding.SRGB,
    alpha=None,
    icc_profile=None,
) -> RasterDescriptor:
    if alpha is None:
        alpha = AlphaMode.STRAIGHT if channel_count in (2, 4) else AlphaMode.NONE
    return RasterDescriptor(CHANNELS[channel_count], color, value_range, alpha, channel_count, icc_profile)


def raster(array: np.ndarray, **kwargs) -> RasterImage:
    kwargs.setdefault("channel_count", array.shape[2])
    return RasterImage.from_array(array, descriptor(**kwargs))


def pattern(height, width, channels, dtype=np.uint8) -> np.ndarray:
    # Distinct value per sample and per axis, so transposed or swapped channels are visible.
    values = np.arange(height * width * channels, dtype=np.int64) * 37 + 11
    return (values % np.iinfo(dtype).max).astype(dtype).reshape(height, width, channels)


# --- PIL ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_pil_round_trip_is_exact_for_8bit_modes(mode):
    channels = len(mode)
    array = pattern(5, 7, channels)
    source = Image.fromarray(array[:, :, 0] if channels == 1 else array, mode)
    converted = raster_from_pil(source)
    assert converted.dtype == np.uint8 and converted.is_owned
    assert converted.descriptor.color is ColorEncoding.SRGB
    assert converted.descriptor.alpha is (AlphaMode.STRAIGHT if mode.endswith("A") else AlphaMode.NONE)
    assert (converted.width, converted.height, converted.channel_count) == (7, 5, channels)
    np.testing.assert_array_equal(converted.view(), array)
    back = raster_to_pil(converted)
    assert back.mode == mode and back.tobytes() == source.tobytes()


def test_pil_import_does_not_alias_the_pil_image():
    source = Image.fromarray(pattern(3, 3, 3), "RGB")
    converted = raster_from_pil(source)
    expected = converted.view().copy()
    source.paste((255, 255, 255), (0, 0, 3, 3))
    np.testing.assert_array_equal(converted.view(), expected)


def test_pil_export_is_independent_of_the_raster():
    source = raster(pattern(3, 4, 3))
    expected = source.view().copy()
    image = raster_to_pil(source)
    image.paste((1, 2, 3), (0, 0, 4, 3))
    np.testing.assert_array_equal(source.view(), expected)


@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B"])
def test_pil_16bit_grayscale_imports_exactly_with_low_bits_set(mode):
    # Non-zero low bytes that are not any uint8 value times 257.
    values = np.array([[0, 1, 255, 256, 257, 0x1234], [0x00FF, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF, 40001]], np.uint16)
    image = Image.new(mode, (6, 2))
    for y in range(2):
        for x in range(6):
            image.putpixel((x, y), int(values[y, x]))
    converted = raster_from_pil(image)
    assert converted.dtype == np.uint16 and converted.descriptor.channels is RasterChannels.L
    np.testing.assert_array_equal(converted.view()[:, :, 0], values)


def test_uint16_gray_exports_to_pil_exactly():
    values = np.array([[0, 1, 255, 256, 0x1234, 0xFFFF], [3, 0x0102, 0x7FFF, 0x8000, 0xABCD, 65534]], np.uint16)
    image = raster_to_pil(raster(values[:, :, np.newaxis]))
    assert image.mode == "I;16" and image.size == (6, 2)
    exported = np.array([[image.getpixel((x, y)) for x in range(6)] for y in range(2)])
    np.testing.assert_array_equal(exported, values)
    np.testing.assert_array_equal(raster_from_pil(image).view()[:, :, 0], values)


def test_pil_bilevel_imports_as_0_255_gray():
    image = Image.new("1", (3, 1))
    image.putpixel((1, 0), 1)
    converted = raster_from_pil(image)
    assert converted.descriptor.channels is RasterChannels.L
    assert converted.view()[0, :, 0].tolist() == [0, 255, 0]


def test_pil_palette_imports_as_rgb_or_rgba_depending_on_transparency():
    opaque = Image.new("P", (2, 1))
    opaque.putpalette([10, 20, 30, 40, 50, 60])
    opaque.putpixel((1, 0), 1)
    converted = raster_from_pil(opaque)
    assert converted.descriptor.channels is RasterChannels.RGB
    assert converted.view()[0].tolist() == [[10, 20, 30], [40, 50, 60]]

    transparent = opaque.copy()
    transparent.info["transparency"] = 1
    converted = raster_from_pil(transparent)
    assert converted.descriptor.channels is RasterChannels.RGBA
    assert converted.view()[0, :, 3].tolist() == [255, 0]


def test_pil_palette_alpha_mode_imports_as_rgba():
    image = Image.new("PA", (2, 1))
    image.putpalette([10, 20, 30, 40, 50, 60])
    image.putpixel((0, 0), (0, 255))
    image.putpixel((1, 0), (1, 77))
    converted = raster_from_pil(image)
    assert converted.descriptor.channels is RasterChannels.RGBA
    assert converted.view()[0].tolist() == [[10, 20, 30, 255], [40, 50, 60, 77]]


@pytest.mark.parametrize(
    ("mode", "key", "keyed", "other", "expected_channels"),
    [
        ("L", 7, (7,), (9,), RasterChannels.LA),
        ("RGB", (1, 2, 3), (1, 2, 3), (4, 5, 6), RasterChannels.RGBA),
    ],
)
def test_pil_color_key_becomes_straight_alpha(mode, key, keyed, other, expected_channels):
    image = Image.new(mode, (2, 1))
    image.putpixel((0, 0), keyed if len(keyed) > 1 else keyed[0])
    image.putpixel((1, 0), other if len(other) > 1 else other[0])
    image.info["transparency"] = key
    converted = raster_from_pil(image)
    assert converted.descriptor.channels is expected_channels
    assert converted.descriptor.alpha is AlphaMode.STRAIGHT
    assert converted.view()[0].tolist() == [[*keyed, 0], [*other, 255]]


def test_pil_16_bit_color_key_becomes_uint16_alpha():
    image = Image.fromarray(np.array([[5, 900, 65535]], np.uint16))
    image.info["transparency"] = 900
    converted = raster_from_pil(image)
    assert converted.dtype == np.uint16 and converted.descriptor.channels is RasterChannels.LA
    assert converted.descriptor.alpha is AlphaMode.STRAIGHT
    assert converted.view()[0].tolist() == [[5, 65535], [900, 0], [65535, 65535]]


@pytest.mark.parametrize(("key", "expected_alpha"), [(255, [0, 255]), (0, [255, 0])])
def test_pil_bilevel_color_key_becomes_alpha(key, expected_alpha):
    image = Image.new("1", (2, 1))
    image.putpixel((0, 0), 1)
    image.info["transparency"] = key
    converted = raster_from_pil(image)
    assert converted.descriptor.channels is RasterChannels.LA and converted.dtype == np.uint8
    assert converted.view()[0, :, 0].tolist() == [255, 0]
    assert converted.view()[0, :, 1].tolist() == expected_alpha


@pytest.mark.parametrize("mode", ["CMYK", "I", "F", "YCbCr", "LAB", "HSV", "La", "RGBa"])
def test_pil_rejects_unsupported_modes(mode):
    with pytest.raises(RasterConversionError):
        raster_from_pil(Image.new(mode, (2, 2)))


def test_icc_profile_round_trips_through_pil():
    profile = b"not-a-real-profile-but-bytes"
    image = Image.new("RGB", (2, 2))
    image.info["icc_profile"] = profile
    converted = raster_from_pil(image)
    assert converted.descriptor.icc_profile == profile
    assert raster_to_pil(converted).info["icc_profile"] == profile
    assert "icc_profile" not in raster_to_pil(raster(pattern(2, 2, 3))).info


@pytest.mark.parametrize(
    "bad",
    [
        raster(pattern(2, 2, 3, np.uint16)),
        raster(pattern(2, 2, 3, np.uint8).astype(np.float32) / 255),
        raster(pattern(2, 2, 3), color=ColorEncoding.LINEAR_SRGB),
        raster(pattern(2, 2, 4), alpha=AlphaMode.PREMULTIPLIED),
        raster(pattern(2, 2, 2, np.uint16)),
        RasterImage.from_array(
            pattern(2, 2, 3),
            RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNIT, AlphaMode.NONE, 3),
        ),
    ],
)
def test_pil_export_rejects_what_it_cannot_represent_exactly(bad):
    with pytest.raises(RasterConversionError):
        raster_to_pil(bad)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: raster(pattern(2, 2, 3), color=ColorEncoding.LINEAR_SRGB), "linear_srgb"),
        (lambda: raster(pattern(2, 2, 4), alpha=AlphaMode.PREMULTIPLIED), "premultiplied"),
        (
            lambda: RasterImage.from_array(
                pattern(2, 2, 3),
                RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNIT, AlphaMode.NONE, 3),
            ),
            "data",
        ),
        (lambda: raster(pattern(2, 2, 3, np.uint16)), "uint16 RGB"),
        (lambda: raster(pattern(2, 2, 3).astype(np.float32) / 255), "float32 RGB"),
    ],
    ids=["linear", "premultiplied", "data", "uint16-color", "float"],
)
def test_pil_refusal_names_each_class_raster_to_pil_refuses(make, reason):
    bad = make()

    refusal = pil_refusal(bad)

    assert refusal is not None and reason in refusal
    with pytest.raises(RasterConversionError, match=refusal):
        raster_to_pil(bad)


@pytest.mark.parametrize(
    "good",
    [raster(pattern(2, 2, 3)), raster(pattern(2, 2, 1, np.uint16)), raster(pattern(2, 2, 2))],
    ids=["uint8-rgb", "uint16-l", "uint8-la"],
)
def test_pil_refusal_is_none_for_what_raster_to_pil_exports(good):
    assert pil_refusal(good) is None
    raster_to_pil(good)


# --- torch -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_torch_round_trip_keeps_axes_and_values(dtype):
    # 3 x 5 x 7 (C x H x W): non-square, so a transposed or channel-swapped layout cannot pass.
    chw = (torch.arange(3 * 5 * 7).reshape(3, 5, 7) % 200).to(dtype)
    converted = raster_from_torch(chw, descriptor(3))
    assert (converted.height, converted.width, converted.channel_count) == (5, 7, 3)
    for c, y, x in [(0, 0, 0), (1, 4, 6), (2, 0, 6), (0, 3, 1)]:
        assert converted.view()[y, x, c] == chw[c, y, x].item()
    assert converted.dtype == chw.numpy().dtype
    assert torch.equal(torch.from_numpy(converted.view().transpose(2, 0, 1).copy()), chw)


def test_torch_import_copies_detaches_and_never_aliases():
    chw = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    converted = raster_from_torch(chw, descriptor(2))
    expected = converted.view().copy()
    with torch.no_grad():
        chw.zero_()
    np.testing.assert_array_equal(converted.view(), expected)


def test_torch_import_accepts_non_contiguous_tensors():
    chw = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6).flip(2)
    converted = raster_from_torch(chw, descriptor(3))
    assert converted.view()[1, 0, 2] == chw[2, 1, 0].item()


def test_torch_rejects_bf16_batches_and_other_dtypes():
    with pytest.raises(RasterConversionError):
        raster_from_torch(torch.zeros(3, 2, 2, dtype=torch.bfloat16), descriptor(3))
    with pytest.raises(RasterConversionError):
        raster_from_torch(torch.zeros(3, 2, 2, dtype=torch.float64), descriptor(3))
    with pytest.raises(RasterConversionError):
        raster_from_torch(torch.zeros(1, 3, 2, 2), descriptor(3))


# --- quantize_unit -----------------------------------------------------------------------------------------------


def expected_quantized(value: float, full_scale: int) -> int:
    """Independent oracle in exact rational arithmetic; Fraction.__round__ is round-half-to-even."""
    exact = min(max(Fraction(float(value)), Fraction(0)), Fraction(1)) * full_scale
    return round(exact)


FLOAT_SAMPLES = (
    [0.0, 1.0, 0.5, -0.0, 1e-9, 1 - 1e-9, 0.25, 0.75, 0.999, 1.25, -3.0, 60000.0]
    + [float(np.nextafter(np.float32(0.5), np.float32(0))), float(np.nextafter(np.float32(0.5), np.float32(1)))]
    + [k / 255 for k in range(256)]
    + [(k + 0.5) / 255 for k in range(255)]
    + [k / 65535 for k in range(0, 65536, 997)]
    + list(np.random.default_rng(7).random(300))
)


@pytest.mark.parametrize(("dtype", "scale"), [(np.uint8, 255), (np.uint16, 65535)])
@pytest.mark.parametrize("float_dtype", [np.float32, np.float16])
def test_quantize_matches_exact_rational_rounding(dtype, scale, float_dtype):
    samples = np.array(FLOAT_SAMPLES, dtype=float_dtype)
    result = quantize_unit(raster(samples.reshape(1, -1, 1)), dtype)
    assert result.dtype == dtype and result.descriptor == descriptor(1)
    assert result.view()[0, :, 0].tolist() == [expected_quantized(v, scale) for v in samples]


@pytest.mark.parametrize(("dtype", "tie_result"), [(np.uint8, 128), (np.uint16, 32768)])
def test_quantize_rounds_exact_ties_to_even(dtype, tie_result):
    # 0.5 * odd full scale is an exact tie (127.5, 32767.5); the even neighbour is 128 / 32768.
    below = np.nextafter(np.float32(0.5), np.float32(0))
    above = np.nextafter(np.float32(0.5), np.float32(1))
    result = quantize_unit(raster(np.array([[[below], [0.5], [above]]], np.float32)), dtype)
    assert result.view()[0, :, 0].tolist() == [tie_result - 1, tie_result, tie_result]


def test_quantize_clips_overshoot():
    source = np.array([[[-0.25, 0.0, 1.0, 7.5], [2.0, 0.5, -1e6, 1.0]]], np.float32)
    result = quantize_unit(raster(source), np.uint8)
    assert result.view().tolist() == [[[0, 0, 255, 255], [255, 128, 0, 255]]]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_quantize_rejects_nonfinite(bad):
    with pytest.raises(RasterConversionError):
        quantize_unit(raster(np.array([[[0.1, bad, 0.3]]], np.float32)), np.uint8)


def test_quantize_rejects_unbounded_and_unsupported_targets():
    unbounded = raster(np.zeros((1, 1, 3), np.float32), value_range=ValueRange.UNBOUNDED)
    with pytest.raises(RasterConversionError):
        quantize_unit(unbounded, np.uint8)
    with pytest.raises(RasterConversionError):
        quantize_unit(raster(np.zeros((1, 1, 3), np.float32)), np.float32)  # type: ignore[arg-type]


def test_uint8_to_uint16_is_exact_for_every_value():
    every = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
    result = quantize_unit(raster(every), np.uint16)
    assert result.dtype == np.uint16
    assert result.view()[0, :, 0].tolist() == [v * 257 for v in range(256)]


def test_uint16_to_uint8_rounds_to_nearest_and_inverts_widening():
    values = np.array(
        [0, 1, 128, 129, 256, 257, 257 + 128, 257 + 129, 32639, 32896, 65535 - 129, 65535 - 128, 65535, 40000],
        np.uint16,
    )
    result = quantize_unit(raster(values.reshape(1, -1, 1)), np.uint8)
    assert result.view()[0, :, 0].tolist() == [round(Fraction(int(v), 257)) for v in values]
    every = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
    round_trip = quantize_unit(quantize_unit(raster(every), np.uint16), np.uint8)
    np.testing.assert_array_equal(round_trip.view(), every)


def test_quantize_same_dtype_returns_an_owned_copy():
    source = RasterImage.borrow(pattern(2, 2, 3), descriptor(3))
    result = quantize_unit(source, np.uint8)
    assert result is not source and result.is_owned and not np.shares_memory(result.view(), source.view())
    np.testing.assert_array_equal(result.view(), source.view())


# --- to_unit_float -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("dtype", "scale"), [(np.uint8, 255), (np.uint16, 65535)])
def test_integer_to_float32_divides_by_full_scale_exactly(dtype, scale):
    values = np.unique(np.concatenate([np.arange(0, scale + 1, max(1, scale // 300)), [1, scale - 1, scale]]))
    result = to_unit_float(raster(values.astype(dtype).reshape(1, -1, 1)), np.float32)
    assert result.dtype == np.float32
    expected = np.array([float(Fraction(int(v), scale)) for v in values]).astype(np.float32)
    np.testing.assert_array_equal(result.view()[0, :, 0], expected)
    assert result.view().max() == 1.0 and result.descriptor.value_range is ValueRange.UNIT


def test_integer_to_float16_divides_in_float64_then_casts():
    values = np.arange(65536, dtype=np.uint16)[::251].reshape(1, -1, 1)
    result = to_unit_float(raster(values), np.float16)
    assert result.dtype == np.float16
    expected = np.array([float(Fraction(int(v), 65535)) for v in values.ravel()]).astype(np.float16)
    np.testing.assert_array_equal(result.view()[0, :, 0], expected)


def test_float_cast_preserves_values_and_flags_float16_overflow():
    source = np.array([[[0.1, 1.5, -0.25]]], np.float32)
    narrowed = to_unit_float(raster(source), np.float16)
    np.testing.assert_array_equal(narrowed.view(), source.astype(np.float16))
    widened = to_unit_float(narrowed, np.float32)
    np.testing.assert_array_equal(widened.view(), source.astype(np.float16).astype(np.float32))
    with pytest.raises(RasterConversionError):
        to_unit_float(raster(np.array([[[1e6, 0.0, 0.0]]], np.float32)), np.float16)
    # Infinities already present are not new overflow.
    unbounded = raster(np.array([[[np.inf, np.nan, 1.0]]], np.float32), value_range=ValueRange.UNBOUNDED)
    assert np.isinf(to_unit_float(unbounded, np.float16).view()[0, 0, 0])


def test_to_unit_float_result_is_owned_and_independent():
    source = RasterImage.borrow(np.full((2, 2, 3), 0.5, np.float32), descriptor(3))
    result = to_unit_float(source, np.float32)
    assert result.is_owned and not np.shares_memory(result.view(), source.view())


def test_float32_division_equals_float64_division_cast_for_every_integer_value():
    # to_unit_float divides in float32; that must match exact rational division rounded once to float32.
    for dtype, scale in [(np.uint8, 255), (np.uint16, 65535)]:
        every = np.arange(scale + 1, dtype=dtype).reshape(1, -1, 1)
        result = to_unit_float(raster(every), np.float32).view().ravel()
        expected = np.array([float(Fraction(v, scale)) for v in range(scale + 1)]).astype(np.float32)
        np.testing.assert_array_equal(result, expected)


def test_to_unit_float_rejects_integer_targets():
    with pytest.raises(RasterConversionError):
        to_unit_float(raster(pattern(1, 1, 3)), np.uint8)  # type: ignore[arg-type]


def test_conversions_of_borrowed_transposed_sources_give_c_contiguous_owned_rasters():
    source = np.ascontiguousarray(pattern(6, 4, 3)).transpose(1, 0, 2)
    borrowed = RasterImage.borrow(source, descriptor(3))
    for result in (quantize_unit(borrowed, np.uint16), to_unit_float(borrowed, np.float32)):
        assert result.is_owned and result.view().flags.c_contiguous
        assert result.owned() is result
    np.testing.assert_array_equal(quantize_unit(borrowed, np.uint16).view(), source.astype(np.uint16) * 257)


@pytest.fixture
def two_row_blocks(monkeypatch):
    # 9 rows of 5x3 samples in blocks of 2 rows: 2, 2, 2, 2, 1.
    monkeypatch.setattr("invokeai.backend.image_util.raster.convert._BLOCK_SAMPLES", 2 * 5 * 3)


def test_conversions_cover_every_row_including_the_partial_last_block(two_row_blocks):
    values = np.random.default_rng(3).random((9, 5, 3), dtype=np.float32)
    quantized = quantize_unit(raster(values), np.uint8)
    assert quantized.view().tolist() == [[[expected_quantized(v, 255) for v in px] for px in row] for row in values]
    ints = pattern(9, 5, 3, np.uint16)
    assert quantize_unit(raster(ints), np.uint8).view().tolist() == [
        [[round(Fraction(int(v), 257)) for v in px] for px in row] for row in ints
    ]
    np.testing.assert_array_equal(to_unit_float(raster(ints), np.float32).view(), (ints / 65535).astype(np.float32))


@pytest.mark.parametrize("row", [7, 8])  # second row of the last full block, and the partial last block
def test_nonfinite_and_overflow_are_caught_beyond_the_first_block(two_row_blocks, row):
    values = np.full((9, 5, 3), 0.5, np.float32)
    with_nan = values.copy()
    with_nan[row, 4, 2] = np.nan
    with pytest.raises(RasterConversionError):
        quantize_unit(raster(with_nan), np.uint8)
    with_overflow = values.copy()
    with_overflow[row, 0, 0] = 1e6
    with pytest.raises(RasterConversionError):
        to_unit_float(raster(with_overflow), np.float16)


def test_torch_uint16_round_trip():
    chw = torch.tensor([[[0, 1, 65535], [256, 40001, 7]]], dtype=torch.uint16)
    converted = raster_from_torch(chw, descriptor(1))
    assert converted.dtype == np.uint16 and converted.view()[1, 1, 0] == 40001
    assert torch.equal(torch.from_numpy(converted.view().transpose(2, 0, 1).copy()), chw)


@pytest.mark.parametrize("mode", ["L", "1", "I;16", "I;16B", "RGB"])
def test_imported_pil_rasters_are_fully_readonly(mode):
    converted = raster_from_pil(Image.new(mode, (3, 2)))
    with pytest.raises(ValueError):
        converted.view().flags.writeable = True
    with pytest.raises(ValueError):
        converted.view()[0, 0, 0] = 1


def test_pil_palette_with_alpha_column_keeps_its_alpha():
    rgba = Image.new("RGBA", (2, 1))
    rgba.putpixel((0, 0), (200, 100, 50, 255))
    rgba.putpixel((1, 0), (10, 20, 30, 0))
    palettized = rgba.quantize()
    assert palettized.mode == "P" and "transparency" not in palettized.info
    converted = raster_from_pil(palettized)
    assert converted.descriptor.channels is RasterChannels.RGBA
    assert converted.view()[0, :, 3].tolist() == [255, 0]
