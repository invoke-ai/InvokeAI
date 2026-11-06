import numpy as np
import pytest
import torch

from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    raster_as_unit_tensor,
    resize_unit_tensor,
    trim_unit_tensor_to_grid,
)


def _raster(array: np.ndarray, channels: RasterChannels, **overrides) -> RasterImage:
    fields = {
        "channels": channels,
        "color": ColorEncoding.SRGB,
        "value_range": ValueRange.UNIT,
        "alpha": AlphaMode.STRAIGHT if channels in (RasterChannels.LA, RasterChannels.RGBA) else AlphaMode.NONE,
        "channel_count": len(channels.value),
    }
    fields.update(overrides)
    return RasterImage.from_array(array, RasterDescriptor(**fields))


def _pixels(channel_count: int, dtype) -> np.ndarray:
    values = np.arange(3 * 4 * channel_count).reshape(3, 4, channel_count)
    return (values * 5).astype(dtype)


@pytest.mark.parametrize(("dtype", "scale"), [(np.uint8, 255.0), (np.uint16, 65535.0)])
def test_integer_rasters_map_to_unit_range_per_channel(dtype, scale) -> None:
    pixels = _pixels(4, dtype)

    rgb = raster_as_unit_tensor(_raster(pixels, RasterChannels.RGBA))
    rgba = raster_as_unit_tensor(_raster(pixels, RasterChannels.RGBA), alpha=True)

    expected = (pixels.astype(np.float64) / scale).transpose(2, 0, 1)
    assert rgb.dtype == torch.float32 and rgb.is_contiguous()
    np.testing.assert_allclose(rgb.numpy(), expected[:3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(rgba.numpy(), expected, rtol=0, atol=1e-6)


def test_float_values_pass_through_unclamped() -> None:
    pixels = np.linspace(-0.5, 1.75, 3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)

    tensor = raster_as_unit_tensor(_raster(pixels, RasterChannels.RGB))

    np.testing.assert_array_equal(tensor.numpy(), pixels.transpose(2, 0, 1))


def test_half_precision_is_widened() -> None:
    pixels = np.linspace(-0.5, 1.5, 3 * 4 * 3).reshape(3, 4, 3).astype(np.float16)

    tensor = raster_as_unit_tensor(_raster(pixels, RasterChannels.RGB))

    assert tensor.dtype == torch.float32
    np.testing.assert_array_equal(tensor.numpy(), pixels.astype(np.float32).transpose(2, 0, 1))


def test_alpha_is_filled_with_one_where_the_raster_has_none() -> None:
    pixels = np.linspace(0, 1, 3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)

    tensor = raster_as_unit_tensor(_raster(pixels, RasterChannels.RGB), alpha=True)

    assert tensor.shape == (4, 3, 4)
    np.testing.assert_array_equal(tensor[:3].numpy(), pixels.transpose(2, 0, 1))
    assert (tensor[3] == 1.0).all()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gray_is_replicated_and_la_keeps_its_alpha_only_on_request(dtype) -> None:
    scale = float(np.iinfo(dtype).max)
    gray = _pixels(1, dtype)
    gray_alpha = _pixels(2, dtype)

    plain = raster_as_unit_tensor(_raster(gray, RasterChannels.L)).numpy()
    plain_with_alpha = raster_as_unit_tensor(_raster(gray, RasterChannels.L), alpha=True).numpy()
    la = raster_as_unit_tensor(_raster(gray_alpha, RasterChannels.LA)).numpy()
    la_with_alpha = raster_as_unit_tensor(_raster(gray_alpha, RasterChannels.LA), alpha=True).numpy()

    luma = gray[:, :, 0] / scale
    for channel in range(3):
        np.testing.assert_allclose(plain[channel], luma, atol=1e-6)
        np.testing.assert_allclose(plain_with_alpha[channel], luma, atol=1e-6)
        np.testing.assert_allclose(la[channel], gray_alpha[:, :, 0] / scale, atol=1e-6)
        np.testing.assert_allclose(la_with_alpha[channel], gray_alpha[:, :, 0] / scale, atol=1e-6)
    assert plain.shape[0] == 3 and la.shape[0] == 3
    assert (plain_with_alpha[3] == 1.0).all()
    np.testing.assert_allclose(la_with_alpha[3], gray_alpha[:, :, 1] / scale, atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"color": ColorEncoding.LINEAR_SRGB},
        {"value_range": ValueRange.UNBOUNDED},
        {"alpha": AlphaMode.PREMULTIPLIED},
    ],
    ids=["linear", "unbounded", "premultiplied"],
)
def test_rasters_without_a_display_policy_are_refused(overrides) -> None:
    raster = _raster(np.zeros((4, 4, 4), np.float32), RasterChannels.RGBA, **overrides)

    with pytest.raises(ValueError):
        raster_as_unit_tensor(raster, alpha=True)


def test_a_data_raster_is_refused() -> None:
    descriptor = RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNBOUNDED, AlphaMode.NONE, 3)

    with pytest.raises(ValueError):
        raster_as_unit_tensor(RasterImage.from_array(np.zeros((4, 4, 3), np.float32), descriptor))


def test_resize_unit_tensor_is_antialiased_bicubic_and_unclamped() -> None:
    chw = torch.linspace(-0.5, 1.5, 3 * 8 * 12).reshape(3, 8, 12)
    chw[:, 3:5, 5:7] = 4.0

    resized = resize_unit_tensor(chw, 4, 6)

    assert resized.shape == (3, 4, 6)
    expected = torch.nn.functional.interpolate(chw[None], size=(4, 6), mode="bicubic", antialias=True)[0]
    torch.testing.assert_close(resized, expected)
    assert resized.max() > 1.0


@pytest.mark.parametrize(("size", "grid"), [((16, 24), (16, 24)), ((19, 27), (16, 24)), ((37, 50), (32, 48))])
def test_trim_unit_tensor_to_grid_resamples_only_when_a_side_is_off_the_grid(size, grid) -> None:
    chw = torch.rand(3, *size)

    trimmed = trim_unit_tensor_to_grid(chw, multiple_of=8)

    assert trimmed.shape[-2:] == grid
    if size == grid:
        assert trimmed is chw
    else:
        torch.testing.assert_close(trimmed, resize_unit_tensor(chw, *grid))
