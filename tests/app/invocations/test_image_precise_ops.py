"""Crop, pad, bounding-box crop, resize, scale, lerp and inverse lerp on images with a precise asset.

Images without an asset keep the PIL path, whose output must stay byte-identical. Images with one are processed as
float rasters and saved precisely, or explicitly display-quantized to 8 bits when precise storage is off.
"""

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.fields import BoundingBoxField
from invokeai.app.invocations.image import (
    CenterPadCropInvocation,
    CropImageToBoundingBoxInvocation,
    ImageCropInvocation,
    ImageInverseLerpInvocation,
    ImageLerpInvocation,
    ImageResizeInvocation,
    ImageScaleInvocation,
)
from invokeai.backend.image_util.raster import AlphaMode, RasterChannels, RasterImage

RGB, RGBA, LA = RasterChannels.RGB, RasterChannels.RGBA, RasterChannels.LA


def _overshooting(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """float32 pixels in [-0.25, 1.5]: a precise-only image that no 8-bit PNG could hold."""
    return np.random.default_rng(seed).uniform(-0.25, 1.5, size=shape).astype(np.float32)


def _run(world, node_type, source: str, **fields):
    node = node_type(id="node", image={"image_name": source}, **fields)
    return node.invoke(world.context).image.image_name


def _raster(world, name: str):
    return world.images.get_raster(name)


def _expected_window(source: np.ndarray, left: int, top: int, width: int, height: int, rgba: bool) -> np.ndarray:
    """Pixel by pixel: zeros outside the source; inside, the source pixel (as RGBA, opaque, when `rgba`)."""
    channels = 4 if rgba else source.shape[2]
    out = np.zeros((height, width, channels), source.dtype)
    for y in range(height):
        for x in range(width):
            sy, sx = y + top, x + left
            if 0 <= sy < source.shape[0] and 0 <= sx < source.shape[1]:
                pixel = source[sy, sx]
                if rgba and source.shape[2] == 3:
                    pixel = np.append(pixel, 1 if source.dtype.kind == "f" else np.iinfo(source.dtype).max)
                elif rgba and source.shape[2] == 2:
                    pixel = np.array([pixel[0], pixel[0], pixel[0], pixel[1]], source.dtype)
                out[y, x] = pixel
    return out


def test_crop_outside_the_image_is_transparent_and_overshoot_survives(precise_world) -> None:
    source = _overshooting((4, 5, 3))
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, ImageCropInvocation, name, x=-2, y=1, width=6, height=5)

    raster = _raster(precise_world, out)
    assert precise_world.images.has_precise_asset(out)
    assert raster.dtype == np.float32
    assert raster.descriptor.channels is RGBA and raster.descriptor.alpha is AlphaMode.STRAIGHT
    np.testing.assert_array_equal(raster.view(), _expected_window(source, left=-2, top=1, width=6, height=5, rgba=True))
    assert raster.view().max() > 1.0 and raster.view().min() < 0.0


def test_crop_keeps_a_uint16_source_exact_and_makes_gray_alpha_rgba(precise_world) -> None:
    wide = np.random.default_rng(1).integers(0, 65536, size=(3, 4, 3), dtype=np.uint16)
    gray_alpha = _overshooting((3, 4, 2), seed=2)
    wide_name = precise_world.store_precise(wide, RGB)
    gray_name = precise_world.store_precise(np.clip(gray_alpha, 0, 1), LA)

    wide_out = _raster(precise_world, _run(precise_world, ImageCropInvocation, wide_name, x=1, y=-1, width=4, height=3))
    gray_out = _raster(precise_world, _run(precise_world, ImageCropInvocation, gray_name, x=0, y=0, width=5, height=3))

    assert wide_out.dtype == np.uint16
    np.testing.assert_array_equal(wide_out.view(), _expected_window(wide, left=1, top=-1, width=4, height=3, rgba=True))
    np.testing.assert_array_equal(
        gray_out.view(), _expected_window(np.clip(gray_alpha, 0, 1), left=0, top=0, width=5, height=3, rgba=True)
    )


def test_pad_and_crop_from_the_sides_in_one_step(precise_world) -> None:
    source = _overshooting((3, 5, 4), seed=3)
    source[:, :, 3] = np.clip(source[:, :, 3], 0, 1)
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, CenterPadCropInvocation, name, left=2, right=-1, top=-1, bottom=3)

    raster = _raster(precise_world, out)
    assert (raster.width, raster.height) == (5 - 1 + 2, 3 + 3 - 1)
    # left=2 pads two columns; top=-1 removes the first row.
    np.testing.assert_array_equal(raster.view(), _expected_window(source, left=-2, top=1, width=6, height=5, rgba=True))


def test_a_pad_crop_that_leaves_nothing_is_an_error(precise_world) -> None:
    name = precise_world.store_precise(_overshooting((3, 5, 3)), RGB)

    with pytest.raises(ValueError):
        _run(precise_world, CenterPadCropInvocation, name, left=-3, right=-3)


def test_pad_crop_then_crop_back_is_exact_across_nodes(precise_world) -> None:
    source = _overshooting((5, 3, 3), seed=4)
    name = precise_world.store_precise(source, RGB)

    padded = _run(precise_world, CenterPadCropInvocation, name, left=2, right=1, top=3, bottom=0)
    assert _raster(precise_world, padded).dtype == np.float32
    back = _run(precise_world, ImageCropInvocation, padded, x=2, y=3, width=3, height=5)

    np.testing.assert_array_equal(
        _raster(precise_world, back).view(), _expected_window(source, left=0, top=0, width=3, height=5, rgba=True)
    )


def test_bounding_box_crop_keeps_the_channels_and_zero_fills(precise_world) -> None:
    source = _overshooting((4, 6, 3), seed=5)
    name = precise_world.store_precise(source, RGB)

    out = _run(
        precise_world,
        CropImageToBoundingBoxInvocation,
        name,
        bounding_box=BoundingBoxField(x_min=4, y_min=-1, x_max=8, y_max=3),
    )

    raster = _raster(precise_world, out)
    assert raster.descriptor.channels is RGB
    np.testing.assert_array_equal(
        raster.view(), _expected_window(source, left=4, top=-1, width=4, height=4, rgba=False)
    )


def test_bounding_box_crop_without_a_box_trims_transparent_pixels(precise_world) -> None:
    source = _overshooting((6, 7, 4), seed=6)
    source[:, :, 3] = 0.0
    source[2:5, 1:4, 3] = [0.25, 1.0, 0.5]  # opaque enough to count, rows 2-4 and columns 1-3
    source[0, 6, :3] = 1.5  # color alone, with alpha 0, does not count
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, CropImageToBoundingBoxInvocation, name)

    np.testing.assert_array_equal(_raster(precise_world, out).view(), source[2:5, 1:4])


def test_bounding_box_crop_without_a_box_follows_the_displayed_content(precise_world) -> None:
    source = np.zeros((8, 9, 4), np.float32)
    source[:, :, :3] = 0.5
    source[:, :, 3] = 1e-4  # rounds to 0 in the gallery PNG
    source[2:6, 3:7] = (0.9, 0.2, 0.1, 1.0)
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, CropImageToBoundingBoxInvocation, name)

    np.testing.assert_array_equal(_raster(precise_world, out).view(), source[2:6, 3:7])


def test_bounding_box_crop_without_alpha_ignores_near_black_borders(precise_world) -> None:
    source = np.full((8, 9, 3), 1e-4, np.float32)  # rounds to black in the gallery PNG
    source[1:7, 2:5] = (0.3, 0.6, 0.9)
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, CropImageToBoundingBoxInvocation, name)

    np.testing.assert_array_equal(_raster(precise_world, out).view(), source[1:7, 2:5])


def test_bounding_box_crop_of_an_empty_image_keeps_the_whole_image(precise_world) -> None:
    source = _overshooting((3, 4, 4), seed=7)
    source[:, :, 3] = 0.0
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, CropImageToBoundingBoxInvocation, name)

    np.testing.assert_array_equal(_raster(precise_world, out).view(), source)


def test_resize_box_filter_averages_blocks_and_keeps_overshoot(precise_world) -> None:
    source = _overshooting((2, 4, 3), seed=8) + 1.0  # block averages beyond 1
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, ImageResizeInvocation, name, width=2, height=1, resample_mode="box")

    expected = np.stack([source[:, :2].mean(axis=(0, 1)), source[:, 2:].mean(axis=(0, 1))])[np.newaxis]
    raster = _raster(precise_world, out)
    assert raster.dtype == np.float32
    np.testing.assert_allclose(raster.view(), expected, atol=1e-6)
    assert raster.view().max() > 1.0


def test_resize_of_transparent_edges_does_not_bleed_hidden_color(precise_world) -> None:
    source = np.zeros((4, 8, 4), np.float32)
    source[:, :4] = (1.0, 0.0, 0.0, 1.0)
    source[:, 4:] = (0.0, 1.0, 0.0, 0.0)  # fully transparent green
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, ImageResizeInvocation, name, width=16, height=8, resample_mode="bilinear")

    pixels = _raster(precise_world, out).view()
    assert pixels[:, :, 1].max() == 0.0
    assert pixels[:, :, 3].min() == 0.0 and pixels[:, :, 3].max() == 1.0


def test_scale_uses_truncated_output_sizes(precise_world) -> None:
    name = precise_world.store_precise(_overshooting((3, 5, 3), seed=9), RGB)

    small = _raster(precise_world, _run(precise_world, ImageScaleInvocation, name, scale_factor=0.5))
    large = _raster(precise_world, _run(precise_world, ImageScaleInvocation, name, scale_factor=1.5))

    assert (small.width, small.height) == (2, 1)
    assert (large.width, large.height) == (7, 4)


@pytest.mark.parametrize("resample_mode", ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"])
def test_resize_of_precise_pixels_matches_the_png_path_within_pillows_rounding(precise_world, resample_mode) -> None:
    eight_bit = np.random.default_rng(10).integers(
        64, 193, size=(5, 7, 3), dtype=np.uint8
    )  # away from Pillow's clipping
    # float32 is never PNG-representable, so this is a precise image whose values are exactly the 8-bit ones.
    precise = (eight_bit / np.float32(255)).astype(np.float32)
    precise_name = precise_world.store_precise(precise, RGB)
    png_name = precise_world.store_png(eight_bit, "RGB")
    precise_world.configuration.precise_image_storage = False  # the float result is quantized explicitly

    precise_out = _run(
        precise_world, ImageResizeInvocation, precise_name, width=11, height=9, resample_mode=resample_mode
    )
    png_out = _run(precise_world, ImageResizeInvocation, png_name, width=11, height=9, resample_mode=resample_mode)

    quantized = _raster(precise_world, precise_out)
    assert quantized.dtype == np.uint8 and not precise_world.images.has_precise_asset(precise_out)
    difference = np.abs(quantized.view().astype(int) - _raster(precise_world, png_out).view().astype(int))
    # Pillow rounds its intermediate pass to 8 bits, the float path does not.
    assert difference.max() <= 2


def test_lerp_maps_every_channel_in_unit_values_without_truncation(precise_world) -> None:
    source = _overshooting((3, 4, 4), seed=11)
    source[:, :, 3] = np.clip(source[:, :, 3], 0, 1)
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, ImageLerpInvocation, name, min=51, max=204)

    expected = source.astype(np.float64) * (204 - 51) / 255 + 51 / 255
    np.testing.assert_allclose(_raster(precise_world, out).view(), expected, atol=1e-6)
    assert _raster(precise_world, out).view().max() > 1.0


def test_lerp_with_min_above_max_inverts(precise_world) -> None:
    source = np.clip(_overshooting((2, 3, 3), seed=12), 0, 1)
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, ImageLerpInvocation, name, min=255, max=0)

    np.testing.assert_allclose(_raster(precise_world, out).view(), 1 - source.astype(np.float64), atol=1e-6)


def test_inverse_lerp_clamps_to_the_unit_range(precise_world) -> None:
    source = _overshooting((3, 4, 3), seed=13)
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, ImageInverseLerpInvocation, name, min=51, max=204)

    expected = np.clip((source.astype(np.float64) - 51 / 255) / ((204 - 51) / 255), 0, 1)
    result = _raster(precise_world, out).view()
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.min() == 0.0 and result.max() == 1.0


def test_inverse_lerp_with_equal_bounds_is_a_step(precise_world) -> None:
    source = np.array([[[0.0, 0.2, 0.2], [0.2001, 0.5, 1.5]]], np.float32)
    name = precise_world.store_precise(source, RGB)

    out = _run(precise_world, ImageInverseLerpInvocation, name, min=51, max=51)  # 51 / 255 = 0.2

    np.testing.assert_array_equal(_raster(precise_world, out).view(), [[[0, 0, 0], [1, 1, 1]]])


def test_lerp_then_inverse_lerp_reads_the_first_result_exactly(precise_world) -> None:
    source = _overshooting((3, 3, 3), seed=14)
    name = precise_world.store_precise(source, RGB)

    lerped = _run(precise_world, ImageLerpInvocation, name, min=0, max=255)  # identity map
    restored = _run(precise_world, ImageInverseLerpInvocation, lerped, min=0, max=255)

    np.testing.assert_array_equal(_raster(precise_world, lerped).view(), source)
    np.testing.assert_allclose(_raster(precise_world, restored).view(), np.clip(source, 0, 1), atol=1e-6)


@pytest.mark.parametrize(
    ("node_type", "fields"),
    [
        (ImageCropInvocation, {"x": 1, "y": 0, "width": 4, "height": 4}),
        (CenterPadCropInvocation, {"left": 1, "right": 2, "top": -1, "bottom": 1}),
        (CropImageToBoundingBoxInvocation, {}),
        (ImageResizeInvocation, {"width": 5, "height": 4, "resample_mode": "lanczos"}),
        (ImageScaleInvocation, {"scale_factor": 0.5}),
        (ImageLerpInvocation, {"min": 30, "max": 220}),
        (ImageInverseLerpInvocation, {"min": 30, "max": 220}),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else "",
)
def test_with_precise_storage_off_the_result_is_an_explicit_8_bit_quantization(
    precise_world, node_type, fields
) -> None:
    source = np.clip(_overshooting((4, 6, 4), seed=15), 0, 1)
    name = precise_world.store_precise(source, RGBA)
    on = _raster(precise_world, _run(precise_world, node_type, name, **fields))
    precise_world.configuration.precise_image_storage = False

    off_name = _run(precise_world, node_type, name, **fields)

    off = _raster(precise_world, off_name)
    assert not precise_world.images.has_precise_asset(off_name)
    assert off.dtype == np.uint8
    expected = np.rint(np.clip(on.view().astype(np.float64), 0, 1) * 255)
    np.testing.assert_array_equal(off.view(), expected.astype(np.uint8))


def _legacy_crop(image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(image, (-x, -y))
    return canvas


def _legacy_pad(image: Image.Image, left: int, right: int, top: int, bottom: int) -> Image.Image:
    canvas = Image.new("RGBA", (image.width + right + left, image.height + top + bottom), (0, 0, 0, 0))
    canvas.paste(image, (left, top))
    return canvas


def _legacy_lerp(image: Image.Image, low: int, high: int) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255
    return Image.fromarray(np.uint8(array * (high - low) + low))


def _legacy_ilerp(image: Image.Image, low: int, high: int) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    return Image.fromarray(np.uint8(np.minimum(np.maximum(array - low, 0) / float(high - low), 1) * 255))


@pytest.mark.parametrize("storage_on", [True, False])
@pytest.mark.parametrize(
    ("node_type", "fields", "legacy"),
    [
        (
            ImageCropInvocation,
            {"x": -1, "y": 2, "width": 6, "height": 5},
            lambda image: _legacy_crop(image, -1, 2, 6, 5),
        ),
        (
            CenterPadCropInvocation,
            {"left": 2, "right": -1, "top": 1, "bottom": 2},
            lambda image: _legacy_pad(image, 2, -1, 1, 2),
        ),
        (
            CropImageToBoundingBoxInvocation,
            {"bounding_box": BoundingBoxField(x_min=1, y_min=1, x_max=6, y_max=4)},
            lambda image: image.crop((1, 1, 6, 4)),
        ),
        (
            ImageResizeInvocation,
            {"width": 9, "height": 4, "resample_mode": "bicubic"},
            lambda image: image.resize((9, 4), Image.Resampling.BICUBIC),
        ),
        (
            ImageScaleInvocation,
            {"scale_factor": 1.5, "resample_mode": "lanczos"},
            lambda image: image.resize((int(7 * 1.5), int(5 * 1.5)), Image.Resampling.LANCZOS),
        ),
        (ImageLerpInvocation, {"min": 20, "max": 230}, lambda image: _legacy_lerp(image, 20, 230)),
        (ImageInverseLerpInvocation, {"min": 20, "max": 230}, lambda image: _legacy_ilerp(image, 20, 230)),
    ],
    ids=["crop", "pad_crop", "bbox_crop", "resize", "scale", "lerp", "ilerp"],
)
def test_images_without_an_asset_keep_the_pil_path_byte_for_byte(
    precise_world, node_type, fields, legacy, storage_on
) -> None:
    precise_world.configuration.precise_image_storage = storage_on
    pixels = np.random.default_rng(16).integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    name = precise_world.store_png(pixels, "RGBA")
    assert not precise_world.images.has_precise_asset(name)

    new_name = _run(precise_world, node_type, name, **fields)

    with precise_world.service.get_pil_image(name).copy() as source:
        old_name = precise_world.images.save(legacy(source)).image_name
    assert precise_world.stored_bytes(new_name) == precise_world.stored_bytes(old_name)
    assert not precise_world.images.has_precise_asset(new_name)


def _wide_source(dtype) -> tuple[np.ndarray, np.ndarray]:
    """A uint16 or float16 RGB source and its exact float64 values."""
    rng = np.random.default_rng(20)
    if dtype == np.uint16:
        wide = rng.integers(0, 65536, size=(4, 6, 3), dtype=np.uint16)
        return wide, wide / 65535
    half = rng.uniform(0.0, 1.0, size=(4, 6, 3)).astype(np.float16)
    return half, half.astype(np.float64)


@pytest.mark.parametrize("dtype", [np.uint16, np.float16])
@pytest.mark.parametrize(
    ("node_type", "fields", "expected"),
    [
        (ImageLerpInvocation, {"min": 51, "max": 204}, lambda x: x * (204 - 51) / 255 + 51 / 255),
        (
            ImageInverseLerpInvocation,
            {"min": 51, "max": 204},
            lambda x: np.clip((x - 51 / 255) / ((204 - 51) / 255), 0, 1),
        ),
    ],
    ids=["lerp", "ilerp"],
)
def test_lerp_nodes_widen_16_bit_sources_to_unit_float(precise_world, dtype, node_type, fields, expected) -> None:
    source, exact = _wide_source(dtype)
    name = precise_world.store_precise(source, RGB)

    result = _raster(precise_world, _run(precise_world, node_type, name, **fields))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result.view(), expected(exact), atol=1e-5)


@pytest.mark.parametrize("dtype", [np.uint16, np.float16])
def test_resize_and_scale_widen_16_bit_sources_to_unit_float(precise_world, dtype) -> None:
    source, exact = _wide_source(dtype)
    name = precise_world.store_precise(source, RGB)
    # box filter: 2x3 pixel blocks for the 2x2 resize, 2x2 blocks for the 0.5 scale (4x6 -> 2x3)
    expected_resize = exact.reshape(2, 2, 2, 3, 3).mean(axis=(1, 3))
    expected_scale = exact.reshape(2, 2, 3, 2, 3).mean(axis=(1, 3))

    resized = _raster(
        precise_world, _run(precise_world, ImageResizeInvocation, name, width=2, height=2, resample_mode="box")
    )
    halved = _raster(
        precise_world, _run(precise_world, ImageScaleInvocation, name, scale_factor=0.5, resample_mode="box")
    )

    assert resized.dtype == halved.dtype == np.float32
    np.testing.assert_allclose(resized.view(), expected_resize, atol=1e-5)
    np.testing.assert_allclose(halved.view(), expected_scale, atol=1e-5)


def test_crop_and_pad_keep_the_icc_profile_of_rgb_sources(precise_world) -> None:
    from PIL import ImageCms

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = _overshooting((3, 4, 3), seed=21)
    name = precise_world.images.save_raster(
        RasterImage.from_array(source, precise_world.descriptor(RGB, icc_profile=profile))
    ).image_name

    cropped = _raster(precise_world, _run(precise_world, ImageCropInvocation, name, x=1, y=1, width=3, height=3))
    padded = _raster(precise_world, _run(precise_world, CenterPadCropInvocation, name, left=1))

    assert cropped.descriptor.icc_profile == profile
    assert padded.descriptor.icc_profile == profile
