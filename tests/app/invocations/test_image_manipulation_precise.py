"""Clarity, texture and contrast dehaze nodes on images with and without a precise asset.

Without an asset the PIL path must stay byte-identical. With one the filter runs on float32 unit pixels and the result
is saved precisely, or explicitly display-quantized to 8 bits when precise storage is off.
"""

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.image_manipulation import ClarityInvocation, ContrastDehazeInvocation, TextureInvocation
from invokeai.backend.image_util.clarity import clarity, clarity_unit
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze, contrast_dehaze_unit
from invokeai.backend.image_util.raster import RasterChannels
from invokeai.backend.image_util.texture import texture, texture_unit

RGB, RGBA = RasterChannels.RGB, RasterChannels.RGBA

# node type, node fields, legacy function, float function (both with the node's fields)
NODES = {
    "clarity": (
        ClarityInvocation,
        {"amount": 100, "size": 12},
        lambda image: clarity(image, 100, 12),
        lambda pixels: clarity_unit(pixels, 100, 12),
    ),
    "texture": (
        TextureInvocation,
        {"amount": 100, "size": 8},
        lambda image: texture(image, 100, 8),
        lambda pixels: texture_unit(pixels, 100, 8),
    ),
    "contrast_dehaze": (
        ContrastDehazeInvocation,
        {"amount": 100},
        lambda image: contrast_dehaze(image, 100),
        lambda pixels: contrast_dehaze_unit(pixels, 100),
    ),
}


@pytest.fixture(params=sorted(NODES))
def node(request):
    return NODES[request.param]


def _eight_bit_rgba() -> np.ndarray:
    y, x = np.indices((48, 56))
    noise = np.random.default_rng(3).normal(0, 4, y.shape)
    base = np.clip(50 + 150 * x / 56 + 20 * np.sin(2 * np.pi * x / 9) + 12 * np.sin(2 * np.pi * y / 31) + noise, 0, 255)
    rgb = np.stack([base, base * 0.8, base * 0.6], axis=-1)
    alpha = np.full(y.shape, 255.0)
    alpha[:, :14] = 130
    alpha[:8, :8] = 0
    return np.rint(np.concatenate([rgb, alpha[..., None]], axis=-1)).astype(np.uint8)


def _run(world, node_type, fields: dict, source: str, **extra) -> str:
    return node_type(id="node", image={"image_name": source}, **fields, **extra).invoke(world.context).image.image_name


def _png_pixels(world, name: str) -> np.ndarray:
    with Image.open(world.storage.get_path(name)) as image:
        return np.asarray(image.convert("RGBA")).copy()


def test_an_image_without_an_asset_is_routed_to_the_pil_path(precise_world, node) -> None:
    node_type, fields, legacy, _ = node
    pixels = _eight_bit_rgba()
    name = precise_world.store_png(pixels, "RGBA")
    assert not precise_world.images.has_precise_asset(name)

    out = _run(precise_world, node_type, fields, name)

    assert not precise_world.images.has_precise_asset(out)
    expected = legacy(Image.fromarray(pixels, "RGBA"))
    np.testing.assert_array_equal(_png_pixels(precise_world, out), np.asarray(expected))


def test_a_precise_image_is_filtered_in_float_and_saved_precisely(precise_world, node) -> None:
    node_type, fields, _, unit = node
    rng = np.random.default_rng(4)
    source = (_eight_bit_rgba().astype(np.float32) / 255) + rng.normal(0, 0.0004, (48, 56, 4)).astype(np.float32)
    source[:, :, 3] = _eight_bit_rgba()[:, :, 3] / 255
    source[:8, :8, :3] = 1.6  # hidden color beyond 0-1
    name = precise_world.store_precise(source, RGBA)

    out = _run(precise_world, node_type, fields, name)

    assert precise_world.images.has_precise_asset(out)
    raster = precise_world.images.get_raster(out)
    assert raster.dtype == np.float32 and raster.descriptor.channels is RGBA
    np.testing.assert_array_equal(raster.view(), unit(source))
    np.testing.assert_array_equal(raster.view()[:, :, 3], source[:, :, 3])
    np.testing.assert_array_equal(raster.view()[:8, :8], source[:8, :8])  # hidden pixels come back untouched
    # sub-8-bit information is still there: the result is not on the 8-bit grid
    assert np.abs(raster.view() * 255 - np.rint(raster.view() * 255)).max() > 0.1


def test_with_precise_storage_off_the_float_result_is_display_quantized_explicitly(precise_world, node) -> None:
    node_type, fields, _, unit = node
    source = np.random.default_rng(5).uniform(0.1, 0.9, (40, 44, 3)).astype(np.float32)
    name = precise_world.store_precise(source, RGB)
    precise_world.configuration.precise_image_storage = False

    out = _run(precise_world, node_type, fields, name)

    assert not precise_world.images.has_precise_asset(out)
    expected = np.rint(np.clip(unit(source), 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(_png_pixels(precise_world, out)[:, :, :3], expected)


def test_a_uint16_asset_is_widened_exactly_and_agrees_with_the_8_bit_result(precise_world, node) -> None:
    node_type, fields, legacy, _ = node
    pixels = _eight_bit_rgba()
    name = precise_world.store_precise(pixels.astype(np.uint16) * 257, RGBA)
    precise_world.configuration.precise_image_storage = False

    out = _run(precise_world, node_type, fields, name)

    expected = np.asarray(legacy(Image.fromarray(pixels, "RGBA"))).astype(np.int16)
    difference = np.abs(_png_pixels(precise_world, out).astype(np.int16) - expected)
    assert difference.max() <= 1
    assert np.mean(difference == 0) > 0.99


def test_amount_zero_keeps_a_precise_image_exactly_including_overshoot(precise_world) -> None:
    source = np.random.default_rng(6).uniform(-0.4, 1.8, (12, 14, 3)).astype(np.float32)
    name = precise_world.store_precise(source, RGB)

    for node_type in (ClarityInvocation, TextureInvocation, ContrastDehazeInvocation):
        out = _run(precise_world, node_type, {"amount": 0}, name)

        np.testing.assert_array_equal(precise_world.images.get_raster(out).view(), source)
