"""Masks and compositing nodes on images with a precise asset.

Masks stay 8-bit coverage (read from the gallery PNG or tensor, divided by 255); image inputs with an asset are
composited in float32 unit values. Without an asset on any image input the PIL path is untouched.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops, ImageOps

from invokeai.app.invocations.create_denoise_mask import CreateDenoiseMaskInvocation
from invokeai.app.invocations.create_gradient_mask import CreateGradientMaskInvocation
from invokeai.app.invocations.fields import BoundingBoxField
from invokeai.app.invocations.image import (
    ApplyMaskToImageInvocation,
    CanvasPasteBackInvocation,
    CanvasV2MaskAndCropInvocation,
    ImageMultiplyInvocation,
    ImagePasteInvocation,
    PasteImageIntoBoundingBoxInvocation,
)
from invokeai.app.invocations.mask import ApplyMaskTensorToImageInvocation
from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.backend.image_util.raster import RasterChannels
from invokeai.backend.model_manager.taxonomy import ModelType, ModelVariantType

RGB, RGBA = RasterChannels.RGB, RasterChannels.RGBA


def _eight_bit(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _float_twin(world, pixels: np.ndarray, channels: RasterChannels) -> str:
    """A precise image whose values are exactly `pixels / 255` (float32 is never PNG-representable)."""
    return world.store_precise((pixels / np.float32(255)).astype(np.float32), channels)


def _run(world, node) -> str:
    return node.invoke(world.context).image.image_name


def _floats(world, name: str) -> np.ndarray:
    return world.images.get_raster(name).view()


def _in_8bit_steps(world, name: str) -> np.ndarray:
    """A result in 0-255 units, whatever it is stored as."""
    raster = world.images.get_raster(name)
    return raster.view().astype(np.float64) * (255 if raster.dtype.kind == "f" else 1)


def _assert_matches_pil(world, precise_name: str, legacy_name: str, tolerance: float = 1.0) -> None:
    """The float result, in 8-bit steps, is the PIL result up to PIL's own rounding."""
    expected = _in_8bit_steps(world, legacy_name)
    result = np.clip(_in_8bit_steps(world, precise_name), 0, 255)
    assert result.shape == expected.shape
    assert np.abs(result - expected).max() <= tolerance


def _mask_png(world, coverage_u8: np.ndarray) -> str:
    return world.store_png(coverage_u8[:, :, np.newaxis], "L")


def test_paste_matches_pillow_for_every_mask_offset_and_crop(precise_world) -> None:
    base_pixels, image_pixels = _eight_bit((6, 7, 4), 1), _eight_bit((4, 5, 4), 2)
    base_png, image_png = precise_world.store_png(base_pixels, "RGBA"), precise_world.store_png(image_pixels, "RGBA")
    base_float, image_float = (
        _float_twin(precise_world, base_pixels, RGBA),
        _float_twin(precise_world, image_pixels, RGBA),
    )
    mask = _mask_png(precise_world, _eight_bit((4, 5, 1), 3)[:, :, 0])

    for x, y in [(0, 0), (2, 1), (-2, -1), (4, 5)]:
        for use_mask in (False, True):
            for crop in (False, True):
                fields = {"x": x, "y": y, "crop": crop}
                if use_mask:
                    fields["mask"] = {"image_name": mask}
                legacy = _run(
                    precise_world,
                    ImagePasteInvocation(
                        id="p", base_image={"image_name": base_png}, image={"image_name": image_png}, **fields
                    ),
                )
                for base, image in [(base_float, image_float), (base_float, image_png), (base_png, image_float)]:
                    precise = _run(
                        precise_world,
                        ImagePasteInvocation(
                            id="p", base_image={"image_name": base}, image={"image_name": image}, **fields
                        ),
                    )
                    assert precise_world.images.has_precise_asset(precise)
                    # Two Pillow roundings stack here (the masked paste, then the fixed-point alpha composite).
                    _assert_matches_pil(precise_world, precise, legacy, tolerance=2.0)


def test_paste_scales_the_pasted_image_by_the_inverted_mask_then_composites(precise_world) -> None:
    base = np.full((1, 3, 4), (0.2, 0.2, 0.2, 1.0), np.float32)
    image = np.full((1, 3, 4), (1.4, 0.5, -0.1, 1.0), np.float32)  # color beyond 0-1
    base_name = precise_world.store_precise(base, RGBA)
    image_name = precise_world.store_precise(image, RGBA)
    mask = _mask_png(precise_world, np.array([[0, 51, 255]], np.uint8))  # inverted: coverage 1, 0.8, 0

    out = _floats(
        precise_world,
        _run(
            precise_world,
            ImagePasteInvocation(
                id="p",
                base_image={"image_name": base_name},
                image={"image_name": image_name},
                mask={"image_name": mask},
            ),
        ),
    )

    # The layer is the image times the coverage (alpha too), composited straight over the opaque base.
    coverage = np.array([1.0, 0.8, 0.0])
    layer_alpha = coverage
    expected_alpha = layer_alpha + 1.0 * (1 - layer_alpha)
    for column in range(3):
        a = layer_alpha[column]
        for channel in range(3):
            layer_color = image[0, column, channel] * coverage[column]
            expected = (layer_color * a + base[0, column, channel] * 1.0 * (1 - a)) / expected_alpha[column]
            assert out[0, column, channel] == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(out[0, :, 3], expected_alpha, atol=1e-6)
    assert out[0, 0, 0] == pytest.approx(1.4)  # fully covered: the overshoot survives exactly
    np.testing.assert_array_equal(out[0, 2], base[0, 2])  # no coverage: the base, unchanged


def test_paste_keeps_hidden_color_of_the_base_where_nothing_is_pasted(precise_world) -> None:
    base = np.zeros((2, 4, 4), np.float32)
    base[:, :, :3] = 0.9  # hidden under alpha 0
    base_name = precise_world.store_precise(base, RGBA)
    image_name = precise_world.store_precise(np.full((2, 2, 4), (0.1, 0.2, 0.3, 1.0), np.float32), RGBA)

    out = _floats(
        precise_world,
        _run(
            precise_world,
            ImagePasteInvocation(id="p", base_image={"image_name": base_name}, image={"image_name": image_name}, x=2),
        ),
    )

    np.testing.assert_array_equal(out[:, :2], base[:, :2])
    np.testing.assert_allclose(out[:, 2:], np.full((2, 2, 4), (0.1, 0.2, 0.3, 1.0)), atol=1e-7)


def test_paste_with_a_mask_of_another_size_is_an_error_like_pillow(precise_world) -> None:
    base = precise_world.store_precise(np.zeros((4, 4, 4), np.float32), RGBA)
    image = precise_world.store_precise(np.ones((3, 3, 4), np.float32), RGBA)
    mask = _mask_png(precise_world, np.zeros((4, 4), np.uint8))

    with pytest.raises(ValueError):
        _run(
            precise_world,
            ImagePasteInvocation(
                id="p", base_image={"image_name": base}, image={"image_name": image}, mask={"image_name": mask}
            ),
        )


@pytest.mark.parametrize("channels, mode", [(RGB, "RGB"), (RGBA, "RGBA")])
def test_multiply_matches_pillow_and_keeps_overshoot(precise_world, channels, mode) -> None:
    first, second = _eight_bit((5, 3, len(mode)), 4), _eight_bit((5, 3, len(mode)), 5)
    legacy = _run(
        precise_world,
        ImageMultiplyInvocation(
            id="m",
            image1={"image_name": precise_world.store_png(first, mode)},
            image2={"image_name": precise_world.store_png(second, mode)},
        ),
    )
    first_float, second_float = (
        _float_twin(precise_world, first, channels),
        _float_twin(precise_world, second, channels),
    )
    second_png = precise_world.store_png(second, mode)

    for one, two in [(first_float, second_float), (first_float, second_png)]:
        precise = _run(
            precise_world, ImageMultiplyInvocation(id="m", image1={"image_name": one}, image2={"image_name": two})
        )
        assert precise_world.images.has_precise_asset(precise)
        _assert_matches_pil(precise_world, precise, legacy)

    big = precise_world.store_precise(np.full((2, 2, len(mode)), 1.5, np.float32), channels)
    product = _floats(
        precise_world,
        _run(precise_world, ImageMultiplyInvocation(id="m", image1={"image_name": big}, image2={"image_name": big})),
    )
    np.testing.assert_allclose(product, 2.25)


def test_multiply_of_different_channels_is_an_error_like_pillow(precise_world) -> None:
    rgb = precise_world.store_precise(np.zeros((2, 2, 3), np.float32), RGB)
    rgba = precise_world.store_precise(np.zeros((2, 2, 4), np.float32), RGBA)

    with pytest.raises(ValueError):
        _run(precise_world, ImageMultiplyInvocation(id="m", image1={"image_name": rgb}, image2={"image_name": rgba}))


@pytest.mark.parametrize("invert", [False, True])
def test_apply_mask_sets_alpha_to_the_mask_and_keeps_color(precise_world, invert) -> None:
    pixels = np.random.default_rng(6).uniform(-0.2, 1.4, size=(3, 4, 4)).astype(np.float32)
    pixels[:, :, 3] = 0.0  # the image's own alpha (and the color hidden under it) is replaced, not multiplied
    name = precise_world.store_precise(pixels, RGBA)
    coverage = np.array([[0, 255, 128, 1], [255, 0, 7, 254], [10, 20, 30, 40]], np.uint8)

    out = _floats(
        precise_world,
        _run(
            precise_world,
            ApplyMaskToImageInvocation(
                id="a",
                image={"image_name": name},
                mask={"image_name": _mask_png(precise_world, coverage)},
                invert_mask=invert,
            ),
        ),
    )

    expected_alpha = (255 - coverage if invert else coverage) / np.float32(255)
    np.testing.assert_array_equal(out[:, :, :3], pixels[:, :, :3])
    np.testing.assert_array_equal(out[:, :, 3], expected_alpha)


def test_apply_mask_matches_pillow(precise_world) -> None:
    pixels, coverage = _eight_bit((4, 6, 4), 7), _eight_bit((4, 6), 8)
    mask = _mask_png(precise_world, coverage)
    legacy = _run(
        precise_world,
        ApplyMaskToImageInvocation(
            id="a", image={"image_name": precise_world.store_png(pixels, "RGBA")}, mask={"image_name": mask}
        ),
    )
    precise = _run(
        precise_world,
        ApplyMaskToImageInvocation(
            id="a", image={"image_name": _float_twin(precise_world, pixels, RGBA)}, mask={"image_name": mask}
        ),
    )

    _assert_matches_pil(precise_world, precise, legacy, tolerance=0.0001)


def _blur_free_paste_back(source: str, target: str, mask: str):
    return CanvasPasteBackInvocation(
        id="c",
        source_image={"image_name": source},
        target_image={"image_name": target},
        mask={"image_name": mask},
        mask_blur=0,
    )


@pytest.mark.parametrize("mask_blur", [0, 2])
def test_canvas_paste_back_matches_pillow(precise_world, mask_blur) -> None:
    source_pixels, target_pixels = _eight_bit((8, 9, 3), 9), _eight_bit((8, 9, 4), 10)
    mask = _mask_png(
        precise_world, np.where(np.arange(9) < 4, 255, 0).astype(np.uint8)[np.newaxis, :].repeat(8, axis=0)
    )
    source_png, target_png = (
        precise_world.store_png(source_pixels, "RGB"),
        precise_world.store_png(target_pixels, "RGBA"),
    )
    fields = {"mask": {"image_name": mask}, "mask_blur": mask_blur}
    legacy = _run(
        precise_world,
        CanvasPasteBackInvocation(
            id="c", source_image={"image_name": source_png}, target_image={"image_name": target_png}, **fields
        ),
    )

    for source, target in [
        (_float_twin(precise_world, source_pixels, RGB), target_png),
        (source_png, _float_twin(precise_world, target_pixels, RGBA)),
    ]:
        precise = _run(
            precise_world,
            CanvasPasteBackInvocation(
                id="c", source_image={"image_name": source}, target_image={"image_name": target}, **fields
            ),
        )
        assert (
            precise_world.images.get_raster(precise).descriptor.channels is RGB
        )  # the destination's channels, as paste
        assert precise_world.images.has_precise_asset(precise)
        _assert_matches_pil(precise_world, precise, legacy)


def test_canvas_paste_back_extremes_return_each_side_exactly(precise_world) -> None:
    source = np.random.default_rng(11).uniform(-0.3, 1.6, size=(4, 4, 3)).astype(np.float32)
    target = np.random.default_rng(12).uniform(-0.3, 1.6, size=(4, 4, 3)).astype(np.float32)
    source_name, target_name = precise_world.store_precise(source, RGB), precise_world.store_precise(target, RGB)

    # A white mask keeps the source, a black mask pastes the target (the node inverts the mask).
    keep = _floats(
        precise_world,
        _run(
            precise_world,
            _blur_free_paste_back(source_name, target_name, _mask_png(precise_world, np.full((4, 4), 255, np.uint8))),
        ),
    )
    paste = _floats(
        precise_world,
        _run(
            precise_world,
            _blur_free_paste_back(source_name, target_name, _mask_png(precise_world, np.zeros((4, 4), np.uint8))),
        ),
    )

    np.testing.assert_array_equal(keep, source)
    np.testing.assert_array_equal(paste, target)


def test_canvas_v2_mask_and_crop_pastes_or_adds_the_mask_as_alpha(precise_world) -> None:
    generated = np.random.default_rng(13).uniform(-0.3, 1.6, size=(3, 3, 3)).astype(np.float32)
    source = np.random.default_rng(14).uniform(0.0, 1.0, size=(3, 3, 3)).astype(np.float32)
    generated_name, source_name = precise_world.store_precise(generated, RGB), precise_world.store_precise(source, RGB)
    coverage = np.array([[0, 255, 255], [255, 0, 255], [255, 255, 0]], np.uint8)
    mask = _mask_png(precise_world, coverage)

    pasted = _floats(
        precise_world,
        _run(
            precise_world,
            CanvasV2MaskAndCropInvocation(
                id="v",
                generated_image={"image_name": generated_name},
                source_image={"image_name": source_name},
                mask={"image_name": mask},
            ),
        ),
    )
    with_alpha = _floats(
        precise_world,
        _run(
            precise_world,
            CanvasV2MaskAndCropInvocation(
                id="v", generated_image={"image_name": generated_name}, mask={"image_name": mask}
            ),
        ),
    )

    # The mask is inverted: white (255) keeps the source, black (0) takes the generated image.
    keep_source = (coverage == 255)[:, :, np.newaxis]
    np.testing.assert_array_equal(pasted, np.where(keep_source, source, generated))
    np.testing.assert_array_equal(with_alpha[:, :, :3], generated)
    np.testing.assert_array_equal(with_alpha[:, :, 3], (255 - coverage) / np.float32(255))


def test_paste_into_bounding_box_blends_every_channel_by_the_source_alpha(precise_world) -> None:
    target = np.random.default_rng(15).uniform(0.0, 1.0, size=(4, 5, 4)).astype(np.float32)
    source = np.random.default_rng(16).uniform(-0.2, 1.4, size=(2, 3, 4)).astype(np.float32)
    source[:, :, 3] = [[0.0, 0.25, 1.0], [1.0, 0.5, 0.0]]
    target_name, source_name = precise_world.store_precise(target, RGBA), precise_world.store_precise(source, RGBA)

    out = _floats(
        precise_world,
        _run(
            precise_world,
            PasteImageIntoBoundingBoxInvocation(
                id="b",
                source_image={"image_name": source_name},
                target_image={"image_name": target_name},
                bounding_box=BoundingBoxField(x_min=1, y_min=2, x_max=4, y_max=4),
            ),
        ),
    )

    expected = target.astype(np.float64).copy()
    alpha = source[:, :, 3:4].astype(np.float64)
    expected[2:4, 1:4] = target[2:4, 1:4] * (1 - alpha) + source * alpha  # alpha is blended like any channel
    np.testing.assert_allclose(out, expected, atol=1e-6)
    np.testing.assert_array_equal(out[:2], target[:2])


def test_paste_into_bounding_box_matches_pillow_and_checks_the_box_size(precise_world) -> None:
    target_pixels, source_pixels = _eight_bit((6, 6, 4), 17), _eight_bit((3, 4, 4), 18)
    box = BoundingBoxField(x_min=1, y_min=2, x_max=5, y_max=5)
    legacy = _run(
        precise_world,
        PasteImageIntoBoundingBoxInvocation(
            id="b",
            source_image={"image_name": precise_world.store_png(source_pixels, "RGBA")},
            target_image={"image_name": precise_world.store_png(target_pixels, "RGBA")},
            bounding_box=box,
        ),
    )
    target_png = precise_world.store_png(target_pixels, "RGBA")

    precise = _run(
        precise_world,
        PasteImageIntoBoundingBoxInvocation(
            id="b",
            source_image={"image_name": _float_twin(precise_world, source_pixels, RGBA)},
            target_image={"image_name": target_png},
            bounding_box=box,
        ),
    )
    assert precise_world.images.has_precise_asset(precise)
    _assert_matches_pil(precise_world, precise, legacy)

    with pytest.raises(ValueError):
        _run(
            precise_world,
            PasteImageIntoBoundingBoxInvocation(
                id="b",
                source_image={"image_name": _float_twin(precise_world, source_pixels, RGBA)},
                target_image={"image_name": target_png},
                bounding_box=BoundingBoxField(x_min=0, y_min=0, x_max=2, y_max=2),
            ),
        )


def test_apply_tensor_mask_replaces_visible_alpha_and_keeps_hidden_pixels(precise_world) -> None:
    pixels = np.random.default_rng(19).uniform(-0.2, 1.4, size=(2, 3, 4)).astype(np.float32)
    pixels[:, :, 3] = [[1.0, 0.0, 0.5], [0.001, 1.0, 0.0]]
    name = precise_world.store_precise(pixels, RGBA)
    tensor_name = precise_world.context.tensors.save(torch.tensor([[[True, True, False], [True, False, True]]]))

    out = _floats(
        precise_world,
        _run(
            precise_world,
            ApplyMaskTensorToImageInvocation(id="t", image={"image_name": name}, mask={"tensor_name": tensor_name}),
        ),
    )
    inverted = _floats(
        precise_world,
        _run(
            precise_world,
            ApplyMaskTensorToImageInvocation(
                id="t", image={"image_name": name}, mask={"tensor_name": tensor_name}, invert=True
            ),
        ),
    )

    np.testing.assert_array_equal(out[:, :, :3], pixels[:, :, :3])
    # Only pixels the gallery image shows (display alpha above 0) take the mask. Alpha 0 and the sub-visible 0.001
    # (which displays as 0) keep their alpha, exactly as without an asset.
    np.testing.assert_allclose(out[:, :, 3], [[1.0, 0.0, 0.0], [0.001, 0.0, 0.0]], atol=1e-9)
    np.testing.assert_allclose(inverted[:, :, 3], [[0.0, 0.0, 1.0], [0.001, 1.0, 0.0]], atol=1e-9)


# --- every float result, with precise storage off, and the untouched PIL path -------------------------------------


def _all_nodes(world):
    """Each compositing node, wired to one precise RGBA image and one gallery RGBA image."""
    precise = world.store_precise(np.random.default_rng(20).uniform(-0.2, 1.4, size=(6, 6, 4)).astype(np.float32), RGBA)
    gallery = world.store_png(_eight_bit((6, 6, 4), 21), "RGBA")
    mask = _mask_png(world, _eight_bit((6, 6), 22))
    tensor = world.context.tensors.save(torch.rand(1, 6, 6) > 0.5)
    box = BoundingBoxField(x_min=0, y_min=0, x_max=6, y_max=6)
    ref = lambda name: {"image_name": name}  # noqa: E731
    return {
        "paste": ImagePasteInvocation(id="n", base_image=ref(gallery), image=ref(precise), mask=ref(mask), x=1),
        "multiply": ImageMultiplyInvocation(id="n", image1=ref(precise), image2=ref(gallery)),
        "apply_mask": ApplyMaskToImageInvocation(id="n", image=ref(precise), mask=ref(mask)),
        "paste_back": CanvasPasteBackInvocation(
            id="n", source_image=ref(gallery), target_image=ref(precise), mask=ref(mask)
        ),
        "canvas_v2": CanvasV2MaskAndCropInvocation(id="n", generated_image=ref(precise), mask=ref(mask)),
        "bounding_box": PasteImageIntoBoundingBoxInvocation(
            id="n", source_image=ref(precise), target_image=ref(gallery), bounding_box=box
        ),
        "tensor_mask": ApplyMaskTensorToImageInvocation(id="n", image=ref(precise), mask={"tensor_name": tensor}),
    }


@pytest.mark.parametrize(
    "name", ["paste", "multiply", "apply_mask", "paste_back", "canvas_v2", "bounding_box", "tensor_mask"]
)
def test_with_precise_storage_off_the_result_is_an_explicit_8_bit_quantization(precise_world, name) -> None:
    node = _all_nodes(precise_world)[name]
    on = _floats(precise_world, _run(precise_world, node))
    assert on.dtype == np.float32
    precise_world.configuration.precise_image_storage = False

    off_name = _run(precise_world, node)

    off = precise_world.images.get_raster(off_name)
    assert not precise_world.images.has_precise_asset(off_name) and off.dtype == np.uint8
    np.testing.assert_array_equal(off.view(), np.rint(np.clip(on.astype(np.float64), 0, 1) * 255).astype(np.uint8))


def _legacy_references():
    """The pre-precise implementations, written out: name -> (node factory over gallery names, reference over PIL)."""
    return {
        "paste": (
            lambda n: ImagePasteInvocation(id="n", base_image=n["a"], image=n["b"], mask=n["m"], x=-1, y=1),
            lambda p: _legacy_paste(p["a"], p["b"], ImageOps.invert(p["m"]), -1, 1),
        ),
        "multiply": (
            lambda n: ImageMultiplyInvocation(id="n", image1=n["a"], image2=n["b"]),
            lambda p: ImageChops.multiply(p["a"], p["b"]),
        ),
        "apply_mask": (
            lambda n: ApplyMaskToImageInvocation(id="n", image=n["a"], mask=n["m"], invert_mask=True),
            lambda p: _legacy_apply_mask(p["a"], ImageOps.invert(p["m"])),
        ),
        "paste_back": (
            lambda n: CanvasPasteBackInvocation(id="n", source_image=n["a"], target_image=n["b"], mask=n["m"]),
            lambda p: _legacy_paste_back(p["a"], p["b"], p["m"]),
        ),
        "bounding_box": (
            lambda n: PasteImageIntoBoundingBoxInvocation(
                id="n",
                source_image=n["b"],
                target_image=n["a"],
                bounding_box=BoundingBoxField(x_min=0, y_min=0, x_max=8, y_max=8),
            ),
            lambda p: _legacy_bounding_box(p["b"], p["a"]),
        ),
        "tensor_mask": (
            lambda n: ApplyMaskTensorToImageInvocation(id="n", image=n["a"], mask=n["t"], invert=True),
            lambda p: _legacy_tensor_mask(p["a"], p["t"], invert=True),
        ),
        "canvas_v2_pasted": (
            lambda n: CanvasV2MaskAndCropInvocation(id="n", generated_image=n["b"], source_image=n["a"], mask=n["m"]),
            lambda p: _legacy_paste_back(p["a"], p["b"], p["m"]),
        ),
        "canvas_v2_alpha": (
            lambda n: CanvasV2MaskAndCropInvocation(id="n", generated_image=n["b"], mask=n["m"]),
            lambda p: _legacy_put_alpha(p["b"], ImageOps.invert(p["m"])),
        ),
    }


def _legacy_paste(base: Image.Image, image: Image.Image, mask: Image.Image, x: int, y: int) -> Image.Image:
    min_x, min_y = min(0, x), min(0, y)
    size = (max(base.width, image.width + x) - min_x, max(base.height, image.height + y) - min_y)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.paste(base, (abs(min_x), abs(min_y)))
    layer = Image.new("RGBA", size)
    layer.paste(image, (max(0, x), max(0, y)), mask=mask)
    return Image.alpha_composite(canvas, layer)


def _legacy_apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    r, g, b, _ = image.split()
    return Image.merge("RGBA", (r, g, b, mask))


def _legacy_paste_back(source: Image.Image, target: Image.Image, mask: Image.Image) -> Image.Image:
    source = source.copy()
    source.paste(target, (0, 0), ImageOps.invert(mask))
    return source


def _legacy_tensor_mask(image: Image.Image, tensor: torch.Tensor, invert: bool) -> Image.Image:
    mask = tensor.squeeze(0) if tensor.dim() == 3 else tensor
    mask = mask if mask.dtype == torch.bool else mask > 0.5
    mask_np = (mask.float() * 255).byte().numpy()
    if invert:
        mask_np = 255 - mask_np
    pixels = np.asarray(image)
    alpha = np.where(pixels[:, :, 3] > 0, mask_np, pixels[:, :, 3])
    return Image.fromarray(np.dstack([pixels[:, :, :3], alpha]).astype(np.uint8), "RGBA")


def _legacy_put_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.copy()
    image.putalpha(mask)
    return image


def _legacy_bounding_box(source: Image.Image, target: Image.Image) -> Image.Image:
    target = target.copy()
    target.paste(source, (0, 0, 8, 8), source)
    return target


@pytest.mark.parametrize("storage_on", [True, False])
@pytest.mark.parametrize(
    "name",
    [
        "paste",
        "multiply",
        "apply_mask",
        "paste_back",
        "bounding_box",
        "tensor_mask",
        "canvas_v2_pasted",
        "canvas_v2_alpha",
    ],
)
def test_without_any_precise_input_the_pil_path_is_byte_identical(precise_world, name, storage_on) -> None:
    precise_world.configuration.precise_image_storage = storage_on
    factory, reference = _legacy_references()[name]
    arrays = {"a": _eight_bit((8, 8, 4), 23), "b": _eight_bit((8, 8, 4), 24), "m": _eight_bit((8, 8, 1), 25)}
    names = {
        key: {"image_name": precise_world.store_png(value, "L" if key == "m" else "RGBA")}
        for key, value in arrays.items()
    }
    pil = {key: precise_world.service.get_pil_image(value["image_name"]).copy() for key, value in names.items()}
    pil = {key: image if key == "m" else image.convert("RGBA") for key, image in pil.items()}
    tensor_name = precise_world.context.tensors.save(torch.rand(8, 8) > 0.5)
    names["t"] = {"tensor_name": tensor_name}
    pil["t"] = precise_world.context.tensors.load(tensor_name)

    result = _run(precise_world, factory(names))

    expected = precise_world.images.save(reference(pil)).image_name
    assert precise_world.stored_bytes(result) == precise_world.stored_bytes(expected)
    assert not precise_world.images.has_precise_asset(result)


def test_multi_node_chain_reads_each_precise_result_exactly(precise_world) -> None:
    base = np.random.default_rng(26).uniform(-0.2, 1.4, size=(4, 4, 4)).astype(np.float32)
    base[:, :, 3] = 1.0
    name = precise_world.store_precise(base, RGBA)
    mask = _mask_png(precise_world, np.full((4, 4), 255, np.uint8))

    masked = _run(
        precise_world, ApplyMaskToImageInvocation(id="a", image={"image_name": name}, mask={"image_name": mask})
    )
    squared = _run(
        precise_world, ImageMultiplyInvocation(id="m", image1={"image_name": masked}, image2={"image_name": masked})
    )

    np.testing.assert_array_equal(_floats(precise_world, masked), base)
    np.testing.assert_array_equal(_floats(precise_world, squared), base * base)


# --- denoise / gradient masks: the masked image reaches the VAE with its stored values -------------------------


@pytest.fixture
def encode_spy(precise_world, monkeypatch):
    """Replaces the VAE encode with a recorder of the tensor it is given."""
    seen: list[torch.Tensor] = []

    def fake_encode(vae_info, fp32, tiled, image_tensor, *args, **kwargs):
        seen.append(image_tensor.clone())
        return torch.zeros(1, 4, 2, 2)

    monkeypatch.setattr(ImageToLatentsInvocation, "vae_encode", staticmethod(fake_encode))
    precise_world.use_vae(MagicMock())
    precise_world.context.models.get_config = lambda key: SimpleNamespace(
        type=ModelType.Main, variant=ModelVariantType.Inpaint
    )
    return seen


def _half_mask(world) -> tuple[str, np.ndarray]:
    coverage = np.zeros((16, 16), np.uint8)
    coverage[:, :8] = 255
    return _mask_png(world, coverage), coverage


def _denoise_node(image: str, mask: str) -> CreateDenoiseMaskInvocation:
    return CreateDenoiseMaskInvocation.model_construct(
        id="d",
        vae=SimpleNamespace(vae="vae"),
        image=SimpleNamespace(image_name=image),
        mask=SimpleNamespace(image_name=mask),
        tiled=False,
        fp32=False,
    )


def _gradient_node(image: str, mask: str) -> CreateGradientMaskInvocation:
    return CreateGradientMaskInvocation.model_construct(
        id="g",
        mask=SimpleNamespace(image_name=mask),
        edge_radius=0,
        coherence_mode="Gaussian Blur",
        minimum_denoise=0.0,
        image=SimpleNamespace(image_name=image),
        unet=SimpleNamespace(unet=SimpleNamespace(key="unet")),
        vae=SimpleNamespace(vae="vae"),
        tiled=False,
        fp32=False,
    )


@pytest.mark.parametrize("make_node", [_denoise_node, _gradient_node], ids=["denoise", "gradient"])
def test_mask_nodes_encode_the_precise_pixels_unquantized(precise_world, encode_spy, make_node) -> None:
    pixels = np.random.default_rng(27).uniform(-0.2, 1.4, size=(16, 16, 3)).astype(np.float32)
    mask, coverage = _half_mask(precise_world)

    make_node(precise_world.store_precise(pixels, RGB), mask).invoke(precise_world.context)

    keep = (coverage >= 128)[np.newaxis]
    expected = torch.from_numpy(np.transpose(pixels, (2, 0, 1)) * 2 - 1) * torch.from_numpy(keep.astype(np.float32))
    (seen,) = encode_spy
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=1e-6)
    assert float(seen[0].max()) > 1.0  # values beyond the 8-bit range reached the encoder


@pytest.mark.parametrize("make_node", [_denoise_node, _gradient_node], ids=["denoise", "gradient"])
def test_mask_nodes_without_a_precise_image_encode_the_unchanged_8_bit_tensor(
    precise_world, encode_spy, make_node
) -> None:
    from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

    pixels = _eight_bit((16, 16, 3), 28)
    name = precise_world.store_png(pixels, "RGB")
    mask, coverage = _half_mask(precise_world)

    make_node(name, mask).invoke(precise_world.context)

    keep = torch.from_numpy((coverage >= 128).astype(np.float32))[np.newaxis]
    expected = image_resized_to_grid_as_tensor(Image.fromarray(pixels, "RGB")) * keep
    torch.testing.assert_close(encode_spy[0][0], expected, rtol=0, atol=0)


# --- more routing and precision cases --------------------------------------------------------------------------


def test_multiply_crops_to_the_shared_top_left_area_like_pillow(precise_world) -> None:
    first_pixels, second_pixels = _eight_bit((4, 5, 3), 30), _eight_bit((6, 3, 3), 31)
    legacy = _run(
        precise_world,
        ImageMultiplyInvocation(
            id="m",
            image1={"image_name": precise_world.store_png(first_pixels, "RGB")},
            image2={"image_name": precise_world.store_png(second_pixels, "RGB")},
        ),
    )
    first = _float_twin(precise_world, first_pixels, RGB)
    second = _float_twin(precise_world, second_pixels, RGB)

    precise = _run(
        precise_world, ImageMultiplyInvocation(id="m", image1={"image_name": first}, image2={"image_name": second})
    )

    result = _floats(precise_world, precise)
    assert result.shape == (4, 3, 3)  # min height x min width
    expected = (first_pixels[:4, :3] / 255.0) * (second_pixels[:4, :3] / 255.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    _assert_matches_pil(precise_world, precise, legacy)


def test_canvas_v2_with_only_the_source_precise_pastes_in_float(precise_world) -> None:
    source = np.random.default_rng(32).uniform(-0.3, 1.6, size=(3, 3, 3)).astype(np.float32)
    generated_pixels = _eight_bit((3, 3, 3), 33)
    coverage = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], np.uint8)

    out = _run(
        precise_world,
        CanvasV2MaskAndCropInvocation(
            id="v",
            generated_image={"image_name": precise_world.store_png(generated_pixels, "RGB")},
            source_image={"image_name": precise_world.store_precise(source, RGB)},
            mask={"image_name": _mask_png(precise_world, coverage)},
        ),
    )

    assert precise_world.images.has_precise_asset(out)
    keep_source = (coverage == 255)[:, :, np.newaxis]  # white keeps the source
    np.testing.assert_allclose(
        _floats(precise_world, out), np.where(keep_source, source, generated_pixels / np.float32(255)), atol=1e-6
    )


def test_paste_into_bounding_box_with_only_the_target_precise(precise_world) -> None:
    target = np.random.default_rng(34).uniform(-0.3, 1.6, size=(2, 2, 4)).astype(np.float32)
    source_pixels = np.array([[[10, 20, 30, 255], [200, 100, 50, 0]], [[1, 2, 3, 255], [4, 5, 6, 0]]], np.uint8)

    out = _run(
        precise_world,
        PasteImageIntoBoundingBoxInvocation(
            id="b",
            source_image={"image_name": precise_world.store_png(source_pixels, "RGBA")},
            target_image={"image_name": precise_world.store_precise(target, RGBA)},
            bounding_box=BoundingBoxField(x_min=0, y_min=0, x_max=2, y_max=2),
        ),
    )

    result = _floats(precise_world, out)
    assert precise_world.images.has_precise_asset(out)
    np.testing.assert_allclose(result[:, 0], source_pixels[:, 0] / np.float32(255), atol=1e-6)  # alpha 255: the source
    np.testing.assert_array_equal(result[:, 1], target[:, 1])  # alpha 0: the target, overshoot untouched


def test_16_bit_precise_inputs_mix_with_8_bit_ones_at_their_full_scale(precise_world) -> None:
    wide = np.random.default_rng(35).integers(0, 65536, size=(3, 4, 3), dtype=np.uint16)
    narrow = _eight_bit((3, 4, 3), 36)
    wide_name, narrow_name = precise_world.store_precise(wide, RGB), precise_world.store_png(narrow, "RGB")

    product = _run(
        precise_world,
        ImageMultiplyInvocation(id="m", image1={"image_name": wide_name}, image2={"image_name": narrow_name}),
    )

    assert precise_world.images.has_precise_asset(product)
    np.testing.assert_allclose(
        _floats(precise_world, product), (wide / 65535.0) * (narrow / 255.0), atol=1e-6
    )  # not clipped to 8 bits

    rgba = np.random.default_rng(37).integers(0, 65536, size=(3, 4, 4), dtype=np.uint16)
    rgba[:, :, 3] = 65535
    opaque_image = np.concatenate([_eight_bit((2, 2, 3), 38), np.full((2, 2, 1), 255, np.uint8)], axis=2)
    pasted = _floats(
        precise_world,
        _run(
            precise_world,
            ImagePasteInvocation(
                id="p",
                base_image={"image_name": precise_world.store_precise(rgba, RGBA)},
                image={"image_name": precise_world.store_png(opaque_image, "RGBA")},
            ),
        ),
    )
    np.testing.assert_allclose(pasted[:, 2:], rgba[:, 2:] / 65535.0, atol=1e-6)
    np.testing.assert_allclose(pasted[:2, :2], opaque_image / 255.0, atol=1e-6)


def test_a_sub_visible_pasted_alpha_leaves_a_transparent_bases_hidden_color(precise_world) -> None:
    base = np.zeros((2, 2, 4), np.float32)
    base[:, :, :3] = 0.9  # hidden color under alpha 0
    image = np.full((2, 2, 4), (0.1, 0.2, 0.3, 0.001), np.float32)  # alpha 0.001 displays as 0

    out = _floats(
        precise_world,
        _run(
            precise_world,
            ImagePasteInvocation(
                id="p",
                base_image={"image_name": precise_world.store_precise(base, RGBA)},
                image={"image_name": precise_world.store_precise(image, RGBA)},
            ),
        ),
    )

    np.testing.assert_array_equal(out, base)


def test_an_output_beyond_the_precise_bounds_is_saved_as_8_bit_with_a_warning(precise_world, monkeypatch) -> None:
    from invokeai.app.invocations import precise_ops
    from invokeai.backend.image_util.raster.precise_asset import PreciseAssetError

    def refuse(*args):
        raise PreciseAssetError("too large")

    monkeypatch.setattr(precise_ops, "check_precise_shape_bounds", refuse)
    precise_world.context.logger = MagicMock()
    name = precise_world.store_precise(_overshooting_rgba(), RGBA)

    out = _run(precise_world, ImagePasteInvocation(id="p", base_image={"image_name": name}, image={"image_name": name}))

    assert not precise_world.images.has_precise_asset(out)
    assert precise_world.images.get_raster(out).dtype == np.uint8
    precise_world.context.logger.warning.assert_called_once()


def _overshooting_rgba() -> np.ndarray:
    pixels = np.random.default_rng(39).uniform(-0.2, 1.4, size=(3, 3, 4)).astype(np.float32)
    pixels[:, :, 3] = 1.0
    return pixels
