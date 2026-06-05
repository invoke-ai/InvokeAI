import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import torch
from PIL import Image, ImageFilter

from invokeai.app.invocations.image import ImageField, OklabUnsharpMaskInvocation, OklchImageHueAdjustmentInvocation
from invokeai.backend.image_util.color_conversion import (
    linear_srgb_from_oklab,
    linear_srgb_from_oklch,
    linear_srgb_from_srgb,
    okhsl_from_srgb,
    oklab_from_linear_srgb,
    oklch_from_oklab,
    srgb_from_hsl,
    srgb_from_linear_srgb,
    srgb_from_okhsl,
)

_COMPOSITION_NODES_SPEC = importlib.util.spec_from_file_location(
    "invokeai.app.invocations.composition_nodes",
    Path(__file__).resolve().parents[3] / "invokeai/app/invocations/composition-nodes.py",
)
assert _COMPOSITION_NODES_SPEC is not None
assert _COMPOSITION_NODES_SPEC.loader is not None
composition_nodes = importlib.util.module_from_spec(_COMPOSITION_NODES_SPEC)
_COMPOSITION_NODES_SPEC.loader.exec_module(composition_nodes)
InvokeAdjustImageHuePlusInvocation = composition_nodes.InvokeAdjustImageHuePlusInvocation
InvokeImageBlendInvocation = composition_nodes.InvokeImageBlendInvocation


def _build_context(input_image: Image.Image) -> MagicMock:
    context = MagicMock()
    context.images.get_pil.return_value = input_image
    context.images.save.side_effect = lambda image: SimpleNamespace(
        image_name="out", width=image.width, height=image.height
    )
    return context


def _max_abs_diff_uint8(left: Image.Image, right: Image.Image) -> int:
    left_arr = numpy.asarray(left, dtype=numpy.int16)
    right_arr = numpy.asarray(right, dtype=numpy.int16)
    return int(numpy.abs(left_arr - right_arr).max())


def test_oklab_unsharp_mask_invocation_preserves_alpha_and_sharpens_lightness_only() -> None:
    input_image = Image.new("RGBA", (3, 1))
    input_image.putdata(
        [
            (255, 0, 0, 32),
            (0, 255, 0, 128),
            (0, 0, 255, 224),
        ]
    )

    context = _build_context(input_image)

    invocation = OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=1.0, strength=50.0)
    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]

    assert output.image.image_name == "out"
    assert output.width == 3
    assert output.height == 1
    assert numpy.asarray(saved_image.getchannel("A")).reshape(-1).tolist() == [32, 128, 224]

    rgb = torch.from_numpy(numpy.asarray(input_image.convert("RGB"), dtype=numpy.float32) / 255.0).permute(2, 0, 1)
    blurred_rgb = torch.from_numpy(
        numpy.asarray(input_image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=numpy.float32)
        / 255.0
    ).permute(2, 0, 1)

    rgb_unsharp = torch.clamp(rgb + (rgb - blurred_rgb) * 0.5, 0.0, 1.0)
    rgb_oklab = oklab_from_linear_srgb(linear_srgb_from_srgb(rgb))
    blurred_oklab = oklab_from_linear_srgb(linear_srgb_from_srgb(blurred_rgb))
    expected_oklab = rgb_oklab.clone()
    expected_oklab[0, ...] = torch.clamp(
        rgb_oklab[0, ...] + (rgb_oklab[0, ...] - blurred_oklab[0, ...]) * 0.5,
        -1.0,
        1.0,
    )
    oklab_unsharp = srgb_from_linear_srgb(linear_srgb_from_oklab(expected_oklab))

    assert not torch.allclose(oklab_unsharp, rgb_unsharp, atol=1e-3)
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0,
        oklab_unsharp.permute(1, 2, 0).numpy(),
        atol=1 / 255.0,
    )


def test_oklch_hue_adjustment_invocation_preserves_alpha_and_rotates_hue_in_oklch() -> None:
    input_image = Image.new("RGBA", (2, 1))
    input_image.putdata(
        [
            (210, 80, 30, 64),
            (40, 160, 220, 192),
        ]
    )

    context = _build_context(input_image)

    invocation = OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=180)
    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]

    rgb = torch.from_numpy(numpy.asarray(input_image.convert("RGB"), dtype=numpy.float32) / 255.0).permute(2, 0, 1)
    oklch = oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(rgb)))
    rotated_oklch = oklch.clone()
    rotated_oklch[2, ...] = (rotated_oklch[2, ...] + 180.0) % 360.0
    expected_rgb = srgb_from_linear_srgb(linear_srgb_from_oklch(rotated_oklch))

    assert output.image.image_name == "out"
    assert output.width == 2
    assert output.height == 1
    assert numpy.asarray(saved_image.getchannel("A")).reshape(-1).tolist() == [64, 192]
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0,
        expected_rgb.permute(1, 2, 0).numpy(),
        atol=1 / 255.0,
    )


def test_oklab_unsharp_mask_invocation_zero_strength_returns_original_image() -> None:
    input_image = Image.new("RGBA", (2, 2))
    input_image.putdata(
        [
            (12, 34, 56, 78),
            (90, 123, 45, 67),
            (210, 40, 80, 90),
            (255, 200, 10, 255),
        ]
    )
    context = _build_context(input_image)

    invocation = OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=1.5, strength=0.0)
    invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]

    assert _max_abs_diff_uint8(saved_image, input_image) <= 1


def test_oklab_unsharp_mask_invocation_does_not_introduce_color_on_grayscale_image() -> None:
    input_image = Image.new("RGB", (3, 1))
    input_image.putdata([(32, 32, 32), (128, 128, 128), (224, 224, 224)])
    context = _build_context(input_image)

    invocation = OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=1.0, strength=80.0)
    invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]
    saved_rgb = numpy.asarray(saved_image.convert("RGB"), dtype=numpy.uint8)

    assert numpy.abs(saved_rgb[..., 0].astype(numpy.int16) - saved_rgb[..., 1].astype(numpy.int16)).max() <= 1
    assert numpy.abs(saved_rgb[..., 1].astype(numpy.int16) - saved_rgb[..., 2].astype(numpy.int16)).max() <= 1


def test_oklab_unsharp_mask_invocation_clips_extreme_values_to_valid_rgb_range() -> None:
    input_image = Image.new("RGB", (3, 1))
    input_image.putdata([(255, 255, 255), (0, 0, 0), (255, 255, 255)])
    context = _build_context(input_image)

    invocation = OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=2.0, strength=500.0)
    invocation.invoke(context)
    saved_rgb = numpy.asarray(context.images.save.call_args.kwargs["image"].convert("RGB"), dtype=numpy.uint8)

    assert saved_rgb.min() >= 0
    assert saved_rgb.max() <= 255


def test_oklch_hue_adjustment_invocation_wraps_hue_values_and_supports_rgb_input() -> None:
    input_image = Image.new("RGB", (2, 1))
    input_image.putdata([(210, 80, 30), (40, 160, 220)])

    base_context = _build_context(input_image)
    zero_output = OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=0).invoke(base_context)
    zero_saved = base_context.images.save.call_args.kwargs["image"]

    full_turn_context = _build_context(input_image)
    full_turn_output = OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=360).invoke(
        full_turn_context
    )
    full_turn_saved = full_turn_context.images.save.call_args.kwargs["image"]

    negative_context = _build_context(input_image)
    OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=-180).invoke(negative_context)
    negative_saved = negative_context.images.save.call_args.kwargs["image"]

    positive_context = _build_context(input_image)
    OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=180).invoke(positive_context)
    positive_saved = positive_context.images.save.call_args.kwargs["image"]

    assert zero_output.width == 2
    assert zero_output.height == 1
    assert full_turn_output.width == 2
    assert full_turn_output.height == 1
    assert _max_abs_diff_uint8(zero_saved, input_image) <= 1
    assert _max_abs_diff_uint8(full_turn_saved, input_image) <= 1
    assert _max_abs_diff_uint8(negative_saved, positive_saved) <= 1


def test_new_oklab_nodes_preserve_alpha_for_non_rgba_alpha_modes() -> None:
    la_image = Image.new("LA", (2, 1))
    la_image.putdata([(32, 64), (192, 224)])

    unsharp_context = _build_context(la_image)
    OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=1.0, strength=25.0).invoke(unsharp_context)
    unsharp_saved = unsharp_context.images.save.call_args.kwargs["image"]

    hue_context = _build_context(la_image)
    OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=45).invoke(hue_context)
    hue_saved = hue_context.images.save.call_args.kwargs["image"]

    assert unsharp_saved.mode == "LA"
    assert hue_saved.mode == "LA"
    assert numpy.asarray(unsharp_saved.getchannel("A")).reshape(-1).tolist() == [64, 224]
    assert numpy.asarray(hue_saved.getchannel("A")).reshape(-1).tolist() == [64, 224]


def test_hue_adjust_plus_oklch_uses_degree_based_oklch_contract() -> None:
    input_image = Image.new("RGB", (2, 1))
    input_image.putdata([(210, 80, 30), (40, 160, 220)])

    context = _build_context(input_image)
    invocation = InvokeAdjustImageHuePlusInvocation(
        image=ImageField(image_name="in"),
        space="*Oklch / Oklab",
        degrees=180.0,
        ok_adaptive_gamut=0.0,
    )

    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.args[0]

    rgb = torch.from_numpy(numpy.asarray(input_image, dtype=numpy.float32) / 255.0).permute(2, 0, 1)
    oklch = oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(rgb)))
    rotated_oklch = oklch.clone()
    rotated_oklch[2, ...] = (rotated_oklch[2, ...] + 180.0) % 360.0
    expected_rgb = srgb_from_linear_srgb(linear_srgb_from_oklch(rotated_oklch))

    assert output.width == 2
    assert output.height == 1
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0,
        expected_rgb.permute(1, 2, 0).numpy(),
        atol=1 / 255.0,
    )


def test_hue_adjust_plus_hsv_uses_degree_hue_contract() -> None:
    input_image = Image.new("RGB", (2, 1))
    input_image.putdata([(210, 80, 30), (40, 160, 220)])

    context = _build_context(input_image)
    invocation = InvokeAdjustImageHuePlusInvocation(
        image=ImageField(image_name="in"),
        space="HSV / HSL / RGB",
        degrees=90.0,
    )

    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.args[0]

    hsv = numpy.asarray(input_image.convert("HSV"), dtype=numpy.float32) / 255.0
    hsv[..., 0] = ((hsv[..., 0] * 360.0) + 90.0) % 360.0 / 360.0
    expected_rgb = Image.fromarray((hsv * 255.0).astype(numpy.uint8), mode="HSV").convert("RGB")

    assert output.width == 2
    assert output.height == 1
    assert _max_abs_diff_uint8(saved_image.convert("RGB"), expected_rgb) <= 1


def test_hue_adjust_plus_okhsl_uses_degree_hue_contract() -> None:
    input_image = Image.new("RGB", (2, 1))
    input_image.putdata([(210, 80, 30), (40, 160, 220)])

    context = _build_context(input_image)
    invocation = InvokeAdjustImageHuePlusInvocation(
        image=ImageField(image_name="in"),
        space="Okhsl",
        degrees=90.0,
        ok_adaptive_gamut=0.0,
    )

    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.args[0]

    rgb = torch.from_numpy(numpy.asarray(input_image, dtype=numpy.float32) / 255.0).permute(2, 0, 1)
    okhsl = okhsl_from_srgb(rgb)
    rotated_okhsl = okhsl.clone()
    rotated_okhsl[0, ...] = (rotated_okhsl[0, ...] + 90.0) % 360.0
    expected_rgb = srgb_from_okhsl(rotated_okhsl)

    assert output.width == 2
    assert output.height == 1
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0,
        expected_rgb.permute(1, 2, 0).numpy(),
        atol=1 / 255.0,
    )


def test_image_blend_oklch_subtract_wraps_hue_in_degrees() -> None:
    invocation = InvokeImageBlendInvocation(
        layer_upper=ImageField(image_name="upper"),
        layer_base=ImageField(image_name="base"),
        blend_mode="Subtract",
        color_space="Oklch (Oklab)",
        opacity=1.0,
        adaptive_gamut=0.0,
    )

    upper_oklch = torch.tensor([[[0.0]], [[0.0]], [[20.0]]], dtype=torch.float32)
    lower_oklch = torch.tensor([[[0.6]], [[0.18]], [[350.0]]], dtype=torch.float32)
    expected_linear_srgb = linear_srgb_from_oklch(torch.tensor([[[0.6]], [[0.18]], [[330.0]]], dtype=torch.float32))

    blank_rgb = torch.zeros((3, 1, 1), dtype=torch.float32)
    blank_alpha = torch.ones((1, 1), dtype=torch.float32)
    image_tensors = (
        blank_rgb,
        blank_rgb,
        blank_rgb,
        blank_rgb,
        blank_alpha,
        blank_alpha,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        upper_oklch,
        lower_oklch,
        None,
        None,
        None,
        None,
    )

    blended = invocation.apply_blend(image_tensors)

    assert torch.allclose(blended, expected_linear_srgb, atol=1e-5)


def test_image_blend_hsl_subtract_wraps_hue_in_degrees() -> None:
    invocation = InvokeImageBlendInvocation(
        layer_upper=ImageField(image_name="upper"),
        layer_base=ImageField(image_name="base"),
        blend_mode="Subtract",
        color_space="HSL (RGB)",
        opacity=1.0,
        adaptive_gamut=0.0,
    )

    upper_hsl = torch.tensor([[[20.0]], [[0.0]], [[0.0]]], dtype=torch.float32)
    lower_hsl = torch.tensor([[[350.0]], [[1.0]], [[0.5]]], dtype=torch.float32)
    expected_linear_srgb = linear_srgb_from_srgb(
        srgb_from_hsl(torch.tensor([[[330.0]], [[1.0]], [[0.5]]], dtype=torch.float32))
    )

    blank_rgb = torch.zeros((3, 1, 1), dtype=torch.float32)
    blank_alpha = torch.ones((1, 1), dtype=torch.float32)
    image_tensors = (
        blank_rgb,
        blank_rgb,
        blank_rgb,
        blank_rgb,
        blank_alpha,
        blank_alpha,
        None,
        None,
        None,
        upper_hsl,
        lower_hsl,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    blended = invocation.apply_blend(image_tensors)

    assert torch.allclose(blended, expected_linear_srgb, atol=1e-5)
