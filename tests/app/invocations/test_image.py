from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
from PIL import Image, ImageFilter

from invokeai.app.invocations.image import ImageField, OklabUnsharpMaskInvocation, OklchImageHueAdjustmentInvocation
from invokeai.backend.image_util.color_conversion import (
    linear_srgb_from_oklab,
    linear_srgb_from_oklch,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    oklch_from_oklab,
    srgb_from_linear_srgb,
)


def test_oklab_unsharp_mask_invocation_preserves_alpha_and_sharpens_in_oklab() -> None:
    input_image = Image.new("RGBA", (3, 1))
    input_image.putdata(
        [
            (255, 0, 0, 32),
            (0, 255, 0, 128),
            (0, 0, 255, 224),
        ]
    )

    context = MagicMock()
    context.images.get_pil.return_value = input_image
    context.images.save.side_effect = lambda image: SimpleNamespace(
        image_name="out", width=image.width, height=image.height
    )

    invocation = OklabUnsharpMaskInvocation(image=ImageField(image_name="in"), radius=1.0, strength=50.0)
    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]

    assert output.image.image_name == "out"
    assert output.width == 3
    assert output.height == 1
    assert numpy.asarray(saved_image.getchannel("A")).reshape(-1).tolist() == [32, 128, 224]

    rgb = numpy.asarray(input_image.convert("RGB"), dtype=numpy.float32) / 255.0
    blurred_rgb = (
        numpy.asarray(input_image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=numpy.float32)
        / 255.0
    )

    rgb_unsharp = numpy.clip(rgb + (rgb - blurred_rgb) * 0.5, 0.0, 1.0)
    oklab_unsharp = srgb_from_linear_srgb(
        linear_srgb_from_oklab(
            numpy.clip(
                oklab_from_linear_srgb(linear_srgb_from_srgb(rgb))
                + (
                    oklab_from_linear_srgb(linear_srgb_from_srgb(rgb))
                    - oklab_from_linear_srgb(linear_srgb_from_srgb(blurred_rgb))
                )
                * 0.5,
                -1.0,
                1.0,
            )
        )
    )

    assert not numpy.allclose(oklab_unsharp, rgb_unsharp, atol=1e-3)
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0, oklab_unsharp, atol=1 / 255.0
    )


def test_oklch_hue_adjustment_invocation_preserves_alpha_and_rotates_hue_in_oklch() -> None:
    input_image = Image.new("RGBA", (2, 1))
    input_image.putdata(
        [
            (210, 80, 30, 64),
            (40, 160, 220, 192),
        ]
    )

    context = MagicMock()
    context.images.get_pil.return_value = input_image
    context.images.save.side_effect = lambda image: SimpleNamespace(
        image_name="out", width=image.width, height=image.height
    )

    invocation = OklchImageHueAdjustmentInvocation(image=ImageField(image_name="in"), hue=180)
    output = invocation.invoke(context)
    saved_image = context.images.save.call_args.kwargs["image"]

    rgb = numpy.asarray(input_image.convert("RGB"), dtype=numpy.float32) / 255.0
    oklch = oklch_from_oklab(oklab_from_linear_srgb(linear_srgb_from_srgb(rgb)))
    rotated_oklch = oklch.copy()
    rotated_oklch[..., 2] = (rotated_oklch[..., 2] + 180.0) % 360.0
    expected_rgb = srgb_from_linear_srgb(linear_srgb_from_oklch(rotated_oklch))

    assert output.image.image_name == "out"
    assert output.width == 2
    assert output.height == 1
    assert numpy.asarray(saved_image.getchannel("A")).reshape(-1).tolist() == [64, 192]
    assert numpy.allclose(
        numpy.asarray(saved_image.convert("RGB"), dtype=numpy.float32) / 255.0, expected_rgb, atol=1 / 255.0
    )
