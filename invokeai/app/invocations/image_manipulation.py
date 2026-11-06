# SPDX-License-Identifier: Apache-2.0
"""First-party image manipulation invocations."""

from collections.abc import Callable

import numpy

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.precise_ops import precise_source, save_precise, unit_float32
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.clarity import clarity, clarity_unit
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze, contrast_dehaze_unit
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.image_util.texture import texture, texture_unit


def _invoke_precise(
    context: InvocationContext, image_name: str, adjust: Callable[[numpy.ndarray], numpy.ndarray]
) -> ImageOutput | None:
    """The result of `adjust` on the float32 unit pixels of an image with a precise asset; None without one.

    No asset keeps the node's PIL path (byte-identical). With one nothing is quantized to 8 bits except by the
    explicit display conversion `save_precise` makes when precise storage is off.
    """
    raster = precise_source(context, image_name)
    if raster is None:
        return None
    unit = unit_float32(raster)
    result = RasterImage.from_array(adjust(unit.view()), unit.descriptor)
    return ImageOutput.build(save_precise(context, result))


@invocation(
    "clarity",
    title="Clarity",
    tags=["image", "detail", "clarity", "enhance"],
    category="image",
    version="1.0.1",
)
class ClarityInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust local midtone detail in an image."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-200, le=200, description="Strength of the clarity adjustment")
    size: float = InputField(default=40, ge=4, le=1000, description="Scale of the detail adjustment")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        precise = _invoke_precise(
            context, self.image.image_name, lambda pixels: clarity_unit(pixels, self.amount, self.size)
        )
        if precise is not None:
            return precise
        image = context.images.get_pil(self.image.image_name)
        result = clarity(image, self.amount, self.size)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


@invocation(
    "texture",
    title="Texture",
    tags=["image", "detail", "texture", "enhance"],
    category="image",
    version="1.0.1",
)
class TextureInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust fine-scale texture detail in an image."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-200, le=200, description="Strength of the texture adjustment")
    size: float = InputField(default=8, ge=1, le=100, description="Scale of the texture detail")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        precise = _invoke_precise(
            context, self.image.image_name, lambda pixels: texture_unit(pixels, self.amount, self.size)
        )
        if precise is not None:
            return precise
        image = context.images.get_pil(self.image.image_name)
        result = texture(image, self.amount, self.size)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


@invocation(
    "contrast_dehaze",
    title="Contrast Dehaze (approximation)",
    tags=["image", "contrast", "dehaze", "enhance"],
    category="image",
    version="1.0.1",
)
class ContrastDehazeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust broad-scale contrast and chroma as an aesthetic dehazing approximation."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-200, le=200, description="Strength of the contrast and dehaze adjustment")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        precise = _invoke_precise(
            context, self.image.image_name, lambda pixels: contrast_dehaze_unit(pixels, self.amount)
        )
        if precise is not None:
            return precise
        image = context.images.get_pil(self.image.image_name)
        result = contrast_dehaze(image, self.amount)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)
