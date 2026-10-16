# SPDX-License-Identifier: Apache-2.0
"""First-party image manipulation invocations."""

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.clarity import clarity
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze
from invokeai.backend.image_util.texture import texture


@invocation(
    "clarity",
    title="Clarity",
    tags=["image", "detail", "clarity", "enhance"],
    category="image",
    version="1.0.0",
)
class ClarityInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust local midtone detail in an image."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-100, le=100, description="Strength of the clarity adjustment")
    size: float = InputField(default=40, ge=4, le=1000, description="Scale of the detail adjustment")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        result = clarity(image, self.amount, self.size)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


@invocation(
    "texture",
    title="Texture",
    tags=["image", "detail", "texture", "enhance"],
    category="image",
    version="1.0.0",
)
class TextureInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust fine-scale texture detail in an image."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-100, le=100, description="Strength of the texture adjustment")
    size: float = InputField(default=8, ge=1, le=100, description="Scale of the texture detail")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        result = texture(image, self.amount, self.size)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)


@invocation(
    "contrast_dehaze",
    title="Contrast Dehaze (approximation)",
    tags=["image", "contrast", "dehaze", "enhance"],
    category="image",
    version="1.0.0",
)
class ContrastDehazeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjust broad-scale contrast and chroma as an aesthetic dehazing approximation."""

    image: ImageField = InputField(description="The image to process")
    amount: float = InputField(default=0, ge=-100, le=100, description="Strength of the contrast and dehaze adjustment")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        result = contrast_dehaze(image, self.amount)
        image_dto = context.images.save(image=result)
        return ImageOutput.build(image_dto)
