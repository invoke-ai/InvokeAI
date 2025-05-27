from typing import Optional

from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FluxUnoReferenceField, InputField, OutputField
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext


def preprocess_ref(raw_image: Image.Image, long_size: int = 512) -> Image.Image:
    """Resize and center crop reference image
    Code from https://github.com/bytedance/UNO/blob/main/uno/flux/pipeline.py
    """
    # Get the width and height of the original image
    image_w, image_h = raw_image.size

    # Calculate the long and short sides
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # Scale proportionally to the new width and height
    raw_image = raw_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # Calculate the starting coordinates of the clipping to achieve center clipping
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # Center crop
    raw_image = raw_image.crop((left, top, right, bottom))

    # Convert to RGB mode
    raw_image = raw_image.convert("RGB")
    return raw_image


@invocation_output("flux_uno_output")
class FluxUnoOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Redux invocation."""

    uno_refs: FluxUnoReferenceField = OutputField(description="Reference images container", title="Reference images")


@invocation(
    "flux_uno",
    title="FLUX UNO",
    tags=["uno", "control"],
    category="ip_adapter",
    version="2.1.0",
    classification=Classification.Beta,
)
class FluxUnoInvocation(BaseInvocation):
    """Loads a FLUX UNO reference images."""

    image: ImageField = InputField(description="The UNO reference image.")
    image2: Optional[ImageField] = InputField(default=None, description="2nd reference")
    image3: Optional[ImageField] = InputField(default=None, description="3rd reference")
    image4: Optional[ImageField] = InputField(default=None, description="4th reference")

    def invoke(self, context: InvocationContext) -> FluxUnoOutput:
        images: list[str] = []
        for image in [self.image, self.image2, self.image3, self.image4]:
            if image is not None:
                image_pil = context.images.get_pil(image.image_name)
                images.append(context.images.save(image=image_pil).image_name)

        return FluxUnoOutput(uno_refs=FluxUnoReferenceField(image_names=images))
