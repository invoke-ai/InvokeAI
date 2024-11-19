from typing import Literal

import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "concatenate_images",
    title="Concatenate Images",
    tags=["image", "concatenate"],
    category="image",
    version="1.0.0",
)
class ConcatenateImagesInvocation(BaseInvocation):
    """Concatenate images horizontally or vertically."""

    image_1: ImageField = InputField(description="The first image to concatenate.")
    image_2: ImageField = InputField(description="The second image to concatenate.")
    direction: Literal["horizontal", "vertical"] = InputField(
        default="horizontal", description="The direction along which to concatenate the images."
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # For now, we force the images to be RGB.
        image_1 = np.array(context.images.get_pil(self.image_1.image_name, "RGB"))
        image_2 = np.array(context.images.get_pil(self.image_2.image_name, "RGB"))

        axis: int = 0
        if self.direction == "horizontal":
            axis = 1
        elif self.direction == "vertical":
            axis = 0
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

        concatenated_image = np.concatenate([image_1, image_2], axis=axis)

        image_pil = Image.fromarray(concatenated_image, mode="RGB")
        image_dto = context.images.save(image=image_pil)
        return ImageOutput.build(image_dto)
