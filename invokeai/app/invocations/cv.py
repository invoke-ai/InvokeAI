# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


import cv2 as cv
import numpy
from PIL import Image, ImageOps

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation("cv_inpaint", title="OpenCV Inpaint", tags=["opencv", "inpaint"], category="inpaint", version="1.3.1")
class CvInpaintInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Simple inpaint using opencv."""

    image: ImageField = InputField(description="The image to inpaint")
    mask: ImageField = InputField(description="The mask to use when inpainting")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mask = context.images.get_pil(self.mask.image_name)

        # Convert to cv image/mask
        # TODO: consider making these utility functions
        cv_image = cv.cvtColor(numpy.array(image.convert("RGB")), cv.COLOR_RGB2BGR)
        cv_mask = numpy.array(ImageOps.invert(mask.convert("L")))

        # Inpaint
        cv_inpainted = cv.inpaint(cv_image, cv_mask, 3, cv.INPAINT_TELEA)

        # Convert back to Pillow
        # TODO: consider making a utility function
        image_inpainted = Image.fromarray(cv.cvtColor(cv_inpainted, cv.COLOR_BGR2RGB))

        image_dto = context.images.save(image=image_inpainted)

        return ImageOutput.build(image_dto)
