# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


import cv2 as cv
import numpy
from PIL import Image, ImageOps

from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

from .baseinvocation import BaseInvocation, InvocationContext, invocation
from .fields import InputField, WithMetadata


@invocation("cv_inpaint", title="OpenCV Inpaint", tags=["opencv", "inpaint"], category="inpaint", version="1.2.0")
class CvInpaintInvocation(BaseInvocation, WithMetadata):
    """Simple inpaint using opencv."""

    image: ImageField = InputField(description="The image to inpaint")
    mask: ImageField = InputField(description="The mask to use when inpainting")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mask = context.services.images.get_pil_image(self.mask.image_name)

        # Convert to cv image/mask
        # TODO: consider making these utility functions
        cv_image = cv.cvtColor(numpy.array(image.convert("RGB")), cv.COLOR_RGB2BGR)
        cv_mask = numpy.array(ImageOps.invert(mask.convert("L")))

        # Inpaint
        cv_inpainted = cv.inpaint(cv_image, cv_mask, 3, cv.INPAINT_TELEA)

        # Convert back to Pillow
        # TODO: consider making a utility function
        image_inpainted = Image.fromarray(cv.cvtColor(cv_inpainted, cv.COLOR_BGR2RGB))

        image_dto = context.services.images.create(
            image=image_inpainted,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
