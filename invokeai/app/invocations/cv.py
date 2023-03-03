# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

import cv2 as cv
import numpy
from PIL import Image, ImageOps
from pydantic import Field

from ..services.image_storage import ImageType
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput


class CvInpaintInvocation(BaseInvocation):
    """Simple inpaint using opencv."""
    #fmt: off
    type: Literal["cv_inpaint"] = "cv_inpaint"

    # Inputs
    image: ImageField = Field(default=None, description="The image to inpaint")
    mask: ImageField = Field(default=None, description="The mask to use when inpainting")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        mask = context.services.images.get(self.mask.image_type, self.mask.image_name)

        # Convert to cv image/mask
        # TODO: consider making these utility functions
        cv_image = cv.cvtColor(numpy.array(image.convert("RGB")), cv.COLOR_RGB2BGR)
        cv_mask = numpy.array(ImageOps.invert(mask))

        # Inpaint
        cv_inpainted = cv.inpaint(cv_image, cv_mask, 3, cv.INPAINT_TELEA)

        # Convert back to Pillow
        # TODO: consider making a utility function
        image_inpainted = Image.fromarray(cv.cvtColor(cv_inpainted, cv.COLOR_BGR2RGB))

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        context.services.images.save(image_type, image_name, image_inpainted)
        return ImageOutput(
            image=ImageField(image_type=image_type, image_name=image_name)
        )
