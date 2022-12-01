# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from datetime import datetime, timezone
from typing import Literal
import numpy
from pydantic import Field
from PIL import Image, ImageOps
import cv2 as cv
from .image import ImageField, ImageOutput
from .baseinvocation import BaseInvocation
from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices


class CvInpaintInvocation(BaseInvocation):
    """Simple inpaint using opencv."""
    type: Literal['cv_inpaint'] = 'cv_inpaint'

    # Inputs
    image: ImageField = Field(default=None, description="The image to inpaint")
    mask: ImageField = Field(default=None, description="The mask to use when inpainting")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)
        mask = services.images.get(self.mask.image_type, self.mask.image_name)

        # Convert to cv image/mask
        # TODO: consider making these utility functions
        cv_image = cv.cvtColor(numpy.array(image.convert('RGB')), cv.COLOR_RGB2BGR)
        cv_mask = numpy.array(ImageOps.invert(mask))

        # Inpaint
        cv_inpainted = cv.inpaint(cv_image, cv_mask, 3, cv.INPAINT_TELEA)

        # Convert back to Pillow
        # TODO: consider making a utility function
        image_inpainted = Image.fromarray(cv.cvtColor(cv_inpainted, cv.COLOR_BGR2RGB))

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, image_inpainted)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )
