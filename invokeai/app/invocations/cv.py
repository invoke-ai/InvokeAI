# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

import cv2 as cv
import numpy
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from invokeai.app.models.image import ImageField, ImageType
from .baseinvocation import BaseInvocation, InvocationContext, InvocationConfig
from .image import ImageOutput, build_image_output


class CvInvocationConfig(BaseModel):
    """Helper class to provide all OpenCV invocations with additional config"""

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["cv", "image"],
            },
        }


class CvInpaintInvocation(BaseInvocation, CvInvocationConfig):
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

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        
        context.services.images.save(image_type, image_name, image_inpainted, metadata)
        return build_image_output(
            image_type=image_type,
            image_name=image_name,
            image=image_inpainted,
        )