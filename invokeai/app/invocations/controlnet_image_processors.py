from typing import Literal, Optional

import numpy
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from ..models.image import ImageField, ImageType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)

from controlnet_aux import CannyDetector, HEDdetector, LineartDetector
from .image import ImageOutput, build_image_output, PILInvocationConfig


# Canny Image Processor
class CannyProcessorInvocation(BaseInvocation, PILInvocationConfig):
    """Applies Canny edge detection to image"""

    # fmt: off
    type: Literal["canny"] = "canny"

    # Inputs
    image: ImageField = Field(default=None, description="image to process")
    low_threshold:  float = Field(default=100, ge=0, description="low threshold of Canny pixel gradient")
    high_threshold: float = Field(default=200, ge=0, description="high threshold of Canny pixel gradient")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        canny_processor = CannyDetector()
        processed_image = canny_processor(image, self.low_threshold, self.high_threshold)
        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, processed_image, metadata)
        return build_image_output(
            image_type=image_type, image_name=image_name, image=processed_image
        )
