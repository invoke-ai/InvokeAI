from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field

from ..models.image import ImageField, ImageType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)

from controlnet_aux import CannyDetector
from .image import ImageOutput, build_image_output, PILInvocationConfig


class ControlField(BaseModel):

    image: ImageField = Field(default=None, description="processed image")
    # width: Optional[int] = Field(default=None, description="The width of the image in pixels")
    # height: Optional[int] = Field(default=None, description="The height of the image in pixels")
    # mode: Optional[str] = Field(default=None, description="The mode of the image")
    control_model: Optional[str] = Field(default=None, description="The control model used")
    control_weight: Optional[float] = Field(default=None, description="The control weight used")

    class Config:
        schema_extra = {
            "required": ["image", "control_model", "control_weight"]
            # "required": ["type", "image", "width", "height", "mode"]
        }


class ControlOutput(BaseInvocationOutput):
    """Base class for invocations that output ControlNet info"""

    # fmt: off
    type: Literal["control_output"] = "control_output"
    control: Optional[ControlField] = Field(default=None, description="The control info dict")
    # image: ImageField = Field(default=None, description="outputs just them image info (which is also included in control output)")
    # fmt: on


class CannyControlInvocation(BaseInvocation, PILInvocationConfig):
    """Canny edge detection for ControlNet"""

    # fmt: off
    type: Literal["cannycontrol"] = "cannycontrol"

    # Inputs
    image: ImageField = Field(default=None, description="image to process")
    control_model: str = Field(default=None, description="control model to use")
    control_weight: float = Field(default=0.5, ge=0, le=1, description="control weight")
    # begin_step_percent: float = Field(default=0, ge=0, le=1,
    #                                    description="% of total steps at which controlnet is first applied")
    # end_step_percent: float = Field(default=1, ge=0, le=1,
    #                                  description="% of total steps at which controlnet is last applied")
    # guess_mode: bool = Field(default=False, description="use guess mode (controlnet ignores prompt)")

    low_threshold: float = Field(default=100, ge=0, description="low threshold of Canny pixel gradient")
    high_threshold: float = Field(default=200, ge=0, description="high threshold of Canny pixel gradient")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ControlOutput:
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

        """Builds an ImageOutput and its ImageField"""
        image_field = ImageField(
            image_name=image_name,
            image_type=image_type,
        )
        return ControlOutput(
            control=ControlField(
                image=image_field,
                control_model=self.control_model,
                control_weight=self.control_weight,
            )
        )

