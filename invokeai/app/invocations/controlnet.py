# Invocations for ControlNet image preprocessors
# initial implementation by Gregg Helt, 2023
from typing import List, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    OutputField,
    UIType,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import (
    CONTROLNET_MODE_VALUES,
    CONTROLNET_RESIZE_VALUES,
    heuristic_resize_fast,
)
from invokeai.backend.image_util.util import np_to_pil, pil_to_np


class ControlField(BaseModel):
    image: ImageField = Field(description="The control image")
    control_model: ModelIdentifierField = Field(description="The ControlNet model to use")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self


@invocation_output("control_output")
class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""

    # Outputs
    control: ControlField = OutputField(description=FieldDescriptions.control)


@invocation("controlnet", title="ControlNet - SD1.5, SDXL", tags=["controlnet"], category="controlnet", version="1.1.3")
class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    image: ImageField = InputField(description="The control image")
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.ControlNetModel
    )
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> "ControlNetInvocation":
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> ControlOutput:
        return ControlOutput(
            control=ControlField(
                image=self.image,
                control_model=self.control_model,
                control_weight=self.control_weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                control_mode=self.control_mode,
                resize_mode=self.resize_mode,
            ),
        )


@invocation(
    "heuristic_resize",
    title="Heuristic Resize",
    tags=["image, controlnet"],
    category="image",
    version="1.1.1",
    classification=Classification.Prototype,
)
class HeuristicResizeInvocation(BaseInvocation):
    """Resize an image using a heuristic method. Preserves edge maps."""

    image: ImageField = InputField(description="The image to resize")
    width: int = InputField(default=512, ge=1, description="The width to resize to (px)")
    height: int = InputField(default=512, ge=1, description="The height to resize to (px)")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        np_img = pil_to_np(image)
        np_resized = heuristic_resize_fast(np_img, (self.width, self.height))
        resized = np_to_pil(np_resized)
        image_dto = context.images.save(image=resized)
        return ImageOutput.build(image_dto)
