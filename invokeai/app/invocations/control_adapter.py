from builtins import bool, float
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator
from invokeai.app.invocations.primitives import ImageField

from ...backend.model_management import BaseModelType

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    InputField,
    Input,
    InvocationContext,
    OutputField,
    UIType,
    tags,
    title,
)


CONTROLNET_MODE_VALUES = Literal["balanced", "more_prompt", "more_control", "unbalanced"]
CONTROLNET_RESIZE_VALUES = Literal[
    "just_resize",
    "crop_resize",
    "fill_resize",
    "just_resize_simple",
]


class ControlNetModelField(BaseModel):
    """ControlNet model field"""

    model_name: str = Field(description="Name of the ControlNet model")
    base_model: BaseModelType = Field(description="Base model")


class ControlField(BaseModel):
    image: ImageField = Field(description="The control image")
    control_model: ControlNetModelField = Field(description="The ControlNet model to use")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @validator("control_weight")
    def validate_control_weight(cls, v):
        """Validate that all control weights in the valid range"""
        if isinstance(v, list):
            for i in v:
                if i < -1 or i > 2:
                    raise ValueError("Control weights must be within -1 to 2 range")
        else:
            if v < -1 or v > 2:
                raise ValueError("Control weights must be within -1 to 2 range")
        return v


class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""

    type: Literal["control_output"] = "control_output"

    # Outputs
    control: ControlField = OutputField(description=FieldDescriptions.control)


@title("ControlNet")
@tags("controlnet")
class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    type: Literal["controlnet"] = "controlnet"

    # Inputs
    image: ImageField = InputField(description="The control image")
    control_model: ControlNetModelField = InputField(
        default="lllyasviel/sd-controlnet-canny", description=FieldDescriptions.controlnet_model, input=Input.Direct
    )
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, description="The weight given to the ControlNet", ui_type=UIType.Float
    )
    begin_step_percent: float = InputField(
        default=0, ge=-1, le=2, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

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
