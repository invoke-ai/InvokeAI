from pydantic import BaseModel, Field, field_validator, model_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import CONTROLNET_RESIZE_VALUES


class FluxControlNetField(BaseModel):
    image: ImageField = Field(description="The control image")
    controlnet_model: ModelIdentifierField = Field(description="The ControlNet model to use")
    control_weight: float | list[float] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")
    instantx_control_mode: int = Field(
        default=0,
        description="The control mode for InstantX ControlNet union models. Ignored for other ControlNet models.",
    )

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v: float | list[float]) -> float | list[float]:
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self


@invocation_output("flux_controlnet_output")
class FluxControlNetOutput(BaseInvocationOutput):
    """FLUX ControlNet info"""

    controlnet: FluxControlNetField = OutputField(description=FieldDescriptions.control)


@invocation(
    "flux_controlnet",
    title="FLUX ControlNet",
    tags=["controlnet", "flux"],
    category="controlnet",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxControlNetInvocation(BaseInvocation):
    """Collect FLUX ControlNet info to pass to other nodes."""

    image: ImageField = InputField(description="The control image")
    controlnet_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.ControlNetModel
    )
    control_weight: float | list[float] = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")
    instantx_control_mode: int = InputField(
        default=0,
        description="The control mode for InstantX ControlNet union models. Ignored for other ControlNet models.",
    )

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v: float | list[float]) -> float | list[float]:
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> FluxControlNetOutput:
        return FluxControlNetOutput(
            controlnet=FluxControlNetField(
                image=self.image,
                controlnet_model=self.controlnet_model,
                control_weight=self.control_weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                resize_mode=self.resize_mode,
                instantx_control_mode=self.instantx_control_mode,
            ),
        )
