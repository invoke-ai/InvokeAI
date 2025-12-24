# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Z-Image Control invocation for spatial conditioning."""

from pydantic import BaseModel, Field

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
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


class ZImageControlField(BaseModel):
    """A Z-Image control conditioning field for spatial control (Canny, HED, Depth, Pose, MLSD)."""

    image_name: str = Field(description="The name of the preprocessed control image")
    control_model: ModelIdentifierField = Field(description="The Z-Image ControlNet adapter model")
    control_context_scale: float = Field(
        default=0.75,
        ge=0.0,
        le=2.0,
        description="The strength of the control signal. Recommended range: 0.65-0.80.",
    )
    begin_step_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the control is first applied (% of total steps)",
    )
    end_step_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the control is last applied (% of total steps)",
    )


@invocation_output("z_image_control_output")
class ZImageControlOutput(BaseInvocationOutput):
    """Z-Image Control output containing control configuration."""

    control: ZImageControlField = OutputField(description="Z-Image control conditioning")


@invocation(
    "z_image_control",
    title="Z-Image ControlNet",
    tags=["image", "z-image", "control", "controlnet"],
    category="control",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ZImageControlInvocation(BaseInvocation):
    """Configure Z-Image ControlNet for spatial conditioning.

    Takes a preprocessed control image (e.g., Canny edges, depth map, pose)
    and a Z-Image ControlNet adapter model to enable spatial control.

    Supports 5 control modes: Canny, HED, Depth, Pose, MLSD.
    Recommended control_context_scale: 0.65-0.80.
    """

    image: ImageField = InputField(
        description="The preprocessed control image (Canny, HED, Depth, Pose, or MLSD)",
    )
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model,
        title="Control Model",
        ui_model_base=BaseModelType.ZImage,
        ui_model_type=ModelType.ControlNet,
    )
    control_context_scale: float = InputField(
        default=0.75,
        ge=0.0,
        le=2.0,
        description="Strength of the control signal. Recommended range: 0.65-0.80.",
        title="Control Scale",
    )
    begin_step_percent: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="When the control is first applied (% of total steps)",
    )
    end_step_percent: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="When the control is last applied (% of total steps)",
    )

    def invoke(self, context: InvocationContext) -> ZImageControlOutput:
        return ZImageControlOutput(
            control=ZImageControlField(
                image_name=self.image.image_name,
                control_model=self.control_model,
                control_context_scale=self.control_context_scale,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
            )
        )
