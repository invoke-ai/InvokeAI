from invokeai.backend.bria.controlnet_bria import BRIA_CONTROL_MODES
from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext


class BriaControlNetField(BaseModel):
    image: ImageField = Field(description="The control image")
    model: ModelIdentifierField = Field(description="The ControlNet model to use")
    mode: BRIA_CONTROL_MODES = Field(description="The mode of the ControlNet")
    conditioning_scale: float = Field(description="The weight given to the ControlNet")

@invocation_output("flux_controlnet_output")
class BriaControlNetOutput(BaseInvocationOutput):
    """FLUX ControlNet info"""

    control: BriaControlNetField = OutputField(description=FieldDescriptions.control)


@invocation(
    "bria_controlnet",
    title="Bria ControlNet",
    tags=["controlnet", "bria"],
    category="controlnet",
    version="1.0.0",
)
class BriaControlNetInvocation(BaseInvocation):
    """Collect Bria ControlNet info to pass to denoiser node."""

    control_image: ImageField = InputField(description="The control image")
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.BriaControlNetModel
    )
    control_mode: BRIA_CONTROL_MODES = InputField(
        default="depth", description="The mode of the ControlNet"
    )
    control_weight: float = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )

    def invoke(self, context: InvocationContext) -> BriaControlNetOutput:
        return BriaControlNetOutput(
            control=BriaControlNetField(
                image=self.control_image,
                model=self.control_model,
                mode=self.control_mode,
                conditioning_scale=self.control_weight,
            ),
        )
