"""Select a Krea-2 Control LoRA and the control image latents it conditions the denoise on."""

from pydantic import Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

_CONTROL_LATENTS_DESCRIPTION = (
    "The control image (e.g. a depth map) encoded by 'Image to Latents - Qwen Image' at the generation width and "
    "height."
)
_WEIGHT_DESCRIPTION = (
    "Scales the LoRA's block weights; 0 disables the control LoRA. The trained input projection is always applied "
    "whole."
)


class Krea2ControlLoRAField(LoRAField):
    """A Krea-2 Control LoRA, its block-LoRA weight and the control latents it conditions on."""

    control_latents: LatentsField = Field(description=_CONTROL_LATENTS_DESCRIPTION)


@invocation_output("krea2_control_lora_output")
class Krea2ControlLoRAOutput(BaseInvocationOutput):
    """Krea-2 Control LoRA output."""

    control_lora: Krea2ControlLoRAField = OutputField(title="Control LoRA", description="Krea-2 Control LoRA")


@invocation(
    "krea2_control_lora",
    title="Control LoRA - Krea-2",
    tags=["lora", "control", "krea2", "krea-2"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2ControlLoRAInvocation(BaseInvocation):
    """Condition a Krea-2 generation on a control image through a Krea-2 Control LoRA.

    A Control LoRA is trained for one control type (the public one is depth, near = white). The control image's
    latents are concatenated to the latent tokens at the transformer input. Works on Krea-2 Raw and Turbo.
    """

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.control_lora_model,
        title="Control LoRA",
        ui_model_base=BaseModelType.Krea2,
        ui_model_type=ModelType.ControlLoRa,
    )
    control_latents: LatentsField = InputField(
        description=_CONTROL_LATENTS_DESCRIPTION, input=Input.Connection, title="Control Latents"
    )
    weight: float = InputField(default=1.0, description=_WEIGHT_DESCRIPTION)

    def invoke(self, context: InvocationContext) -> Krea2ControlLoRAOutput:
        if not context.models.exists(self.lora.key):
            raise ValueError(f"Unknown Control LoRA: {self.lora.key}!")
        return Krea2ControlLoRAOutput(
            control_lora=Krea2ControlLoRAField(lora=self.lora, weight=self.weight, control_latents=self.control_latents)
        )
