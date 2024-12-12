from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField, UIType
from invokeai.app.invocations.model import ControlLoRAField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_control_lora_loader_output")
class FluxControlLoRALoaderOutput(BaseInvocationOutput):
    """Flux Control LoRA Loader Output"""

    control_lora: Optional[ControlLoRAField] = OutputField(
        title="Flux Control Lora", description="Control LoRAs to apply on model loading", default=None
    )


@invocation(
    "flux_control_lora_loader",
    title="Flux Control LoRA",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class FluxControlLoRALoaderInvocation(BaseInvocation):
    """LoRA model and Image to use with FLUX transformer generation."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.control_lora_model, title="Control LoRA", ui_type=UIType.ControlLoRAModel
    )
    image: ImageField = InputField(
        description="The image to encode.",
    )

    def invoke(self, context: InvocationContext) -> FluxControlLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        output = FluxControlLoRALoaderOutput()

        output.control_lora = ControlLoRAField(
            lora=self.lora,
            img=self.image,
            weight=1,
        )
        return output
