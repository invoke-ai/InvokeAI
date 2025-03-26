from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField, UIType
from invokeai.app.invocations.model import ControlLoRAField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_control_lora_loader_output")
class FluxControlLoRALoaderOutput(BaseInvocationOutput):
    """Flux Control LoRA Loader Output"""

    control_lora: ControlLoRAField = OutputField(
        title="Flux Control LoRA", description="Control LoRAs to apply on model loading", default=None
    )


@invocation(
    "flux_control_lora_loader",
    title="Control LoRA - FLUX",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.1.1",
)
class FluxControlLoRALoaderInvocation(BaseInvocation):
    """LoRA model and Image to use with FLUX transformer generation."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.control_lora_model, title="Control LoRA", ui_type=UIType.ControlLoRAModel
    )
    image: ImageField = InputField(description="The image to encode.")
    weight: float = InputField(description="The weight of the LoRA.", default=1.0)

    def invoke(self, context: InvocationContext) -> FluxControlLoRALoaderOutput:
        if not context.models.exists(self.lora.key):
            raise ValueError(f"Unknown lora: {self.lora.key}!")

        return FluxControlLoRALoaderOutput(
            control_lora=ControlLoRAField(
                lora=self.lora,
                img=self.image,
                weight=self.weight,
            )
        )
