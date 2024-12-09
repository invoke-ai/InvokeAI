from typing import Optional, Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType, ImageField
from invokeai.app.invocations.model import VAEField, StructuralLoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_structural_lora_loader_output")
class FluxStructuralLoRALoaderOutput(BaseInvocationOutput):
    """Flux Structural LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="FLUX Transformer"
    )


@invocation(
    "flux_structural_lora_loader",
    title="Flux Structural LoRA",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class FluxStructuralLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a FLUX transformer and/or text encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.structural_lora_model, title="Structural LoRA", ui_type=UIType.StructuralLoRAModel
    )
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="FLUX Transformer",
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, ui_order=0)
    image: ImageField = InputField(
        description="The image to encode.",
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)

    def invoke(self, context: InvocationContext) -> FluxStructuralLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        # Check for existing LoRAs with the same key.
        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.structural_loras):
            raise ValueError(f'Structural LoRA "{lora_key}" already applied to transformer.')

        output = FluxStructuralLoRALoaderOutput()

        # Attach LoRA layers to the models.
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.structural_loras.append(
                StructuralLoRAField(
                    lora=self.lora,
                    vae=self.vae,
                    img=self.image,
                    weight=self.weight,
                )
            )

        return output
