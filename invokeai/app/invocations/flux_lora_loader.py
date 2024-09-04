from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("flux_lora_loader_output")
class FluxLoRALoaderOutput(BaseInvocationOutput):
    """FLUX LoRA Loader Output"""

    transformer: TransformerField = OutputField(
        default=None, description=FieldDescriptions.transformer, title="FLUX Transformer"
    )


@invocation(
    "flux_lora_loader",
    title="FLUX LoRA",
    tags=["lora", "model", "flux"],
    category="model",
    version="1.0.0",
)
class FluxLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a FLUX transformer."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model, title="LoRA", ui_type=UIType.LoRAModel
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="FLUX Transformer",
    )

    def invoke(self, context: InvocationContext) -> FluxLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to transformer.')

        transformer = self.transformer.model_copy(deep=True)
        transformer.loras.append(
            LoRAField(
                lora=self.lora,
                weight=self.weight,
            )
        )

        return FluxLoRALoaderOutput(transformer=transformer)
