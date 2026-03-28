from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("qwen_image_lora_loader_output")
class QwenImageLoRALoaderOutput(BaseInvocationOutput):
    """Qwen Image Edit LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )


@invocation(
    "qwen_image_lora_loader",
    title="Apply LoRA - Qwen Image Edit",
    tags=["lora", "model", "qwen_image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a Qwen Image Edit transformer."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=1.0, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> QwenImageLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')

        output = QwenImageLoRALoaderOutput()

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation(
    "qwen_image_lora_collection_loader",
    title="Apply LoRA Collection - Qwen Image Edit",
    tags=["lora", "model", "qwen_image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a Qwen Image Edit transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> QwenImageLoRALoaderOutput:
        output = QwenImageLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue
            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)

        return output
