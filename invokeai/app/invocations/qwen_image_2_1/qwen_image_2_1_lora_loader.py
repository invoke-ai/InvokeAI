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


@invocation_output("qwen_image_2_1_lora_loader_output")
class QwenImage21LoRALoaderOutput(BaseInvocationOutput):
    """Qwen-Image-2.1 LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )


def _require_qwen_image_21_lora(context: InvocationContext, lora: ModelIdentifierField) -> None:
    """Refuse a LoRA that is not installed as a Qwen-Image-2.1 LoRA: its keys would match nothing, or the wrong
    layers, in this transformer."""
    if not context.models.exists(lora.key):
        raise ValueError(f"Unknown LoRA: {lora.key}!")
    config = context.models.get_config(lora.key)
    if config.type is not ModelType.LoRA:
        raise ValueError(f"'{config.name}' is a {config.type.value} model, not a LoRA.")
    if config.base is not BaseModelType.QwenImage21:
        hint = " Qwen Image LoRAs do not fit Qwen-Image-2.1." if config.base is BaseModelType.QwenImage else ""
        raise ValueError(f"LoRA '{config.name}' is for {config.base.value} models, not Qwen-Image-2.1.{hint}")


@invocation(
    "qwen_image_2_1_lora_loader",
    title="Apply LoRA - Qwen-Image-2.1",
    tags=["lora", "model", "qwen_image_2_1"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImage21LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a Qwen-Image-2.1 transformer."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.QwenImage21,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=1.0, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> QwenImage21LoRALoaderOutput:
        _require_qwen_image_21_lora(context, self.lora)
        if self.transformer and any(lora.lora.key == self.lora.key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{self.lora.key}" already applied to transformer.')

        output = QwenImage21LoRALoaderOutput()
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        return output


@invocation(
    "qwen_image_2_1_lora_collection_loader",
    title="Apply LoRA Collection - Qwen-Image-2.1",
    tags=["lora", "model", "qwen_image_2_1"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImage21LoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a Qwen-Image-2.1 transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRA models and weights. May be a single LoRA or collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.QwenImage21],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )

    def invoke(self, context: InvocationContext) -> QwenImage21LoRALoaderOutput:
        output = QwenImage21LoRALoaderOutput()
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)

        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        # Including those an upstream loader applied: a LoRA applied twice doubles its weight.
        added = {lora.lora.key for lora in output.transformer.loras} if output.transformer is not None else set()
        for lora in loras:
            if lora is None or lora.lora.key in added:
                continue
            _require_qwen_image_21_lora(context, lora.lora)
            added.add(lora.lora.key)
            if output.transformer is not None:
                output.transformer.loras.append(lora)
        return output
