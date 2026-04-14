from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, Qwen3EncoderField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("anima_lora_loader_output")
class AnimaLoRALoaderOutput(BaseInvocationOutput):
    """Anima LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Anima Transformer"
    )
    qwen3_encoder: Optional[Qwen3EncoderField] = OutputField(
        default=None, description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder"
    )


@invocation(
    "anima_lora_loader",
    title="Apply LoRA - Anima",
    tags=["lora", "model", "anima"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class AnimaLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to an Anima transformer and/or Qwen3 text encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Anima Transformer",
    )
    qwen3_encoder: Qwen3EncoderField | None = InputField(
        default=None,
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> AnimaLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.qwen3_encoder and any(lora.lora.key == lora_key for lora in self.qwen3_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to Qwen3 encoder.')

        output = AnimaLoRALoaderOutput()

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )
        if self.qwen3_encoder is not None:
            output.qwen3_encoder = self.qwen3_encoder.model_copy(deep=True)
            output.qwen3_encoder.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation(
    "anima_lora_collection_loader",
    title="Apply LoRA Collection - Anima",
    tags=["lora", "model", "anima"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class AnimaLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to an Anima transformer."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )

    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    qwen3_encoder: Qwen3EncoderField | None = InputField(
        default=None,
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> AnimaLoRALoaderOutput:
        output = AnimaLoRALoaderOutput()

        if self.loras is None:
            if self.transformer is not None:
                output.transformer = self.transformer.model_copy(deep=True)
            if self.qwen3_encoder is not None:
                output.qwen3_encoder = self.qwen3_encoder.model_copy(deep=True)
            return output

        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)

        if self.qwen3_encoder is not None:
            output.qwen3_encoder = self.qwen3_encoder.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue

            if not context.models.exists(lora.lora.key):
                raise ValueError(f"Unknown lora: {lora.lora.key}!")

            if lora.lora.base is not BaseModelType.Anima:
                raise ValueError(
                    f"LoRA '{lora.lora.key}' is for {lora.lora.base.value if lora.lora.base else 'unknown'} models, "
                    "not Anima models. Ensure you are using an Anima compatible LoRA."
                )

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)

            if self.qwen3_encoder is not None and output.qwen3_encoder is not None:
                output.qwen3_encoder.loras.append(lora)

        return output
