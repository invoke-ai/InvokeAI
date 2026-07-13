from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, Qwen3VLEncoderField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("krea2_lora_loader_output")
class Krea2LoRALoaderOutput(BaseInvocationOutput):
    """Krea-2 LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Krea-2 Transformer"
    )
    qwen3_vl_encoder: Optional[Qwen3VLEncoderField] = OutputField(
        default=None, description=FieldDescriptions.qwen3_vl_encoder, title="Qwen3-VL Encoder"
    )


@invocation(
    "krea2_lora_loader",
    title="Apply LoRA - Krea-2",
    tags=["lora", "model", "krea2", "krea-2"],
    category="model",
    version="1.0.0",
)
class Krea2LoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a Krea-2 transformer and/or Qwen3-VL text encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Krea2,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Krea-2 Transformer",
    )
    qwen3_vl_encoder: Qwen3VLEncoderField | None = InputField(
        default=None,
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_vl_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> Krea2LoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if self.lora.base is not BaseModelType.Krea2:
            raise ValueError(
                f"LoRA '{lora_key}' is for {self.lora.base.value if self.lora.base else 'unknown'} models, "
                "not Krea-2 models. Ensure you are using a Krea-2 compatible LoRA."
            )

        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.qwen3_vl_encoder and any(lora.lora.key == lora_key for lora in self.qwen3_vl_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to Qwen3-VL encoder.')

        output = Krea2LoRALoaderOutput()

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        if self.qwen3_vl_encoder is not None:
            output.qwen3_vl_encoder = self.qwen3_vl_encoder.model_copy(deep=True)
            output.qwen3_vl_encoder.loras.append(LoRAField(lora=self.lora, weight=self.weight))

        return output


@invocation(
    "krea2_lora_collection_loader",
    title="Apply LoRA Collection - Krea-2",
    tags=["lora", "model", "krea2", "krea-2"],
    category="model",
    version="1.0.0",
)
class Krea2LoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a Krea-2 transformer and/or Qwen3-VL encoder."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    qwen3_vl_encoder: Qwen3VLEncoderField | None = InputField(
        default=None,
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_vl_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> Krea2LoRALoaderOutput:
        output = Krea2LoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
        if self.qwen3_vl_encoder is not None:
            output.qwen3_vl_encoder = self.qwen3_vl_encoder.model_copy(deep=True)

        # Seed the dedup set with LoRAs already present on the incoming fields so chaining collection
        # loaders can't apply the same LoRA twice (the transformer and encoder are kept in sync below).
        if self.transformer is not None:
            added_loras.extend(existing.lora.key for existing in self.transformer.loras)
        if self.qwen3_vl_encoder is not None:
            added_loras.extend(
                existing.lora.key
                for existing in self.qwen3_vl_encoder.loras
                if existing.lora.key not in added_loras
            )

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue
            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")
            if lora.lora.base is not BaseModelType.Krea2:
                raise ValueError(
                    f"LoRA '{lora.lora.key}' is for {lora.lora.base.value if lora.lora.base else 'unknown'} models, "
                    "not Krea-2 models. Ensure you are using a Krea-2 compatible LoRA."
                )

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)
            if self.qwen3_vl_encoder is not None and output.qwen3_vl_encoder is not None:
                output.qwen3_vl_encoder.loras.append(lora)

        return output
