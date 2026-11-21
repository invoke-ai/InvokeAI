from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, QwenVLEncoderField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import QWEN_IMAGE_LORA_QWEN_VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def _warn_if_encoder_layers_are_unused(
    context: InvocationContext, lora: ModelIdentifierField, *, transformer_only: bool
) -> None:
    """Saved workflows predate the encoder connection, so a LoRA with text-encoder layers would silently lose them."""
    if not transformer_only:
        return
    patch = context.models.load(lora).model
    if isinstance(patch, ModelPatchRaw) and any(key.startswith(QWEN_IMAGE_LORA_QWEN_VL_PREFIX) for key in patch.layers):
        context.logger.warning(
            f'LoRA "{context.models.get_config(lora.key).name}" has text-encoder layers, but no Qwen VL Encoder is '
            "connected to this node, so they are not applied. Connect the Qwen VL Encoder through the LoRA node."
        )


@invocation_output("qwen_image_lora_loader_output")
class QwenImageLoRALoaderOutput(BaseInvocationOutput):
    """Qwen Image LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    qwen_vl_encoder: Optional[QwenVLEncoderField] = OutputField(
        default=None, description=FieldDescriptions.qwen_vl_encoder, title="Qwen VL Encoder"
    )


@invocation(
    "qwen_image_lora_loader",
    title="Apply LoRA - Qwen Image",
    tags=["lora", "model", "qwen_image"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImageLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a Qwen Image transformer and/or Qwen2.5-VL text encoder."""

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
    qwen_vl_encoder: QwenVLEncoderField | None = InputField(
        default=None,
        description=FieldDescriptions.qwen_vl_encoder,
        input=Input.Connection,
        title="Qwen VL Encoder",
    )

    def invoke(self, context: InvocationContext) -> QwenImageLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.qwen_vl_encoder and any(lora.lora.key == lora_key for lora in self.qwen_vl_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to Qwen VL encoder.')

        output = QwenImageLoRALoaderOutput()

        _warn_if_encoder_layers_are_unused(
            context, self.lora, transformer_only=self.transformer is not None and self.qwen_vl_encoder is None
        )
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )
        if self.qwen_vl_encoder is not None:
            output.qwen_vl_encoder = self.qwen_vl_encoder.model_copy(deep=True)
            output.qwen_vl_encoder.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation(
    "qwen_image_lora_collection_loader",
    title="Apply LoRA Collection - Qwen Image",
    tags=["lora", "model", "qwen_image"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImageLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a Qwen Image transformer and/or Qwen2.5-VL text encoder."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRA models and weights. May be a single LoRA or collection.",
        title="LoRAs",
        ui_model_base=[BaseModelType.QwenImage],
        ui_model_type=ModelType.LoRA,
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    qwen_vl_encoder: QwenVLEncoderField | None = InputField(
        default=None,
        description=FieldDescriptions.qwen_vl_encoder,
        input=Input.Connection,
        title="Qwen VL Encoder",
    )

    def invoke(self, context: InvocationContext) -> QwenImageLoRALoaderOutput:
        output = QwenImageLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)

        if self.qwen_vl_encoder is not None:
            output.qwen_vl_encoder = self.qwen_vl_encoder.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue
            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            added_loras.append(lora.lora.key)
            _warn_if_encoder_layers_are_unused(
                context, lora.lora, transformer_only=self.transformer is not None and self.qwen_vl_encoder is None
            )

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)
            if self.qwen_vl_encoder is not None and output.qwen_vl_encoder is not None:
                output.qwen_vl_encoder.loras.append(lora)

        return output
