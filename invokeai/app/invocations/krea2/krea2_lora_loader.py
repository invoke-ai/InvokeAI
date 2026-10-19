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

        stored_config = context.models.get_config(lora_key)
        if (
            self.lora.base is not BaseModelType.Krea2
            or stored_config.base is not BaseModelType.Krea2
            or stored_config.type is not ModelType.LoRA
        ):
            raise ValueError(
                f"LoRA '{lora_key}' is for {stored_config.base.value if stored_config.base else 'unknown'} models, "
                "not Krea-2 models. Ensure you are using a Krea-2 compatible LoRA."
            )

        output = Krea2LoRALoaderOutput()

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
        if self.qwen3_vl_encoder is not None:
            output.qwen3_vl_encoder = self.qwen3_vl_encoder.model_copy(deep=True)

        transformer_lora = (
            next((item for item in output.transformer.loras if item.lora.key == lora_key), None)
            if output.transformer is not None
            else None
        )
        encoder_lora = (
            next((item for item in output.qwen3_vl_encoder.loras if item.lora.key == lora_key), None)
            if output.qwen3_vl_encoder is not None
            else None
        )
        if transformer_lora is not None and encoder_lora is not None and transformer_lora.weight != encoder_lora.weight:
            raise ValueError(
                f"LoRA '{lora_key}' has conflicting weights on the transformer ({transformer_lora.weight}) and "
                f"Qwen3-VL encoder ({encoder_lora.weight})."
            )
        effective_lora = transformer_lora or encoder_lora or LoRAField(lora=self.lora, weight=self.weight)

        if output.transformer is not None and transformer_lora is None:
            output.transformer.loras.append(effective_lora.model_copy(deep=True))
        if output.qwen3_vl_encoder is not None and encoder_lora is None:
            output.qwen3_vl_encoder.loras.append(effective_lora.model_copy(deep=True))

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
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
        if self.qwen3_vl_encoder is not None:
            output.qwen3_vl_encoder = self.qwen3_vl_encoder.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if not context.models.exists(lora.lora.key):
                raise ValueError(f"Unknown lora: {lora.lora.key}!")
            stored_config = context.models.get_config(lora.lora.key)
            if (
                lora.lora.base is not BaseModelType.Krea2
                or stored_config.base is not BaseModelType.Krea2
                or stored_config.type is not ModelType.LoRA
            ):
                raise ValueError(
                    f"LoRA '{lora.lora.key}' is for "
                    f"{stored_config.base.value if stored_config.base else 'unknown'} models, "
                    "not Krea-2 models. Ensure you are using a Krea-2 compatible LoRA."
                )

            transformer_lora = (
                next((item for item in output.transformer.loras if item.lora.key == lora.lora.key), None)
                if output.transformer is not None
                else None
            )
            encoder_lora = (
                next((item for item in output.qwen3_vl_encoder.loras if item.lora.key == lora.lora.key), None)
                if output.qwen3_vl_encoder is not None
                else None
            )
            if (
                transformer_lora is not None
                and encoder_lora is not None
                and transformer_lora.weight != encoder_lora.weight
            ):
                raise ValueError(
                    f"LoRA '{lora.lora.key}' has conflicting weights on the transformer "
                    f"({transformer_lora.weight}) and Qwen3-VL encoder ({encoder_lora.weight})."
                )
            effective_lora = transformer_lora or encoder_lora or lora

            if self.transformer is not None and output.transformer is not None:
                if transformer_lora is None:
                    output.transformer.loras.append(effective_lora.model_copy(deep=True))
            if self.qwen3_vl_encoder is not None and output.qwen3_vl_encoder is not None:
                if encoder_lora is None:
                    output.qwen3_vl_encoder.loras.append(effective_lora.model_copy(deep=True))

        return output
