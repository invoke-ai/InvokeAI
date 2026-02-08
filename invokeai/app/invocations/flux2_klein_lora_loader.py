"""FLUX.2 Klein LoRA Loader Invocation.

Applies LoRA models to a FLUX.2 Klein transformer and/or Qwen3 text encoder.
Unlike standard FLUX which uses CLIP+T5, Klein uses only Qwen3 for text encoding.
"""

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


@invocation_output("flux2_klein_lora_loader_output")
class Flux2KleinLoRALoaderOutput(BaseInvocationOutput):
    """FLUX.2 Klein LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    qwen3_encoder: Optional[Qwen3EncoderField] = OutputField(
        default=None, description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder"
    )


@invocation(
    "flux2_klein_lora_loader",
    title="Apply LoRA - Flux2 Klein",
    tags=["lora", "model", "flux", "klein", "flux2"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2KleinLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a FLUX.2 Klein transformer and/or Qwen3 text encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
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

    def invoke(self, context: InvocationContext) -> Flux2KleinLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        # Warn if LoRA variant doesn't match transformer variant
        lora_config = context.models.get_config(lora_key)
        lora_variant = getattr(lora_config, "variant", None)
        if lora_variant and self.transformer is not None:
            transformer_config = context.models.get_config(self.transformer.transformer.key)
            transformer_variant = getattr(transformer_config, "variant", None)
            if transformer_variant and lora_variant != transformer_variant:
                context.logger.warning(
                    f"LoRA variant mismatch: LoRA '{lora_config.name}' is for {lora_variant.value} "
                    f"but transformer is {transformer_variant.value}. This may cause shape errors."
                )

        # Check for existing LoRAs with the same key.
        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.qwen3_encoder and any(lora.lora.key == lora_key for lora in self.qwen3_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to Qwen3 encoder.')

        output = Flux2KleinLoRALoaderOutput()

        # Attach LoRA layers to the models.
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
    "flux2_klein_lora_collection_loader",
    title="Apply LoRA Collection - Flux2 Klein",
    tags=["lora", "model", "flux", "klein", "flux2"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2KleinLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a FLUX.2 Klein transformer and/or Qwen3 text encoder."""

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

    def invoke(self, context: InvocationContext) -> Flux2KleinLoRALoaderOutput:
        output = Flux2KleinLoRALoaderOutput()
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
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            assert lora.lora.base in (BaseModelType.Flux, BaseModelType.Flux2)

            # Warn if LoRA variant doesn't match transformer variant
            lora_config = context.models.get_config(lora.lora.key)
            lora_variant = getattr(lora_config, "variant", None)
            if lora_variant and self.transformer is not None:
                transformer_config = context.models.get_config(self.transformer.transformer.key)
                transformer_variant = getattr(transformer_config, "variant", None)
                if transformer_variant and lora_variant != transformer_variant:
                    context.logger.warning(
                        f"LoRA variant mismatch: LoRA '{lora_config.name}' is for {lora_variant.value} "
                        f"but transformer is {transformer_variant.value}. This may cause shape errors."
                    )

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)

            if self.qwen3_encoder is not None and output.qwen3_encoder is not None:
                output.qwen3_encoder.loras.append(lora)

        return output
