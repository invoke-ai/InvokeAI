"""FLUX.2 [dev] LoRA loader invocations.

Mirror of the Klein LoRA loader, but routes encoder LoRAs to the Mistral text
encoder rather than the Qwen3 encoder.
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
from invokeai.app.invocations.model import (
    LoRAField,
    MistralEncoderField,
    ModelIdentifierField,
    TransformerField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType, ModelType


def _assert_dev_lora(context: InvocationContext, lora_config) -> None:
    """Reject a non-dev FLUX.2 LoRA applied via the FLUX.2 [dev] loaders.

    A Klein LoRA (hidden 3072/4096) applied to a dev transformer/encoder (hidden 5120/6144)
    is guaranteed to raise a shape-mismatch ``RuntimeError`` partway through denoise. Fail
    fast here with an actionable message instead. This is independent of *which* input the
    LoRA is wired to — the mismatch happens on whichever module it patches — so the check
    is not gated on the transformer being connected. The frontend also filters these out
    before they reach the graph (see ``addFlux2DevLoRAs``); this is the backend backstop for
    hand-built workflow graphs.
    """
    lora_variant = getattr(lora_config, "variant", None)
    if lora_variant is not None and lora_variant != Flux2VariantType.Dev:
        raise ValueError(
            f"LoRA '{lora_config.name}' is a {lora_variant.value} LoRA and cannot be applied via the "
            "FLUX.2 [dev] loader. Use the FLUX.2 Klein LoRA loader for Klein LoRAs."
        )


@invocation_output("flux2_dev_lora_loader_output")
class Flux2DevLoRALoaderOutput(BaseInvocationOutput):
    """FLUX.2 [dev] LoRA loader output."""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Transformer"
    )
    mistral_encoder: Optional[MistralEncoderField] = OutputField(
        default=None, description=FieldDescriptions.mistral_encoder, title="Mistral Encoder"
    )


@invocation(
    "flux2_dev_lora_loader",
    title="Apply LoRA - FLUX.2 [dev]",
    tags=["lora", "model", "flux", "flux2", "dev"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA to a FLUX.2 [dev] transformer and/or its Mistral text encoder."""

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
    mistral_encoder: MistralEncoderField | None = InputField(
        default=None,
        title="Mistral Encoder",
        description=FieldDescriptions.mistral_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> Flux2DevLoRALoaderOutput:
        lora_key = self.lora.key
        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        lora_config = context.models.get_config(lora_key)

        # Reject variant-mismatched LoRAs regardless of which input they're wired to. A Klein
        # LoRA on a dev transformer/encoder is guaranteed to shape-error during denoise.
        _assert_dev_lora(context, lora_config)

        # Check for duplicate keys.
        if self.transformer and any(existing.lora.key == lora_key for existing in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.mistral_encoder and any(existing.lora.key == lora_key for existing in self.mistral_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to Mistral encoder.')

        output = Flux2DevLoRALoaderOutput()
        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        if self.mistral_encoder is not None:
            output.mistral_encoder = self.mistral_encoder.model_copy(deep=True)
            output.mistral_encoder.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        return output


@invocation(
    "flux2_dev_lora_collection_loader",
    title="Apply LoRA Collection - FLUX.2 [dev]",
    tags=["lora", "model", "flux", "flux2", "dev"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevLoRACollectionLoader(BaseInvocation):
    """Apply a collection of LoRAs to a FLUX.2 [dev] transformer and/or Mistral encoder."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None,
        description="LoRA models and weights. May be a single LoRA or collection.",
        title="LoRAs",
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    mistral_encoder: MistralEncoderField | None = InputField(
        default=None,
        title="Mistral Encoder",
        description=FieldDescriptions.mistral_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> Flux2DevLoRALoaderOutput:
        output = Flux2DevLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
        if self.mistral_encoder is not None:
            output.mistral_encoder = self.mistral_encoder.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue
            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")
            assert lora.lora.base in (BaseModelType.Flux, BaseModelType.Flux2)

            lora_config = context.models.get_config(lora.lora.key)
            # Reject variant-mismatched LoRAs, matching the single-LoRA loader above.
            _assert_dev_lora(context, lora_config)

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)
            if self.mistral_encoder is not None and output.mistral_encoder is not None:
                output.mistral_encoder.loras.append(lora)

        return output
