"""Flux2 Klein Model Loader Invocation.

Loads a Flux2 Klein model with its Qwen3 text encoder and VAE.
Unlike standard FLUX which uses CLIP+T5, Klein uses only Qwen3.
"""

from typing import Literal, Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    ModelIdentifierField,
    Qwen3EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    ModelFormat,
    ModelType,
    Qwen3VariantType,
    SubModelType,
)

# FLUX.2 Klein variant -> the Qwen3 encoder variant it was trained against, and the single place
# that relationship is written down on the backend. Mirrors `KLEIN_TO_QWEN3_VARIANT_MAP` in
# `invokeai/frontend/web/src/features/parameters/util/flux2Klein.ts` — keep the two in sync.
# Variants sharing a Qwen3 entry (`klein_9b` and `klein_9b_base`) are valid encoder sources for
# each other. [dev] is deliberately absent: it uses a Mistral encoder, so it can never satisfy a
# Klein transformer, and `.get()` returning None is what makes the guards below fail closed.
_KLEIN_TO_QWEN3_VARIANT: dict[Flux2VariantType, Qwen3VariantType] = {
    Flux2VariantType.Klein4B: Qwen3VariantType.Qwen3_4B,
    Flux2VariantType.Klein4BBase: Qwen3VariantType.Qwen3_4B,
    Flux2VariantType.Klein9B: Qwen3VariantType.Qwen3_8B,
    Flux2VariantType.Klein9BBase: Qwen3VariantType.Qwen3_8B,
}


@invocation_output("flux2_klein_model_loader_output")
class Flux2KleinModelLoaderOutput(BaseInvocationOutput):
    """Flux2 Klein model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="The max sequence length for the Qwen3 encoder.",
        title="Max Seq Length",
    )


@invocation(
    "flux2_klein_model_loader",
    title="Main Model - Flux2 Klein",
    tags=["model", "flux", "klein", "qwen3"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2KleinModelLoaderInvocation(BaseInvocation):
    """Loads a Flux2 Klein model, outputting its submodels.

    Flux2 Klein uses Qwen3 as the text encoder instead of CLIP+T5.
    It uses a 32-channel VAE (AutoencoderKLFlux2) instead of the 16-channel FLUX.1 VAE.

    When using a Diffusers format model, both VAE and Qwen3 encoder are extracted
    automatically from the main model. You can override with standalone models:
    - Transformer: Always from Flux2 Klein main model
    - VAE: From main model (Diffusers) or standalone VAE
    - Qwen3 Encoder: From main model (Diffusers) or standalone Qwen3 model
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. Flux2 Klein uses the same VAE as FLUX (16-channel). "
        "If not provided, VAE will be loaded from the Qwen3 Source model.",
        input=Input.Direct,
        ui_model_base=[BaseModelType.Flux, BaseModelType.Flux2],
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3 Encoder model. "
        "If not provided, encoder will be loaded from the Qwen3 Source model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3Encoder,
        title="Qwen3 Encoder",
    )

    qwen3_source_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Flux2 Klein model to extract VAE and/or Qwen3 encoder from. "
        "Use this if you don't have separate VAE/Qwen3 models. "
        "Ignored if both VAE and Qwen3 Encoder are provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Qwen3 Source (Diffusers)",
    )

    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Qwen3 encoder.",
        title="Max Seq Length",
    )

    def invoke(self, context: InvocationContext) -> Flux2KleinModelLoaderOutput:
        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Check if main model is Diffusers format (can extract VAE directly)
        main_config = context.models.get_config(self.model)
        main_is_diffusers = main_config.format == ModelFormat.Diffusers

        # Determine VAE source
        # IMPORTANT: FLUX.2 Klein uses a 32-channel VAE (AutoencoderKLFlux2), not the 16-channel FLUX.1 VAE.
        # The VAE should come from the FLUX.2 Klein Diffusers model, not a separate FLUX VAE.
        if self.vae_model is not None:
            # Use standalone VAE (user explicitly selected one)
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            # Extract VAE from main model (recommended for FLUX.2)
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.qwen3_source_model is not None:
            # Extract from Qwen3 source Diffusers model
            self._validate_diffusers_format(context, self.qwen3_source_model, "Qwen3 Source")
            vae = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Standalone safetensors/GGUF models require a separate VAE. "
                "Options:\n"
                "  1. Set 'VAE' to a standalone FLUX VAE model\n"
                "  2. Set 'Qwen3 Source' to a Diffusers Flux2 Klein model to extract the VAE from"
            )

        # Determine Qwen3 Encoder source
        if self.qwen3_encoder_model is not None:
            # Use standalone Qwen3 Encoder - validate it matches the FLUX.2 Klein variant
            self._validate_qwen3_encoder_variant(context, main_config)
            qwen3_tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif main_is_diffusers:
            # Extract from main model (recommended for FLUX.2 Klein)
            qwen3_tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.qwen3_source_model is not None:
            # Extract from separate Diffusers model
            self._validate_encoder_source(context, self.qwen3_source_model, "Qwen3 Source", main_config)
            qwen3_tokenizer = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Qwen3 Encoder source provided. Standalone safetensors/GGUF models require a separate text encoder. "
                "Options:\n"
                "  1. Set 'Qwen3 Encoder' to a standalone Qwen3 text encoder model "
                "(Klein 4B needs Qwen3 4B, Klein 9B needs Qwen3 8B)\n"
                "  2. Set 'Qwen3 Source' to a Diffusers Flux2 Klein model to extract the encoder from"
            )

        return Flux2KleinModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
            max_seq_len=self.max_seq_len,
        )

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> AnyModelConfig:
        """Validate that a model is a Diffusers-format pipeline and return its config.

        Deliberately format-only, because this also gates the VAE-extraction path: the 32-channel
        ``AutoencoderKLFlux2`` is shared between Klein and [dev], and the linear UI relies on that
        (``buildFLUXGraph`` falls back to *any* FLUX.2 diffusers pipeline when only the VAE is
        needed). Variant gating belongs to the encoder path only — see ``_validate_encoder_source``.
        """
        config = context.models.get_config(model)
        if config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The {model_name} model must be a Diffusers format model. "
                f"The selected model '{config.name}' is in {config.format.value} format."
            )
        return config

    def _validate_encoder_source(
        self,
        context: InvocationContext,
        model: ModelIdentifierField,
        model_name: str,
        main_config: AnyModelConfig,
    ) -> None:
        """Validate a Diffusers pipeline used as the *text encoder* source.

        The source's tokenizer + encoder are extracted and paired with *this* model's transformer,
        so they must come from the same Qwen3 family. Mismatched widths produce conditioning that
        only fails as an opaque matmul error deep in denoise, so reject it here where the user still
        gets a clear message. The linear UI (``buildFLUXGraph``) and the standalone-encoder path
        (``_validate_qwen3_encoder_variant``) already enforce the family match; the workflow editor
        lets any FLUX.2 Diffusers pipeline be wired in here, so this is the entry point that closes it.
        """
        config = self._validate_diffusers_format(context, model, model_name)
        source_variant = getattr(config, "variant", None)
        source_qwen3 = _KLEIN_TO_QWEN3_VARIANT.get(source_variant)

        # An allowlist, not "reject [dev]": a future third FLUX.2 variant has to fail closed here
        # the way the [dev] loader's guard already makes it, rather than being silently accepted.
        if source_qwen3 is None:
            described = f"variant '{source_variant.value}'" if source_variant is not None else "not a Klein pipeline"
            raise ValueError(
                f"The {model_name} model must be a FLUX.2 Klein pipeline, "
                f"but the selected model '{config.name}' is {described}. "
                "Its text encoder is incompatible with the Klein transformer. "
                "(Its VAE is compatible - this only blocks encoder extraction.)"
            )

        required_qwen3 = _KLEIN_TO_QWEN3_VARIANT.get(getattr(main_config, "variant", None))
        if required_qwen3 is not None and source_qwen3 != required_qwen3:
            raise ValueError(
                f"Qwen3 encoder variant mismatch: FLUX.2 Klein {main_config.variant.value} requires a "
                f"{required_qwen3.value} encoder, but the {model_name} pipeline '{config.name}' "
                f"({source_variant.value}) carries {source_qwen3.value}. "
                "Select a Klein pipeline from the same family - 4B pairs with 4B, 9B with 9B."
            )

    def _validate_qwen3_encoder_variant(self, context: InvocationContext, main_config: AnyModelConfig) -> None:
        """Validate that the standalone Qwen3 encoder variant matches the FLUX.2 Klein variant.

        - FLUX.2 Klein 4B (and 4B Base) require the Qwen3 4B encoder
        - FLUX.2 Klein 9B (and 9B Base) require the Qwen3 8B encoder
        """
        if self.qwen3_encoder_model is None:
            return

        # `getattr(..., None)` rather than `hasattr`: the field can exist and still be None, and
        # comparing None against the expected variant would then raise `AttributeError` on
        # `.value` in the error path instead of the intended `ValueError`.
        qwen3_variant = getattr(context.models.get_config(self.qwen3_encoder_model), "variant", None)
        if qwen3_variant is None:
            return

        expected_qwen3_variant = _KLEIN_TO_QWEN3_VARIANT.get(getattr(main_config, "variant", None))
        if expected_qwen3_variant is not None and qwen3_variant != expected_qwen3_variant:
            raise ValueError(
                f"Qwen3 encoder variant mismatch: FLUX.2 Klein {main_config.variant.value} requires "
                f"{expected_qwen3_variant.value} encoder, but {qwen3_variant.value} was selected. "
                "Please select a matching Qwen3 encoder or use a Diffusers format model which includes the correct encoder."
            )
