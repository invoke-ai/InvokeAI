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
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    ModelFormat,
    ModelType,
    Qwen3VariantType,
    SubModelType,
)


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
                "No VAE source provided. For FLUX.2 Klein, the VAE should come from the Diffusers model. "
                "Either use a Diffusers format main model, or set 'Qwen3 Source' to a Diffusers Flux2 Klein model."
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
            self._validate_diffusers_format(context, self.qwen3_source_model, "Qwen3 Source")
            qwen3_tokenizer = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Qwen3 Encoder source provided. For FLUX.2 Klein, the Qwen3 encoder should come from the Diffusers model. "
                "Either use a Diffusers format main model, or set 'Qwen3 Source' to a Diffusers Flux2 Klein model."
            )

        return Flux2KleinModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
            max_seq_len=self.max_seq_len,
        )

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> None:
        """Validate that a model is in Diffusers format."""
        config = context.models.get_config(model)
        if config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The {model_name} model must be a Diffusers format model. "
                f"The selected model '{config.name}' is in {config.format.value} format."
            )

    def _validate_qwen3_encoder_variant(self, context: InvocationContext, main_config) -> None:
        """Validate that the standalone Qwen3 encoder variant matches the FLUX.2 Klein variant.

        - FLUX.2 Klein 4B requires Qwen3 4B encoder
        - FLUX.2 Klein 9B requires Qwen3 8B encoder
        """
        if self.qwen3_encoder_model is None:
            return

        # Get the Qwen3 encoder config
        qwen3_config = context.models.get_config(self.qwen3_encoder_model)

        # Check if the config has a variant field
        if not hasattr(qwen3_config, "variant"):
            # Can't validate, skip
            return

        qwen3_variant = qwen3_config.variant

        # Get the FLUX.2 Klein variant from the main model config
        if not hasattr(main_config, "variant"):
            return

        flux2_variant = main_config.variant

        # Validate the variants match
        # Klein4B requires Qwen3_4B, Klein9B requires Qwen3_8B
        expected_qwen3_variant = None
        if flux2_variant == Flux2VariantType.Klein4B:
            expected_qwen3_variant = Qwen3VariantType.Qwen3_4B
        elif flux2_variant == Flux2VariantType.Klein9B:
            expected_qwen3_variant = Qwen3VariantType.Qwen3_8B

        if expected_qwen3_variant is not None and qwen3_variant != expected_qwen3_variant:
            raise ValueError(
                f"Qwen3 encoder variant mismatch: FLUX.2 Klein {flux2_variant.value} requires "
                f"{expected_qwen3_variant.value} encoder, but {qwen3_variant.value} was selected. "
                f"Please select a matching Qwen3 encoder or use a Diffusers format model which includes the correct encoder."
            )
