"""FLUX.2 [dev] model loader invocation.

Loads a FLUX.2 [dev] transformer with its Mistral Small 3.1 text encoder and the
shared FLUX.2 32-channel VAE.
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
    MistralEncoderField,
    ModelIdentifierField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@invocation_output("flux2_dev_model_loader_output")
class Flux2DevModelLoaderOutput(BaseInvocationOutput):
    """FLUX.2 [dev] model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    mistral_encoder: MistralEncoderField = OutputField(
        description=FieldDescriptions.mistral_encoder, title="Mistral Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="Max sequence length for the Mistral encoder.",
        title="Max Seq Length",
    )


@invocation(
    "flux2_dev_model_loader",
    title="Main Model - FLUX.2 [dev]",
    tags=["model", "flux", "flux2", "dev", "mistral"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevModelLoaderInvocation(BaseInvocation):
    """Load a FLUX.2 [dev] transformer plus its Mistral text encoder and VAE.

    FLUX.2 [dev] is a 32B guidance-distilled rectified flow transformer that uses
    Mistral Small 3.1 (24B) as its sole text encoder, sharing the 32-channel
    AutoencoderKLFlux2 VAE with FLUX.2 Klein.

    When the transformer is a Diffusers-format checkpoint, both VAE and Mistral
    encoder can be extracted directly from the main model. For single-file
    safetensors or GGUF transformers, you must supply standalone VAE and
    Mistral encoder models, or point at a Diffusers FLUX.2 [dev] checkout for
    sub-model extraction.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux2_dev_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone FLUX.2 VAE (AutoencoderKLFlux2). "
        "If not provided, the VAE is extracted from the Diffusers source model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    mistral_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Mistral text encoder. Required when the transformer is "
        "a single-file safetensors or GGUF without a sibling Diffusers source.",
        input=Input.Direct,
        ui_model_type=ModelType.MistralEncoder,
        title="Mistral Encoder",
    )

    mistral_source_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers FLUX.2 [dev] model to extract VAE and/or Mistral encoder from. "
        "Use this if you don't have separate VAE / Mistral encoder models. "
        "Ignored if both are provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Mistral Source (Diffusers)",
    )

    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Mistral encoder. FLUX.2 [dev] uses 512 by default.",
        title="Max Seq Length",
    )

    def invoke(self, context: InvocationContext) -> Flux2DevModelLoaderOutput:
        # Validate the selected main model is FLUX.2 [dev], not Klein.
        main_config = context.models.get_config(self.model)
        variant = getattr(main_config, "variant", None)
        if variant is not None and variant != Flux2VariantType.Dev:
            raise ValueError(
                f"FLUX.2 [dev] loader requires a FLUX.2 [dev] transformer, "
                f"but the selected model is variant '{variant.value}'. "
                "Use the FLUX.2 Klein loader for Klein variants."
            )

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        main_is_diffusers = main_config.format == ModelFormat.Diffusers

        # Resolve VAE.
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.mistral_source_model is not None:
            self._validate_diffusers_format(context, self.mistral_source_model, "Mistral Source")
            vae = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Single-file / GGUF transformers require a separate VAE. "
                "Options:\n"
                "  1. Set 'VAE' to a standalone FLUX.2 VAE model\n"
                "  2. Set 'Mistral Source' to a Diffusers FLUX.2 [dev] model to extract the VAE from"
            )

        # Resolve Mistral encoder.
        if self.mistral_encoder_model is not None:
            tokenizer = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif main_is_diffusers:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.mistral_source_model is not None:
            self._validate_diffusers_format(context, self.mistral_source_model, "Mistral Source")
            tokenizer = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Mistral encoder source provided. Single-file / GGUF transformers require a separate "
                "text encoder. Options:\n"
                "  1. Set 'Mistral Encoder' to a standalone Mistral Small 3.1 text encoder model\n"
                "  2. Set 'Mistral Source' to a Diffusers FLUX.2 [dev] model to extract the encoder from"
            )

        return Flux2DevModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            mistral_encoder=MistralEncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
            max_seq_len=self.max_seq_len,
        )

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> None:
        config = context.models.get_config(model)
        if config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The {model_name} model must be a Diffusers format model. "
                f"The selected model '{config.name}' is in {config.format.value} format."
            )
