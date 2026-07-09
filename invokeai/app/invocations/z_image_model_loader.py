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
    ModelIdentifierField,
    Qwen3EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("z_image_model_loader_output")
class ZImageModelLoaderOutput(BaseInvocationOutput):
    """Z-Image base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "z_image_model_loader",
    title="Main Model - Z-Image",
    tags=["model", "z-image"],
    category="model",
    version="3.0.0",
    classification=Classification.Prototype,
)
class ZImageModelLoaderInvocation(BaseInvocation):
    """Loads a Z-Image model, outputting its submodels.

    Similar to FLUX, you can mix and match components:
    - Transformer: From Z-Image main model (GGUF quantized or Diffusers format)
    - VAE: Separate FLUX VAE (shared with FLUX models) or from a Diffusers Z-Image model
    - Qwen3 Encoder: Separate Qwen3Encoder model or from a Diffusers Z-Image model
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.z_image_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.ZImage,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. Z-Image uses the same VAE as FLUX (16-channel). "
        "If not provided, VAE will be loaded from the Qwen3 Source model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux,
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
        description="Z-Image pipeline model to extract VAE and/or Qwen3 encoder from. "
        "Use this if you don't have separate VAE/Qwen3 models. "
        "Ignored if both VAE and Qwen3 Encoder are provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.ZImage,
        ui_model_type=ModelType.Main,
        # No ui_model_format hint: both plain Diffusers pipelines and SDNQ-quantized ZImagePipeline
        # folders (which ship VAE/Qwen3 submodels) are valid sources. _validate_diffusers_format
        # accepts both, so pinning the format here would wrongly filter SDNQ pipelines out of the
        # generic node/workflow model pickers.
        title="Qwen3 Source",
    )

    def invoke(self, context: InvocationContext) -> ZImageModelLoaderOutput:
        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # SDNQ Z-Image pipeline installs are self-contained: a single install ships the transformer,
        # Qwen3 encoder and VAE as submodels of the main model. When the user hasn't selected a
        # standalone VAE / Qwen3 Encoder or a separate Qwen3 Source, fall back to the main model
        # itself for those submodels. This lets a freshly installed SDNQ Z-Image pipeline generate
        # without the user having to manually re-select the same model as a component source.
        self_contained_source = self._get_self_contained_source(context)

        # Determine VAE source
        if self.vae_model is not None:
            # Use standalone FLUX VAE
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.qwen3_source_model is not None:
            # Extract from Diffusers Z-Image model
            self._validate_diffusers_format(context, self.qwen3_source_model, "Qwen3 Source")
            vae = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self_contained_source is not None:
            # Extract from the self-contained SDNQ main model
            vae = self_contained_source.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Either set 'VAE' to a FLUX VAE model, "
                "or set 'Qwen3 Source' to a Diffusers Z-Image model."
            )

        # Determine Qwen3 Encoder source
        if self.qwen3_encoder_model is not None:
            # Use standalone Qwen3 Encoder
            qwen3_tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.qwen3_source_model is not None:
            # Extract from Diffusers Z-Image model
            self._validate_diffusers_format(context, self.qwen3_source_model, "Qwen3 Source")
            qwen3_tokenizer = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_source_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self_contained_source is not None:
            # Extract from the self-contained SDNQ main model
            qwen3_tokenizer = self_contained_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self_contained_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Qwen3 Encoder source provided. Either set 'Qwen3 Encoder' to a standalone model, "
                "or set 'Qwen3 Source' to a Diffusers Z-Image model."
            )

        return ZImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
        )

    def _get_self_contained_source(self, context: InvocationContext) -> Optional[ModelIdentifierField]:
        """Return the main model as a submodel source when it is a self-contained pipeline that
        ships its own VAE and Qwen3 submodels.

        A truthy ``submodels`` dict is not sufficient: Main_SDNQ_Diffusers_ZImage_Config builds
        whatever submodels it recognizes from model_index.json, so a partial (or partially
        recognized) pipeline can expose e.g. only the transformer. The loader then loads the VAE /
        Qwen3 encoder / tokenizer from fixed ``vae`` / ``text_encoder`` / ``tokenizer`` subfolders, so
        we only treat the main model as self-contained when all of those submodels are present.
        Otherwise we fall through and require an explicit VAE / Qwen3 source."""
        config = context.models.get_config(self.model)
        if config.format != ModelFormat.SDNQQuantized:
            return None
        submodels = getattr(config, "submodels", None) or {}
        required = {SubModelType.VAE, SubModelType.TextEncoder, SubModelType.Tokenizer}
        if required.issubset(submodels.keys()):
            return self.model
        return None

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> None:
        """Validate that a model exposes the diffusers-style submodel layout (transformer / vae /
        text_encoder / tokenizer subfolders). Plain diffusers Z-Image pipelines satisfy this;
        SDNQ-quantized ZImagePipeline folders do too because they ship the same submodels. Single-
        file SDNQ Z-Image checkpoints don't have submodels populated and must still be rejected."""
        config = context.models.get_config(model)
        if config.format == ModelFormat.Diffusers:
            return
        if config.format == ModelFormat.SDNQQuantized and getattr(config, "submodels", None):
            return
        raise ValueError(
            f"The {model_name} model must be a Diffusers-style Z-Image pipeline (with VAE / Qwen3 "
            f"submodels). The selected model '{config.name}' is in {config.format.value} format."
        )
