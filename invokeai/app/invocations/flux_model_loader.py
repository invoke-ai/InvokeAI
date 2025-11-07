from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import CLIPField, ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.flux.util import get_flux_max_seq_length
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("flux_model_loader_output")
class FluxModelLoaderOutput(BaseInvocationOutput):
    """Flux base model loader output"""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="The max sequence length to used for the T5 encoder. (256 for schnell transformer, 512 for dev transformer)",
        title="Max Seq Length",
    )


@invocation(
    "flux_model_loader",
    title="Main Model - FLUX",
    tags=["model", "flux"],
    category="model",
    version="1.0.6",
)
class FluxModelLoaderInvocation(BaseInvocation):
    """Loads a flux base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.Main,
    )

    t5_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.t5_encoder,
        input=Input.Direct,
        title="T5 Encoder",
        ui_model_type=ModelType.T5Encoder,
    )

    clip_embed_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.clip_embed_model,
        input=Input.Direct,
        title="CLIP Embed",
        ui_model_type=ModelType.CLIPEmbed,
    )

    vae_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.vae_model,
        title="VAE",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> FluxModelLoaderOutput:
        for key in [self.model.key, self.t5_encoder_model.key, self.clip_embed_model.key, self.vae_model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        tokenizer = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        clip_encoder = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        tokenizer2 = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
        t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)

        transformer_config = context.models.get_config(transformer)
        assert isinstance(transformer_config, Checkpoint_Config_Base)

        return FluxModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            clip=CLIPField(tokenizer=tokenizer, text_encoder=clip_encoder, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer2, text_encoder=t5_encoder, loras=[]),
            vae=VAEField(vae=vae),
            max_seq_len=get_flux_max_seq_length(transformer_config.variant),
        )
