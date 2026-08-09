from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField
from invokeai.app.invocations.model import (
    CLIPField,
    ModelIdentifierField,
    T5EncoderField,
    TransformerField,
    VAEField,
    is_self_contained_sdnq_flux1_pipeline,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.flux.util import get_flux_max_seq_length
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.main import Main_SDNQ_Diffusers_FLUX_Config
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
    version="1.1.0",
)
class FluxModelLoaderInvocation(BaseInvocation):
    """Loads a flux base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.Main,
    )

    # The three component inputs are optional: an SDNQ pipeline install ships its own T5, CLIP and
    # VAE, which is what the SDNQ docs promise ("one install pulls everything you need"). Requiring
    # them anyway forced users to install duplicates of components they already had. Single-file /
    # GGUF / BnB models still need them, and are told so explicitly below.
    t5_encoder_model: ModelIdentifierField | None = InputField(
        default=None,
        description=FieldDescriptions.t5_encoder,
        title="T5 Encoder",
        ui_model_type=ModelType.T5Encoder,
    )

    clip_embed_model: ModelIdentifierField | None = InputField(
        default=None,
        description=FieldDescriptions.clip_embed_model,
        title="CLIP Embed",
        ui_model_type=ModelType.CLIPEmbed,
    )

    vae_model: ModelIdentifierField | None = InputField(
        default=None,
        description=FieldDescriptions.vae_model,
        title="VAE",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> FluxModelLoaderOutput:
        keys = [self.model.key] + [
            m.key for m in (self.t5_encoder_model, self.clip_embed_model, self.vae_model) if m is not None
        ]
        for key in keys:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        main_config = context.models.get_config(self.model)
        self_contained = is_self_contained_sdnq_flux1_pipeline(main_config)

        def resolve(selected: ModelIdentifierField | None) -> ModelIdentifierField | None:
            """Explicit selection wins; otherwise the main model supplies the part if it can."""
            if selected is not None:
                return selected
            return self.model if self_contained else None

        t5_source = resolve(self.t5_encoder_model)
        clip_source = resolve(self.clip_embed_model)
        vae_source = resolve(self.vae_model)

        missing = [
            title
            for title, source in (("T5 Encoder", t5_source), ("CLIP Embed", clip_source), ("VAE", vae_source))
            if source is None
        ]
        if missing:
            raise ValueError(
                f"The selected FLUX model does not ship its own {', '.join(missing)}, so "
                f"{'it' if len(missing) == 1 else 'they'} must be selected explicitly. Only a complete "
                "SDNQ pipeline install (transformer + CLIP + T5 + VAE) can supply these itself."
            )

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = vae_source.model_copy(update={"submodel_type": SubModelType.VAE})

        tokenizer = clip_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        clip_encoder = clip_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        if t5_source is self.model:
            # The pipeline's own T5 lives in the slots discovery recorded, not behind the standalone
            # T5 bundle layouts `preprocess_t5_*` exists to normalize.
            tokenizer2 = t5_source.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
            t5_encoder = t5_source.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        else:
            tokenizer2 = preprocess_t5_tokenizer_model_identifier(t5_source)
            t5_encoder = preprocess_t5_encoder_model_identifier(t5_source)

        transformer_config = main_config
        assert isinstance(transformer_config, (Checkpoint_Config_Base, Main_SDNQ_Diffusers_FLUX_Config))

        return FluxModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            clip=CLIPField(tokenizer=tokenizer, text_encoder=clip_encoder, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer2, text_encoder=t5_encoder, loras=[]),
            vae=VAEField(vae=vae),
            max_seq_len=get_flux_max_seq_length(transformer_config.variant),
        )
