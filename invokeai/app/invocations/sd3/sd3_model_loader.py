from typing import Optional

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
from invokeai.backend.architectures import accepted_vae_bases
from invokeai.backend.model_manager.taxonomy import BaseModelType, ClipVariantType, ModelFormat, ModelType, SubModelType


@invocation_output("sd3_model_loader_output")
class Sd3ModelLoaderOutput(BaseInvocationOutput):
    """SD3 base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    clip_l: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP L")
    clip_g: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP G")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "sd3_model_loader",
    title="Main Model - SD3",
    tags=["model", "sd3"],
    category="model",
    version="1.0.1",
)
class Sd3ModelLoaderInvocation(BaseInvocation):
    """Loads a SD3 base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.sd3_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.StableDiffusion3,
        ui_model_type=ModelType.Main,
    )

    t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.t5_encoder,
        input=Input.Direct,
        title="T5 Encoder",
        default=None,
        ui_model_type=ModelType.T5Encoder,
    )

    clip_l_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.clip_embed_model,
        input=Input.Direct,
        title="CLIP L Encoder",
        default=None,
        ui_model_type=ModelType.CLIPEmbed,
        ui_model_variant=ClipVariantType.L,
    )

    clip_g_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.clip_g_model,
        input=Input.Direct,
        title="CLIP G Encoder",
        default=None,
        ui_model_type=ModelType.CLIPEmbed,
        ui_model_variant=ClipVariantType.G,
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.vae_model,
        title="VAE",
        default=None,
        ui_model_base=accepted_vae_bases(BaseModelType.StableDiffusion3),
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> Sd3ModelLoaderOutput:
        if context.models.get_config(self.model).format is not ModelFormat.Diffusers:
            return self._from_components()

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = (
            self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
            if self.vae_model
            else self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        )
        tokenizer_l = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        clip_encoder_l = (
            self.clip_l_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
            if self.clip_l_model
            else self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        )
        tokenizer_g = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        # A separately selected CLIP-G is a model of its own, whose encoder is its `TextEncoder`; `TextEncoder2`
        # names CLIP-G only inside the SD 3 pipeline.
        clip_encoder_g = (
            self.clip_g_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
            if self.clip_g_model
            else self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        )
        tokenizer_t5 = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model or self.model)
        t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model or self.model)

        return Sd3ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            clip_l=CLIPField(tokenizer=tokenizer_l, text_encoder=clip_encoder_l, loras=[], skipped_layers=0),
            clip_g=CLIPField(tokenizer=tokenizer_g, text_encoder=clip_encoder_g, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer_t5, text_encoder=t5_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )

    def _from_components(self) -> Sd3ModelLoaderOutput:
        """For a main that holds only the transformer (a GGUF): every other part, each tokenizer included, comes
        from its own selected component."""
        t5, clip_l, clip_g, vae = self.t5_encoder_model, self.clip_l_model, self.clip_g_model, self.vae_model
        if t5 is None or clip_l is None or clip_g is None or vae is None:
            missing = [
                title
                for title, selected in (
                    ("T5 Encoder", t5),
                    ("CLIP L Encoder", clip_l),
                    ("CLIP G Encoder", clip_g),
                    ("VAE", vae),
                )
                if selected is None
            ]
            listed = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
            raise ValueError(
                f"'{self.model.name}' holds only the SD3 transformer, so its {listed} must be selected on "
                "Main Model - SD3. Only a Diffusers SD3 model brings its own text encoders and VAE."
            )

        def part(model: ModelIdentifierField, submodel_type: SubModelType) -> ModelIdentifierField:
            return model.model_copy(update={"submodel_type": submodel_type})

        return Sd3ModelLoaderOutput(
            transformer=TransformerField(transformer=part(self.model, SubModelType.Transformer), loras=[]),
            clip_l=CLIPField(
                tokenizer=part(clip_l, SubModelType.Tokenizer),
                text_encoder=part(clip_l, SubModelType.TextEncoder),
                loras=[],
                skipped_layers=0,
            ),
            clip_g=CLIPField(
                tokenizer=part(clip_g, SubModelType.Tokenizer),
                text_encoder=part(clip_g, SubModelType.TextEncoder),
                loras=[],
                skipped_layers=0,
            ),
            t5_encoder=T5EncoderField(
                tokenizer=preprocess_t5_tokenizer_model_identifier(t5),
                text_encoder=preprocess_t5_encoder_model_identifier(t5),
                loras=[],
            ),
            vae=VAEField(vae=part(vae, SubModelType.VAE)),
        )
