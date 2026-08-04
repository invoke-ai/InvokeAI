from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    MiniMaxH3TextEncoderField,
    MiniMaxH3TransformerField,
    ModelIdentifierField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("minimax_h3_model_loader_output")
class MiniMaxH3ModelLoaderOutput(BaseInvocationOutput):
    """MiniMax H3 model loader output."""

    transformer: MiniMaxH3TransformerField = OutputField(
        description="MiniMax H3 FL2VA transformer", title="Transformer"
    )
    text_encoder: MiniMaxH3TextEncoderField = OutputField(
        description=FieldDescriptions.minimax_h3_text_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="Video VAE")
    audio_vae: VAEField = OutputField(description=FieldDescriptions.minimax_h3_audio_vae, title="Audio VAE")


@invocation(
    "minimax_h3_model_loader",
    title="Main Model - MiniMax H3",
    tags=["model", "minimax", "video"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3ModelLoaderInvocation(BaseInvocation):
    """Loads a MiniMax H3 (FL2VA) model, outputting its submodels.

    All six submodels (transformer, text encoder, tokenizer, processor, video VAE, audio VAE)
    come from the one diffusers-layout install; there is no component mix-and-match yet.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.minimax_h3_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.Main,
        title="Model",
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3ModelLoaderOutput:
        if not context.models.exists(self.model.key):
            raise ValueError(f"Unknown model: {self.model.key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        processor = self.model.model_copy(update={"submodel_type": SubModelType.Processor})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        audio_vae = self.model.model_copy(update={"submodel_type": SubModelType.AudioVAE})

        return MiniMaxH3ModelLoaderOutput(
            transformer=MiniMaxH3TransformerField(transformer=transformer),
            text_encoder=MiniMaxH3TextEncoderField(tokenizer=tokenizer, processor=processor, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
            audio_vae=VAEField(vae=audio_vae),
        )
