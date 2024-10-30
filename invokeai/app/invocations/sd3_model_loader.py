from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import CLIPField, ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import SubModelType


@invocation_output("sd3_model_loader_output")
class Sd3ModelLoaderOutput(BaseInvocationOutput):
    """SD3 base model loader output."""

    mmditx: TransformerField = OutputField(description=FieldDescriptions.mmditx, title="MMDiTX")
    clip_l: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP L")
    clip_g: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP G")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "sd3_model_loader",
    title="SD3 Main Model",
    tags=["model", "sd3"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Sd3ModelLoaderInvocation(BaseInvocation):
    """Loads a SD3 base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.sd3_model,
        ui_type=UIType.SD3MainModel,
        input=Input.Direct,
    )

    t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.t5_encoder,
        ui_type=UIType.T5EncoderModel,
        input=Input.Direct,
        title="T5 Encoder",
        default=None,
    )

    # TODO(brandon): Setup UI updates to support selecting a clip l model.
    # clip_l_model: ModelIdentifierField = InputField(
    #     description=FieldDescriptions.clip_l_model,
    #     ui_type=UIType.CLIPEmbedModel,
    #     input=Input.Direct,
    #     title="CLIP L Encoder",
    # )

    # TODO(brandon): Setup UI updates to support selecting a clip g model.
    # clip_g_model: ModelIdentifierField = InputField(
    #     description=FieldDescriptions.clip_g_model,
    #     ui_type=UIType.CLIPGModel,
    #     input=Input.Direct,
    #     title="CLIP G Encoder",
    # )

    # TODO(brandon): Setup UI updates to support selecting an SD3 vae model.
    # vae_model: ModelIdentifierField = InputField(
    #     description=FieldDescriptions.vae_model, ui_type=UIType.FluxVAEModel, title="VAE", default=None
    # )

    def invoke(self, context: InvocationContext) -> Sd3ModelLoaderOutput:
        mmditx = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        tokenizer_l = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        clip_encoder_l = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer_g = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        clip_encoder_g = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        tokenizer_t5 = (
            self.t5_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
            if self.t5_encoder_model
            else self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
        )
        t5_encoder = (
            self.t5_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder3})
            if self.t5_encoder_model
            else self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder3})
        )

        return Sd3ModelLoaderOutput(
            mmditx=TransformerField(transformer=mmditx, loras=[]),
            clip_l=CLIPField(tokenizer=tokenizer_l, text_encoder=clip_encoder_l, loras=[], skipped_layers=0),
            clip_g=CLIPField(tokenizer=tokenizer_g, text_encoder=clip_encoder_g, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer_t5, text_encoder=t5_encoder),
            vae=VAEField(vae=vae),
        )