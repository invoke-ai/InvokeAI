from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import CLIPField, ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import BaseModelType, ModelType, SubModelType


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

    # TODO(ryand): Create a UIType.Sd3MainModelField to use here.
    # model: ModelIdentifierField = InputField(
    #     description=FieldDescriptions.sd3_model,
    #     ui_type=UIType.SD3MainModel,
    #     input=Input.Direct,
    # )
    model: str = InputField(
        description=FieldDescriptions.sd3_model,
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> Sd3ModelLoaderOutput:
        # model_key = self.model.key
        # if not context.models.exists(model_key):
        #     raise ValueError(f"Unknown model: {model_key}")

        # mmditx = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        # vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        # tokenizer_l = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        # clip_encoder_l = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        # tokenizer_g = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        # clip_encoder_g = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        # tokenizer_t5 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
        # t5_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder3})

        # return Sd3ModelLoaderOutput(
        #     mmditx=TransformerField(transformer=mmditx, loras=[]),
        #     clip_l=CLIPField(tokenizer=tokenizer_l, text_encoder=clip_encoder_l, loras=[], skipped_layers=0),
        #     clip_g=CLIPField(tokenizer=tokenizer_g, text_encoder=clip_encoder_g, loras=[], skipped_layers=0),
        #     t5_encoder=T5EncoderField(tokenizer=tokenizer_t5, text_encoder=t5_encoder),
        #     vae=VAEField(vae=vae),
        # )

        model_configs = context.models.search_by_attrs(
            name=self.model, base=BaseModelType.StableDiffusion3, type=ModelType.Main
        )

        if len(model_configs) != 1:
            raise ValueError(f"Expected 1 model config, got {len(model_configs)}")

        model_config = model_configs[0]
        model_identifier = ModelIdentifierField.from_config(model_config)

        mmditx = model_identifier.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = model_identifier.model_copy(update={"submodel_type": SubModelType.VAE})
        tokenizer_l = model_identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        clip_encoder_l = model_identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer_g = model_identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        clip_encoder_g = model_identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        tokenizer_t5 = model_identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
        t5_encoder = model_identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder3})

        return Sd3ModelLoaderOutput(
            mmditx=TransformerField(transformer=mmditx, loras=[]),
            clip_l=CLIPField(tokenizer=tokenizer_l, text_encoder=clip_encoder_l, loras=[], skipped_layers=0),
            clip_g=CLIPField(tokenizer=tokenizer_g, text_encoder=clip_encoder_g, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer_t5, text_encoder=t5_encoder),
            vae=VAEField(vae=vae),
        )
