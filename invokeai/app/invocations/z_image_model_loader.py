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
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


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
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImageModelLoaderInvocation(BaseInvocation):
    """Loads a Z-Image base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.z_image_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.ZImage,
        ui_model_type=ModelType.Main,
    )

    def invoke(self, context: InvocationContext) -> ZImageModelLoaderOutput:
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        qwen3_tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        qwen3_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return ZImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
        )
