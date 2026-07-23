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


@invocation_output("ideogram4_model_loader_output")
class Ideogram4ModelLoaderOutput(BaseInvocationOutput):
    """Ideogram 4 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(
        description=FieldDescriptions.qwen3_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "ideogram4_model_loader",
    title="Main Model - Ideogram 4",
    tags=["model", "ideogram4"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4ModelLoaderInvocation(BaseInvocation):
    """Loads an Ideogram 4 model, outputting its submodels.

    Ideogram 4 is distributed as a single bundled diffusers folder, so the transformer
    (both branches), the Qwen3-VL text encoder + tokenizer, and the VAE are all loaded
    from the one selected model.
    """

    model: ModelIdentifierField = InputField(
        description="The Ideogram 4 model to load.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        title="Model",
    )

    def invoke(self, context: InvocationContext) -> Ideogram4ModelLoaderOutput:
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return Ideogram4ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )
