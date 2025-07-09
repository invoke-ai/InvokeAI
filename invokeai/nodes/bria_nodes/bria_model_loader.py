from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import (
    ModelIdentifierField,
    SubModelType,
    T5EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)


@invocation_output("bria_model_loader_output")
class BriaModelLoaderOutput(BaseInvocationOutput):
    """Bria base model loader output"""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "bria_model_loader",
    title="Main Model - Bria",
    tags=["model", "bria"],
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaModelLoaderInvocation(BaseInvocation):
    """Loads a bria base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description="Bria model (Transformer) to load",
        ui_type=UIType.BriaMainModel,
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> BriaModelLoaderOutput:
        for key in [self.model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return BriaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            t5_encoder=T5EncoderField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )
