from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import (
    GlmEncoderField,
    ModelIdentifierField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import SubModelType


@invocation_output("cogview4_model_loader_output")
class CogView4ModelLoaderOutput(BaseInvocationOutput):
    """CogView4 base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    glm_encoder: GlmEncoderField = OutputField(description=FieldDescriptions.glm_encoder, title="GLM Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "cogview4_model_loader",
    title="Main Model - CogView4",
    tags=["model", "cogview4"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class CogView4ModelLoaderInvocation(BaseInvocation):
    """Loads a CogView4 base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.cogview4_model,
        ui_type=UIType.CogView4MainModel,
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> CogView4ModelLoaderOutput:
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        glm_tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        glm_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return CogView4ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            glm_encoder=GlmEncoderField(tokenizer=glm_tokenizer, text_encoder=glm_encoder),
            vae=VAEField(vae=vae),
        )
