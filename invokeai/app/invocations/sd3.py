from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField, UIType
from invokeai.app.invocations.model import CLIPField, ModelIdentifierField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import SubModelType


class TransformerField(BaseModel):
    transformer: ModelIdentifierField = Field(description="Info to load unet submodel")
    scheduler: ModelIdentifierField = Field(description="Info to load scheduler submodel")


@invocation_output("sd3_model_loader_output")
class SD3ModelLoaderOutput(BaseInvocationOutput):
    """Stable Diffuion 3 base model loader output"""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP 1")
    clip2: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    clip3: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP 3")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation("sd3_model_loader", title="SD3 Main Model", tags=["model", "sd3"], category="model", version="1.0.0")
class SD3ModelLoaderInvocation(BaseInvocation):
    """Loads an SD3 base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.sd3_main_model, ui_type=UIType.SD3MainModel)

    def invoke(self, context: InvocationContext) -> SD3ModelLoaderOutput:
        model_key = self.model.key

        if not context.models.exists(model_key):
            raise Exception(f"Unknown model: {model_key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        scheduler = self.model.model_copy(update={"submodel_type": SubModelType.Scheduler})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer2 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        text_encoder2 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        tokenizer3 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
        text_encoder3 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder3})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return SD3ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, scheduler=scheduler),
            clip=CLIPField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[], skipped_layers=0),
            clip2=CLIPField(tokenizer=tokenizer2, text_encoder=text_encoder2, loras=[], skipped_layers=0),
            clip3=CLIPField(tokenizer=tokenizer3, text_encoder=text_encoder3, loras=[], skipped_layers=0),
            vae=VAEField(vae=vae),
        )
