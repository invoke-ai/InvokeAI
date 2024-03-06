from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager import SubModelType

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from .model import ClipField, ModelField, UNetField, VaeField


@invocation_output("sdxl_model_loader_output")
class SDXLModelLoaderOutput(BaseInvocationOutput):
    """SDXL base model loader output"""

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 1")
    clip2: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation_output("sdxl_refiner_model_loader_output")
class SDXLRefinerModelLoaderOutput(BaseInvocationOutput):
    """SDXL refiner model loader output"""

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip2: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation("sdxl_model_loader", title="SDXL Main Model", tags=["model", "sdxl"], category="model", version="1.0.1")
class SDXLModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl base model, outputting its submodels."""

    model: ModelField = InputField(
        description=FieldDescriptions.sdxl_main_model, input=Input.Direct, ui_type=UIType.SDXLMainModel
    )
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> SDXLModelLoaderOutput:
        model_key = self.model.key

        # TODO: not found exceptions
        if not context.models.exists(model_key):
            raise Exception(f"Unknown model: {model_key}")

        unet = self.model.model_copy(update={"submodel_type": SubModelType.UNet})
        scheduler = self.model.model_copy(update={"submodel_type": SubModelType.Scheduler})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer2 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        text_encoder2 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return SDXLModelLoaderOutput(
            unet=UNetField(unet=unet, scheduler=scheduler, loras=[]),
            clip=ClipField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[], skipped_layers=0),
            clip2=ClipField(tokenizer=tokenizer2, text_encoder=text_encoder2, loras=[], skipped_layers=0),
            vae=VaeField(vae=vae),
        )


@invocation(
    "sdxl_refiner_model_loader",
    title="SDXL Refiner Model",
    tags=["model", "sdxl", "refiner"],
    category="model",
    version="1.0.1",
)
class SDXLRefinerModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl refiner model, outputting its submodels."""

    model: ModelField = InputField(
        description=FieldDescriptions.sdxl_refiner_model, input=Input.Direct, ui_type=UIType.SDXLRefinerModel
    )
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> SDXLRefinerModelLoaderOutput:
        model_key = self.model.key

        # TODO: not found exceptions
        if not context.models.exists(model_key):
            raise Exception(f"Unknown model: {model_key}")

        unet = self.model.model_copy(update={"submodel_type": SubModelType.UNet})
        scheduler = self.model.model_copy(update={"submodel_type": SubModelType.Scheduler})
        tokenizer2 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        text_encoder2 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return SDXLRefinerModelLoaderOutput(
            unet=UNetField(unet=unet, scheduler=scheduler, loras=[]),
            clip2=ClipField(tokenizer=tokenizer2, text_encoder=text_encoder2, loras=[], skipped_layers=0),
            vae=VaeField(vae=vae),
        )
