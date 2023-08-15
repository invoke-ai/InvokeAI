from typing import Literal

from ...backend.model_management import ModelType, SubModelType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    tags,
    title,
)
from .model import ClipField, MainModelField, ModelInfo, UNetField, VaeField


class SDXLModelLoaderOutput(BaseInvocationOutput):
    """SDXL base model loader output"""

    type: Literal["sdxl_model_loader_output"] = "sdxl_model_loader_output"

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 1")
    clip2: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


class SDXLRefinerModelLoaderOutput(BaseInvocationOutput):
    """SDXL refiner model loader output"""

    type: Literal["sdxl_refiner_model_loader_output"] = "sdxl_refiner_model_loader_output"

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")
    clip2: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP 2")
    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


@title("SDXL Main Model Loader")
@tags("model", "sdxl")
class SDXLModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl base model, outputting its submodels."""

    type: Literal["sdxl_model_loader"] = "sdxl_model_loader"

    # Inputs
    model: MainModelField = InputField(
        description=FieldDescriptions.sdxl_main_model, input=Input.Direct, ui_type=UIType.SDXLMainModel
    )
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> SDXLModelLoaderOutput:
        base_model = self.model.base_model
        model_name = self.model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return SDXLModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            clip2=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer2,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder2,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


@title("SDXL Refiner Model Loader")
@tags("model", "sdxl", "refiner")
class SDXLRefinerModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl refiner model, outputting its submodels."""

    type: Literal["sdxl_refiner_model_loader"] = "sdxl_refiner_model_loader"

    # Inputs
    model: MainModelField = InputField(
        description=FieldDescriptions.sdxl_refiner_model,
        input=Input.Direct,
        ui_type=UIType.SDXLRefinerModel,
    )
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> SDXLRefinerModelLoaderOutput:
        base_model = self.model.base_model
        model_name = self.model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return SDXLRefinerModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip2=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer2,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder2,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )
