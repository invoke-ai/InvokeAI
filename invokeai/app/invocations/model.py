from typing import Literal, Optional, Union
from pydantic import BaseModel, Field

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig

from ...backend.util.devices import choose_torch_device, torch_dtype
from ...backend.model_management import SDModelType

class ModelInfo(BaseModel):
    model_name: str = Field(description="Info to load unet submodel")
    model_type: str = Field(description="Info to load unet submodel")
    submodel: Optional[str] = Field(description="Info to load unet submodel")

class UNetField(BaseModel):
    unet: ModelInfo = Field(description="Info to load unet submodel")
    scheduler: ModelInfo = Field(description="Info to load scheduler submodel")
    # loras: List[ModelInfo]

class ClipField(BaseModel):
    tokenizer: ModelInfo = Field(description="Info to load tokenizer submodel")
    text_encoder: ModelInfo = Field(description="Info to load text_encoder submodel")
    # loras: List[ModelInfo]

class VaeField(BaseModel):
    # TODO: better naming?
    vae: ModelInfo = Field(description="Info to load vae submodel")


class ModelLoaderOutput(BaseInvocationOutput):
    """Model loader output"""

    #fmt: off
    type: Literal["model_loader_output"] = "model_loader_output"

    unet: UNetField = Field(default=None, description="UNet submodel")
    clip: ClipField = Field(default=None, description="Tokenizer and text_encoder submodels")
    vae: VaeField = Field(default=None, description="Vae submodel")
    #fmt: on


class ModelLoaderInvocation(BaseInvocation):
    """Loading submodels of selected model."""

    type: Literal["model_loader"] = "model_loader"

    model_name: str = Field(default="", description="Model to load")
    # TODO: precision?

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["model", "loader"],
                "type_hints": {
                  "model_name": "model" # TODO: rename to model_name?
                }
            },
        }

    def invoke(self, context: InvocationContext) -> ModelLoaderOutput:

        # TODO: not found exceptions
        if not context.services.model_manager.valid_model(
            model_name=self.model_name,
            model_type=SDModelType.diffusers,
        ):
            raise Exception(f"Unkown model name: {self.model_name}!")

        """
        if not context.services.model_manager.valid_model(
            model_name=self.model_name,
            model_type=SDModelType.diffusers,
            submodel=SDModelType.tokenizer,
        ):
            raise Exception(
                f"Failed to find tokenizer submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.valid_model(
            model_name=self.model_name,
            model_type=SDModelType.diffusers,
            submodel=SDModelType.text_encoder,
        ):
            raise Exception(
                f"Failed to find text_encoder submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.valid_model(
            model_name=self.model_name,
            model_type=SDModelType.diffusers,
            submodel=SDModelType.unet,
        ):
            raise Exception(
                f"Failed to find unet submodel from {self.model_name}! Check if model corrupted"
            )
        """


        return ModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.diffusers.name,
                    submodel=SDModelType.unet.name,
                ),
                scheduler=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.diffusers.name,
                    submodel=SDModelType.scheduler.name,
                ),
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.diffusers.name,
                    submodel=SDModelType.tokenizer.name,
                ),
                text_encoder=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.diffusers.name,
                    submodel=SDModelType.text_encoder.name,
                ),
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.diffusers.name,
                    submodel=SDModelType.vae.name,
                ),
            )
        )
