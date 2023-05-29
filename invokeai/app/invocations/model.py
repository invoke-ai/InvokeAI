from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field
import copy

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig

from ...backend.util.devices import choose_torch_device, torch_dtype
from ...backend.model_management import SDModelType

class ModelInfo(BaseModel):
    model_name: str = Field(description="Info to load submodel")
    model_type: SDModelType = Field(description="Info to load submodel")
    submodel: Optional[SDModelType] = Field(description="Info to load submodel")

class LoraInfo(ModelInfo):
    weight: float = Field(description="Lora's weight which to use when apply to model")

class UNetField(BaseModel):
    unet: ModelInfo = Field(description="Info to load unet submodel")
    scheduler: ModelInfo = Field(description="Info to load scheduler submodel")
    loras: List[LoraInfo] = Field(description="Loras to apply on model loading")

class ClipField(BaseModel):
    tokenizer: ModelInfo = Field(description="Info to load tokenizer submodel")
    text_encoder: ModelInfo = Field(description="Info to load text_encoder submodel")
    loras: List[LoraInfo] = Field(description="Loras to apply on model loading")

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
        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
        ):
            raise Exception(f"Unkown model name: {self.model_name}!")

        """
        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.Tokenizer,
        ):
            raise Exception(
                f"Failed to find tokenizer submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.TextEncoder,
        ):
            raise Exception(
                f"Failed to find text_encoder submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.UNet,
        ):
            raise Exception(
                f"Failed to find unet submodel from {self.model_name}! Check if model corrupted"
            )
        """


        return ModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.Diffusers,
                    submodel=SDModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.Diffusers,
                    submodel=SDModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.Diffusers,
                    submodel=SDModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.Diffusers,
                    submodel=SDModelType.TextEncoder,
                ),
                loras=[],
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=self.model_name,
                    model_type=SDModelType.Diffusers,
                    submodel=SDModelType.Vae,
                ),
            )
        )

class LoraLoaderOutput(BaseInvocationOutput):
    """Model loader output"""

    #fmt: off
    type: Literal["lora_loader_output"] = "lora_loader_output"

    unet: Optional[UNetField] = Field(default=None, description="UNet submodel")
    clip: Optional[ClipField] = Field(default=None, description="Tokenizer and text_encoder submodels")
    #fmt: on

class LoraLoaderInvocation(BaseInvocation):
    """Apply selected lora to unet and text_encoder."""

    type: Literal["lora_loader"] = "lora_loader"

    lora_name: str = Field(description="Lora model name")
    weight: float = Field(default=0.75, description="With what weight to apply lora")

    unet: Optional[UNetField] = Field(description="UNet model for applying lora")
    clip: Optional[ClipField] = Field(description="Clip model for applying lora")

    def invoke(self, context: InvocationContext) -> LoraLoaderOutput:

        if not context.services.model_manager.model_exists(
            model_name=self.lora_name,
            model_type=SDModelType.Lora,
        ):
            raise Exception(f"Unkown lora name: {self.lora_name}!")

        if self.unet is not None and any(lora.model_name == self.lora_name for lora in self.unet.loras):
            raise Exception(f"Lora \"{self.lora_name}\" already applied to unet")

        if self.clip is not None and any(lora.model_name == self.lora_name for lora in self.clip.loras):
            raise Exception(f"Lora \"{self.lora_name}\" already applied to clip")

        output = LoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)
            output.unet.loras.append(
                LoraInfo(
                    model_name=self.lora_name,
                    model_type=SDModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)
            output.clip.loras.append(
                LoraInfo(
                    model_name=self.lora_name,
                    model_type=SDModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        return output

