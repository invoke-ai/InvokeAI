from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationConfig,
    InvocationContext,
)
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.model import LoRAModelField, MainModelField, VAEModelField


class LoRAMetadataField(BaseModel):
    """LoRA metadata for an image generated in InvokeAI."""

    lora: LoRAModelField = Field(description="The LoRA model")
    weight: float = Field(description="The weight of the LoRA model")


class CoreMetadata(BaseModel):
    """Core generation metadata for an image generated in InvokeAI."""

    generation_mode: str = Field(
        description="The generation mode that output this image",
    )
    positive_prompt: str = Field(description="The positive prompt parameter")
    negative_prompt: str = Field(description="The negative prompt parameter")
    width: int = Field(description="The width parameter")
    height: int = Field(description="The height parameter")
    seed: int = Field(description="The seed used for noise generation")
    rand_device: str = Field(description="The device used for random number generation")
    cfg_scale: float = Field(description="The classifier-free guidance scale parameter")
    steps: int = Field(description="The number of steps used for inference")
    scheduler: str = Field(description="The scheduler used for inference")
    clip_skip: int = Field(
        description="The number of skipped CLIP layers",
    )
    model: MainModelField = Field(description="The main model used for inference")
    controlnets: list[ControlField] = Field(description="The ControlNets used for inference")
    loras: list[LoRAMetadataField] = Field(description="The LoRAs used for inference")
    vae: Union[VAEModelField, None] = Field(
        default=None,
        description="The VAE used for decoding, if the main model's default was not used",
    )

    # Latents-to-Latents
    strength: Union[float, None] = Field(
        default=None,
        description="The strength used for latents-to-latents",
    )
    init_image: Union[str, None] = Field(default=None, description="The name of the initial image")

    # SDXL
    positive_style_prompt: Union[str, None] = Field(default=None, description="The positive style prompt parameter")
    negative_style_prompt: Union[str, None] = Field(default=None, description="The negative style prompt parameter")

    # SDXL Refiner
    refiner_model: Union[MainModelField, None] = Field(default=None, description="The SDXL Refiner model used")
    refiner_cfg_scale: Union[float, None] = Field(
        default=None,
        description="The classifier-free guidance scale parameter used for the refiner",
    )
    refiner_steps: Union[int, None] = Field(default=None, description="The number of steps used for the refiner")
    refiner_scheduler: Union[str, None] = Field(default=None, description="The scheduler used for the refiner")
    refiner_aesthetic_store: Union[float, None] = Field(
        default=None, description="The aesthetic score used for the refiner"
    )
    refiner_start: Union[float, None] = Field(default=None, description="The start value used for refiner denoising")


class ImageMetadata(BaseModel):
    """An image's generation metadata"""

    metadata: Optional[dict] = Field(
        default=None,
        description="The image's core metadata, if it was created in the Linear or Canvas UI",
    )
    graph: Optional[dict] = Field(default=None, description="The graph that created the image")


class MetadataAccumulatorOutput(BaseInvocationOutput):
    """The output of the MetadataAccumulator node"""

    type: Literal["metadata_accumulator_output"] = "metadata_accumulator_output"

    metadata: CoreMetadata = Field(description="The core metadata for the image")


class MetadataAccumulatorInvocation(BaseInvocation):
    """Outputs a Core Metadata Object"""

    type: Literal["metadata_accumulator"] = "metadata_accumulator"

    generation_mode: str = Field(
        description="The generation mode that output this image",
    )
    positive_prompt: str = Field(description="The positive prompt parameter")
    negative_prompt: str = Field(description="The negative prompt parameter")
    width: int = Field(description="The width parameter")
    height: int = Field(description="The height parameter")
    seed: int = Field(description="The seed used for noise generation")
    rand_device: str = Field(description="The device used for random number generation")
    cfg_scale: float = Field(description="The classifier-free guidance scale parameter")
    steps: int = Field(description="The number of steps used for inference")
    scheduler: str = Field(description="The scheduler used for inference")
    clip_skip: int = Field(
        description="The number of skipped CLIP layers",
    )
    model: MainModelField = Field(description="The main model used for inference")
    controlnets: list[ControlField] = Field(description="The ControlNets used for inference")
    loras: list[LoRAMetadataField] = Field(description="The LoRAs used for inference")
    strength: Union[float, None] = Field(
        default=None,
        description="The strength used for latents-to-latents",
    )
    init_image: Union[str, None] = Field(default=None, description="The name of the initial image")
    vae: Union[VAEModelField, None] = Field(
        default=None,
        description="The VAE used for decoding, if the main model's default was not used",
    )

    # SDXL
    positive_style_prompt: Union[str, None] = Field(default=None, description="The positive style prompt parameter")
    negative_style_prompt: Union[str, None] = Field(default=None, description="The negative style prompt parameter")

    # SDXL Refiner
    refiner_model: Union[MainModelField, None] = Field(default=None, description="The SDXL Refiner model used")
    refiner_cfg_scale: Union[float, None] = Field(
        default=None,
        description="The classifier-free guidance scale parameter used for the refiner",
    )
    refiner_steps: Union[int, None] = Field(default=None, description="The number of steps used for the refiner")
    refiner_scheduler: Union[str, None] = Field(default=None, description="The scheduler used for the refiner")
    refiner_aesthetic_store: Union[float, None] = Field(
        default=None, description="The aesthetic score used for the refiner"
    )
    refiner_start: Union[float, None] = Field(default=None, description="The start value used for refiner denoising")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Metadata Accumulator",
                "tags": ["image", "metadata", "generation"],
            },
        }

    def invoke(self, context: InvocationContext) -> MetadataAccumulatorOutput:
        """Collects and outputs a CoreMetadata object"""

        return MetadataAccumulatorOutput(metadata=CoreMetadata(**self.dict()))
