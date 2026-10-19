"""The SDXL refiner."""

from invokeai.backend.model_manager.starter_models.sdxl import sdxl_fp16_vae_fix
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)

sdxl_refiner = StarterModel(
    name="SDXL Refiner",
    base=BaseModelType.StableDiffusionXLRefiner,
    source="stabilityai/stable-diffusion-xl-refiner-1.0",
    description="The OG Stable Diffusion XL refiner model.",
    type=ModelType.Main,
    dependencies=[sdxl_fp16_vae_fix],
)
