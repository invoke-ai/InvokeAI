"""Z-Image starter models."""

from invokeai.backend.model_manager.starter_models.common import (
    z_image_qwen3_encoder_fp4,
    z_image_qwen3_encoder_quantized,
)
from invokeai.backend.model_manager.starter_models.flux import flux_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

z_image_turbo = StarterModel(
    name="Z-Image Turbo",
    base=BaseModelType.ZImage,
    source="Tongyi-MAI/Z-Image-Turbo",
    description="Z-Image Turbo - fast 6B parameter text-to-image model with 8 inference steps. Supports bilingual prompts (English & Chinese). ~33GB",
    type=ModelType.Main,
)

z_image_turbo_quantized = StarterModel(
    name="Z-Image Turbo (quantized)",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K.gguf",
    description="Z-Image Turbo quantized to GGUF Q4_K format. Requires standalone Qwen3 text encoder and Flux VAE. ~4GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[z_image_qwen3_encoder_quantized, flux_vae],
)

z_image_turbo_q8 = StarterModel(
    name="Z-Image Turbo (Q8)",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
    description="Z-Image Turbo quantized to GGUF Q8_0 format. Higher quality, larger size. Requires standalone Qwen3 text encoder and Flux VAE. ~6.6GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[z_image_qwen3_encoder_quantized, flux_vae],
)

z_image_turbo_nvfp4 = StarterModel(
    name="Z-Image Turbo (NVFP4)",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
    description="Z-Image Turbo in Comfy-Org's nvfp4 build. Its quantized layers stay packed, so it needs about a third of the full-precision transformer's memory. Requires standalone Qwen3 text encoder and Flux VAE. ~4.5GB",
    type=ModelType.Main,
    dependencies=[z_image_qwen3_encoder_fp4, flux_vae],
)

z_image_turbo_sdnq = StarterModel(
    name="Z-Image Turbo (SDNQ uint4 + SVD)",
    base=BaseModelType.ZImage,
    source="Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    description="Z-Image Turbo quantized via SDNQ to uint4 + SVD rank 32. Full self-contained "
    "ZImagePipeline (transformer + Qwen3 + VAE). ~5GB",
    type=ModelType.Main,
    format=ModelFormat.SDNQQuantized,
)

z_image_controlnet_union = StarterModel(
    name="Z-Image ControlNet Union",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
    description="Unified ControlNet for Z-Image Turbo supporting Canny, HED, Depth, Pose, MLSD, and Inpainting modes.",
    type=ModelType.ControlNet,
)

z_image_controlnet_tile = StarterModel(
    name="Z-Image ControlNet Tile",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors",
    description="Dedicated Tile ControlNet for Z-Image Turbo. Useful for upscaling and adding detail. ~6.7GB",
    type=ModelType.ControlNet,
)
