"""Krea 2 starter models."""

from invokeai.backend.model_manager.starter_models.common import qwen3_vl_encoder_4b
from invokeai.backend.model_manager.starter_models.qwen_image import qwen_image_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Krea2VariantType,
    ModelFormat,
    ModelType,
)

krea2_turbo = StarterModel(
    name="Krea-2 Turbo",
    base=BaseModelType.Krea2,
    source="krea/Krea-2-Turbo",
    description="Krea-2 Turbo - distilled 12B parameter text-to-image model (8 steps, CFG disabled). "
    "Full diffusers pipeline including the Qwen-Image VAE and Qwen3-VL text encoder. ~26GB",
    type=ModelType.Main,
    variant=Krea2VariantType.Turbo,
)

krea2_raw = StarterModel(
    name="Krea-2 Raw",
    base=BaseModelType.Krea2,
    source="krea/Krea-2-Raw",
    description="Krea-2 Raw - undistilled 12B base model (28 steps, CFG enabled). Full diffusers pipeline "
    "including the Qwen-Image VAE and Qwen3-VL text encoder. Primarily a base for finetuning / LoRA "
    "training; Turbo is recommended for standard inference. ~26GB",
    type=ModelType.Main,
    variant=Krea2VariantType.Base,
)

krea2_turbo_gguf_q4_k_m = StarterModel(
    name="Krea-2 Turbo (Q4_K_M GGUF)",
    base=BaseModelType.Krea2,
    source="https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF/resolve/main/krea2_turbo-Q4_K_M.gguf",
    description="Krea-2 Turbo transformer quantized to GGUF Q4_K_M for lower VRAM (~7GB transformer). "
    "GGUF ships only the transformer, so the Qwen-Image VAE and Qwen3-VL encoder are installed as "
    "dependencies.",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=Krea2VariantType.Turbo,
    dependencies=[qwen_image_vae, qwen3_vl_encoder_4b],
)

krea2_turbo_nvfp4 = StarterModel(
    name="Krea-2 Turbo (NVFP4)",
    base=BaseModelType.Krea2,
    source="https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_nvfp4.safetensors",
    description="Krea-2 Turbo transformer in Comfy-Org's nvfp4 build: under a third of the full-precision "
    "transformer's memory (~7.7GB), about the size of the Q4_K_M GGUF. It saves memory, not time - the weights are "
    "decoded every step. Ships only the transformer, so the Qwen-Image VAE and Qwen3-VL encoder are installed as "
    "dependencies.",
    type=ModelType.Main,
    variant=Krea2VariantType.Turbo,
    dependencies=[qwen_image_vae, qwen3_vl_encoder_4b],
)

krea2_turbo_gguf_q8_0 = StarterModel(
    name="Krea-2 Turbo (Q8_0 GGUF)",
    base=BaseModelType.Krea2,
    source="https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF/resolve/main/krea2_turbo-Q8_0.gguf",
    description="Krea-2 Turbo transformer quantized to GGUF Q8_0 (near-full quality, ~13GB transformer). "
    "GGUF ships only the transformer, so the Qwen-Image VAE and Qwen3-VL encoder are installed as "
    "dependencies.",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=Krea2VariantType.Turbo,
    dependencies=[qwen_image_vae, qwen3_vl_encoder_4b],
)
