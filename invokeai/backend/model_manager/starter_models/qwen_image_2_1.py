"""Qwen-Image-2.1 starter models.

Every one is under the Qwen Research License: research and evaluation only, no commercial use without a separate
license from Alibaba's Tongyi Lab.
"""

from invokeai.backend.model_manager.starter_models.common import qwen3_vl_encoder_8b
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, QwenImage21VariantType

_LICENSE = "Qwen Research License: non-commercial use only."

qwen_image_2_1_vae = StarterModel(
    name="Qwen-Image-2.1 VAE",
    base=BaseModelType.QwenImage21,
    source="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
    description=f"Qwen-Image-2.1's RGBA VAE (64 latent channels, 16x), for its single-file and GGUF transformers. {_LICENSE} ~0.6GB",
    type=ModelType.VAE,
)

qwen_image_2_1 = StarterModel(
    name="Qwen-Image-2.1",
    base=BaseModelType.QwenImage21,
    source="Qwen/Qwen-Image-2.1",
    description="Qwen-Image-2.1, a 7B text-to-image model with strong text rendering and transparent (RGBA) output. "
    f"Full diffusers pipeline with its VAE and Qwen3-VL 8B encoder; 40 steps. {_LICENSE} ~31GB",
    type=ModelType.Main,
    variant=QwenImage21VariantType.Base,
)

qwen_image_2_1_turbo = StarterModel(
    name="Qwen-Image-2.1 Turbo",
    base=BaseModelType.QwenImage21,
    source="Qwen/Qwen-Image-2.1-Turbo",
    description="Qwen-Image-2.1 Turbo, distilled for 8 steps on its own sigma schedule. Full diffusers pipeline with "
    f"its VAE and Qwen3-VL 8B encoder. {_LICENSE} ~30GB",
    type=ModelType.Main,
    variant=QwenImage21VariantType.Turbo,
)

qwen_image_2_1_gguf_q4_k_m = StarterModel(
    name="Qwen-Image-2.1 (Q4_K_M GGUF)",
    base=BaseModelType.QwenImage21,
    source="https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf",
    description="Qwen-Image-2.1 transformer quantized to GGUF Q4_K_M (~4GB in VRAM instead of ~13GB). GGUF ships only "
    f"the transformer, so the VAE and the Qwen3-VL 8B encoder are installed as dependencies. {_LICENSE} ~15GB total",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImage21VariantType.Base,
    dependencies=[qwen_image_2_1_vae, qwen3_vl_encoder_8b],
)
