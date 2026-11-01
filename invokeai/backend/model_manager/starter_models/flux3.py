"""FLUX.3 Image starter models.

License: the FLUX Kommunity License v1.0 -- non-commercial use, or commercial use by users with under US$5M
annual revenue. Black Forest Labs' repositories (``flux-3-image-klein``, ``-base``, ``-comfy``) are gated: the
license must be accepted on the Hugging Face model page and a Hugging Face token configured. The community
GGUF repository is not gated; the same license applies.
"""

from invokeai.backend.model_manager.starter_models.common import qwen3_vl_encoder_4b
from invokeai.backend.model_manager.starter_models.flux2 import flux2_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelFormat, ModelType

_LICENSE = "FLUX Kommunity License (non-commercial, or commercial under US$5M annual revenue). Accept it on Hugging Face first."
_GGUF_LICENSE = "FLUX Kommunity License (non-commercial, or commercial under US$5M annual revenue)."

# The Hub repositories also carry BFL's own single-file package and quantized variants at their root; the
# Diffusers install takes only the pipeline's folders.
flux3_klein = StarterModel(
    name="FLUX.3 Klein",
    base=BaseModelType.Flux3,
    source="black-forest-labs/flux-3-image-klein::modular_model_index.json+transformer+vae+text_encoder",
    description="FLUX.3 Image Klein: the step-distilled 5.6B model for generation and editing with up to 10 "
    "reference images (16 steps, no CFG). Full Diffusers pipeline with the FLUX.2 VAE and the Qwen3-VL 4B "
    f"encoder. ~20GB. {_LICENSE}",
    type=ModelType.Main,
    variant=Flux3VariantType.Klein,
)

flux3_klein_base = StarterModel(
    name="FLUX.3 Klein Base",
    base=BaseModelType.Flux3,
    source="black-forest-labs/flux-3-image-klein-base::modular_model_index.json+transformer+vae+text_encoder",
    description="FLUX.3 Image Klein Base: the undistilled model (50 steps, CFG 3.5, negative prompt), meant for "
    "fine-tuning and LoRA training; Klein is about 6x faster for generation. Full Diffusers pipeline. ~20GB. "
    f"{_LICENSE}",
    type=ModelType.Main,
    variant=Flux3VariantType.KleinBase,
)

flux3_klein_single_file = StarterModel(
    name="FLUX.3 Klein (single file)",
    base=BaseModelType.Flux3,
    source="https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/diffusion_models/"
    "flux3_image_klein_sd_bf16.safetensors",
    description="FLUX.3 Image Klein transformer as one bf16 file; the FLUX.2 VAE and Qwen3-VL 4B encoder are "
    f"installed as dependencies and shared with other models. ~11GB. {_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.Klein,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_base_single_file = StarterModel(
    name="FLUX.3 Klein Base (single file)",
    base=BaseModelType.Flux3,
    source="https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/diffusion_models/"
    "flux3_image_klein_bf16.safetensors",
    description="FLUX.3 Image Klein Base transformer as one bf16 file; the FLUX.2 VAE and Qwen3-VL 4B encoder are "
    f"installed as dependencies. ~11GB. {_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.KleinBase,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

# Comfy-Org's quantized repackages, pinned to a commit rather than `main`. That is a correctness matter, not only
# reproducibility: the int8 decode follows each layer's `comfy_quant` marker (`convrot`, its group size) and the nvfp4
# one the layers the header names, so a re-upload at the same path that changed either would load cleanly and
# generate noise. The repository has renamed its files once already. Black Forest Labs' own `variants/` builds are
# not listed: Klein's and Klein Base's are both `model.safetensors`, so they are installed by URL (see the FLUX.3 docs).
_COMFY_QUANTIZED = (
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/"
    "feef60b0aaf63a4773ace96636ebbcf43b881312/diffusion_models"
)
_INT8_NOTE = (
    "Its quantized layers stay int8 in memory on every GPU and are decoded per step, or with int8 Compute (RTX 30-series "
    "and newer) multiplied in int8."
)
_NVFP4_NOTE = "Its 4-bit weights are decoded on every step."

flux3_klein_int8 = StarterModel(
    name="FLUX.3 Klein (int8)",
    base=BaseModelType.Flux3,
    source=f"{_COMFY_QUANTIZED}/flux3-image-klein-int8-convrot.safetensors",
    description="FLUX.3 Image Klein transformer in Comfy-Org's int8_convrot build, about half the memory of bf16. "
    f"{_INT8_NOTE} The FLUX.2 VAE and Qwen3-VL 4B encoder are installed as dependencies. ~5.8GB. {_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.Klein,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_nvfp4 = StarterModel(
    name="FLUX.3 Klein (NVFP4)",
    base=BaseModelType.Flux3,
    source=f"{_COMFY_QUANTIZED}/flux3-image-klein-nvfp4.safetensors",
    description="FLUX.3 Image Klein transformer in Comfy-Org's nvfp4 build: 33 of its 43 blocks in 4 bits, the rest "
    f"bf16. {_NVFP4_NOTE} The FLUX.2 VAE and Qwen3-VL 4B encoder are installed as dependencies. ~5.3GB. {_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.Klein,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_base_int8 = StarterModel(
    name="FLUX.3 Klein Base (int8)",
    base=BaseModelType.Flux3,
    source=f"{_COMFY_QUANTIZED}/flux3-image-klein-base-int8-convrot.safetensors",
    description="FLUX.3 Image Klein Base transformer in Comfy-Org's int8_convrot build, about half the memory of bf16. "
    f"{_INT8_NOTE} The FLUX.2 VAE and Qwen3-VL 4B encoder are installed as dependencies. ~5.8GB. {_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.KleinBase,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_base_nvfp4 = StarterModel(
    name="FLUX.3 Klein Base (NVFP4)",
    base=BaseModelType.Flux3,
    source=f"{_COMFY_QUANTIZED}/flux3-image-klein-base-nvfp4.safetensors",
    description="FLUX.3 Image Klein Base transformer in Comfy-Org's nvfp4 build: 33 of its 43 blocks in 4 bits, the "
    f"rest bf16. {_NVFP4_NOTE} The FLUX.2 VAE and Qwen3-VL 4B encoder are installed as dependencies. ~5.3GB. "
    f"{_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=Flux3VariantType.KleinBase,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_gguf_q4_k_m = StarterModel(
    name="FLUX.3 Klein (GGUF Q4_K_M)",
    base=BaseModelType.Flux3,
    source="https://huggingface.co/Abiray/flux-3-image-klein-sd-GGUF/resolve/main/flux3_image_klein_sd_Q4_K_M.gguf",
    description="FLUX.3 Image Klein transformer quantized to GGUF Q4_K_M for lower VRAM; the FLUX.2 VAE and "
    f"Qwen3-VL 4B encoder are installed as dependencies. ~3.3GB. {_GGUF_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=Flux3VariantType.Klein,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)

flux3_klein_gguf_q8_0 = StarterModel(
    name="FLUX.3 Klein (GGUF Q8_0)",
    base=BaseModelType.Flux3,
    source="https://huggingface.co/Abiray/flux-3-image-klein-sd-GGUF/resolve/main/flux3_image_klein_sd_Q8_0.gguf",
    description="FLUX.3 Image Klein transformer quantized to GGUF Q8_0, close to bf16 quality at half its memory; "
    f"the FLUX.2 VAE and Qwen3-VL 4B encoder are installed as dependencies. ~6GB. {_GGUF_LICENSE}",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=Flux3VariantType.Klein,
    dependencies=[flux2_vae, qwen3_vl_encoder_4b],
)
