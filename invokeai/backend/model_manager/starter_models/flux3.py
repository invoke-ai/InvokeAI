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

# The prompt upsampler: a Qwen3.5-9B vision-language fine-tune that rewrites a short request into the caption FLUX.3 is
# trained on. A Text LLM, so it serves Expand Prompt, which sends it the system prompt the user picks. Its own
# repository is Apache-2.0 and not gated; the int8 file comes from the gated Comfy repository, under that repository's
# license.
_UPSAMPLER_SYSTEM_PROMPT = "Use it with the 't2i' system prompt from the model's system_prompts.json."

flux3_prompt_upsampler = StarterModel(
    name="FLUX.3 Prompt Upsampler",
    base=BaseModelType.Any,
    source="black-forest-labs/flux-3-image-klein-pup",
    description="FLUX.3 Image Klein's prompt upsampler (Qwen3.5-9B): a Text LLM for Expand Prompt that rewrites a "
    f"short prompt into a detailed FLUX.3 caption. {_UPSAMPLER_SYSTEM_PROMPT} bf16, ~17.5GB. Apache-2.0.",
    type=ModelType.TextLLM,
)

# Pinned to the Comfy repository commit the int8 layout was checked against (the bf16 single files of FLUX.3 moved
# within this repository once already). The file's `comfy_quant` markers drive the decode, so a re-upload at the same
# path could change how it loads.
flux3_prompt_upsampler_int8 = StarterModel(
    name="FLUX.3 Prompt Upsampler (int8)",
    base=BaseModelType.Any,
    source="https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/"
    "feef60b0aaf63a4773ace96636ebbcf43b881312/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
    description="The FLUX.3 prompt upsampler as one ComfyUI int8_convrot file, a Text LLM for Expand Prompt. It "
    "needs about 10.7GB of memory instead of 17.5GB, but dequantizes its int8 language-model weights for every generated "
    "token unless int8 Compute runs them on int8 kernels; use the bf16 model where it fits. "
    f"{_UPSAMPLER_SYSTEM_PROMPT} ~9.1GB. The Comfy repository is gated under the FLUX Kommunity License: accept it on "
    "Hugging Face first.",
    type=ModelType.TextLLM,
    format=ModelFormat.Checkpoint,
)
