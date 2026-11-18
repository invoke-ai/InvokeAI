"""MOSS-Image starter models."""

from invokeai.backend.model_manager.starter_models.flux2 import flux2_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen35VariantType

# Both are complete diffusers pipelines. Their text encoder and VAE are byte-identical, but each install keeps its own
# copy, so the descriptions say what installing both costs.
_MOSS_IMAGE_REQUIREMENTS = (
    "Needs ~27GB of RAM or VRAM for its weights: the Qwen3.5-9B text encoder (~17GB in bf16) and the ~10GB "
    "transformer, which low-VRAM mode loads one after the other. Apache-2.0."
)

moss_image_base = StarterModel(
    name="MOSS-Image Base",
    base=BaseModelType.MossImage,
    source="OpenMOSS-Team/MOSS-Image-Base",
    description=(
        "OpenMOSS MOSS-Image: a 5B single-stream DiT conditioned by Qwen3.5-9B, decoding with the FLUX.2 VAE. "
        "Takes natural-language prompts, tag sequences and JSON layouts. Defaults to 50 steps with CFG 3. "
        f"{_MOSS_IMAGE_REQUIREMENTS} Installing the SFT model as well downloads the same encoder again. ~27GB"
    ),
    type=ModelType.Main,
)

moss_image_sft = StarterModel(
    name="MOSS-Image SFT",
    base=BaseModelType.MossImage,
    source="OpenMOSS-Team/MOSS-Image-SFT",
    description=(
        "The supervised fine-tune of MOSS-Image Base: the same encoder and VAE with a fine-tuned transformer, which "
        f"groups text and lays out scenes more cleanly. {_MOSS_IMAGE_REQUIREMENTS} Installing the Base model as well "
        "downloads the same encoder again. ~27GB"
    ),
    type=ModelType.Main,
)

moss_image_prompt_enhancer = StarterModel(
    name="Gemma 4 E4B (Prompt Enhancer)",
    base=BaseModelType.Any,
    source="google/gemma-4-E4B-it",
    description=(
        "The recommended Expand Prompt model for MOSS-Image. Pair it with the seeded MOSS-Image Prompt Enhancer "
        "system prompt, which turns a short idea into a detailed caption or, for posters and other images with "
        "visible text, a layout. Optional. (~16 GB)"
    ),
    type=ModelType.TextLLM,
)

# The file has the key set of the FLUX.3 prompt upsampler (a Text LLM), so only its source tells identification what it
# is (`is_moss_image_encoder_source`); the starter names the type as well. Pinned to the Comfy repository commit the
# int8 layout was checked against: the file's `comfy_quant` markers drive the decode, so a re-upload at the same path
# could change how it loads, and the pinned URL still matches the source check.
moss_image_qwen3_5_encoder_int8 = StarterModel(
    name="Qwen3.5 9B Encoder (int8)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/MOSS-Image/resolve/"
    "2929efe3cdd506c0a430bcfca70a9a8d805cc095/text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    description=(
        "ComfyUI's int8 build of the Qwen3.5-9B text encoder that MOSS-Image reads, about half the size of the bf16 "
        "encoder. Select it as the Qwen3.5 encoder of a MOSS-Image model. The file also holds a vision tower and an LM "
        "head, which are not loaded. Apache-2.0. ~9.8GB"
    ),
    type=ModelType.Qwen35Encoder,
    format=ModelFormat.Checkpoint,
    variant=Qwen35VariantType.Qwen35_9B,
)

# Transformer-only builds ship neither the FLUX.2 VAE nor the Qwen3.5 encoder, and the loader requires both, so every
# entry below installs them as dependencies. Pinned to commits: the int8 decode follows each file's own `comfy_quant`
# markers, and a re-upload at the same path could change how it loads.
_MOSS_COMFY = (
    "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/2929efe3cdd506c0a430bcfca70a9a8d805cc095/diffusion_models"
)
_MOSS_BASE_GGUF = "https://huggingface.co/unsloth/MOSS-Image-Base-GGUF/resolve/63e7dee4bdc4254f125a704913edd84a3291b80f"
_MOSS_SFT_GGUF = "https://huggingface.co/unsloth/MOSS-Image-SFT-GGUF/resolve/a013f7442e672e67b21f485b4b4f369df5500847"
_MOSS_TRANSFORMER_ONLY = (
    "Ships only the transformer, so the FLUX.2 VAE and the Qwen3.5 9B encoder are installed as dependencies. "
    "Apache-2.0."
)

moss_image_base_single_file = StarterModel(
    name="MOSS-Image Base (single file)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_COMFY}/moss_image_base_bf16.safetensors",
    description=(
        "MOSS-Image Base transformer as one bf16 file, from Comfy-Org. Defaults to 50 steps with CFG 3. "
        f"{_MOSS_TRANSFORMER_ONLY} ~10.2GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_sft_single_file = StarterModel(
    name="MOSS-Image SFT (single file)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_COMFY}/moss_image_sft_bf16.safetensors",
    description=(
        "The supervised fine-tune of MOSS-Image Base as one bf16 file, from Comfy-Org. "
        f"{_MOSS_TRANSFORMER_ONLY} ~10.2GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_base_int8 = StarterModel(
    name="MOSS-Image Base (int8)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_COMFY}/moss_image_base_int8_convrot.safetensors",
    description=(
        "MOSS-Image Base transformer in Comfy-Org's int8_convrot build, about half the size of the bf16 file. "
        f"{_MOSS_TRANSFORMER_ONLY} ~5.4GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_sft_int8 = StarterModel(
    name="MOSS-Image SFT (int8)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_COMFY}/moss_image_sft_int8_convrot.safetensors",
    description=(
        "The supervised fine-tune of MOSS-Image Base in Comfy-Org's int8_convrot build, about half the size of the "
        f"bf16 file. {_MOSS_TRANSFORMER_ONLY} ~5.4GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_base_gguf_q4_k_m = StarterModel(
    name="MOSS-Image Base (GGUF Q4_K_M)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_BASE_GGUF}/MOSS-Image-Base-Q4_K_M.gguf",
    description=(
        "MOSS-Image Base transformer quantized to GGUF Q4_K_M for lower VRAM, from unsloth. "
        f"{_MOSS_TRANSFORMER_ONLY} ~3.2GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_base_gguf_q8_0 = StarterModel(
    name="MOSS-Image Base (GGUF Q8_0)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_BASE_GGUF}/MOSS-Image-Base-Q8_0.gguf",
    description=(
        "MOSS-Image Base transformer quantized to GGUF Q8_0 with near-full quality, from unsloth. "
        f"{_MOSS_TRANSFORMER_ONLY} ~5.5GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_sft_gguf_q4_k_m = StarterModel(
    name="MOSS-Image SFT (GGUF Q4_K_M)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_SFT_GGUF}/MOSS-Image-SFT-Q4_K_M.gguf",
    description=(
        "The supervised fine-tune of MOSS-Image Base quantized to GGUF Q4_K_M for lower VRAM, from unsloth. "
        f"{_MOSS_TRANSFORMER_ONLY} ~3.2GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)

moss_image_sft_gguf_q8_0 = StarterModel(
    name="MOSS-Image SFT (GGUF Q8_0)",
    base=BaseModelType.MossImage,
    source=f"{_MOSS_SFT_GGUF}/MOSS-Image-SFT-Q8_0.gguf",
    description=(
        "The supervised fine-tune of MOSS-Image Base quantized to GGUF Q8_0 with near-full quality, from unsloth. "
        f"{_MOSS_TRANSFORMER_ONLY} ~5.5GB"
    ),
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, moss_image_qwen3_5_encoder_int8],
)
