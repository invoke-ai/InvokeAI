"""Starter models with no architecture of their own — CLIP encoders, upscalers, and the
like — shared by everything that needs them."""

from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

# This is CLIP-ViT-H-14-laion2B-s32B-b79K
ip_adapter_sd_image_encoder = StarterModel(
    name="IP Adapter SD1.5 Image Encoder",
    base=BaseModelType.Any,
    source="InvokeAI/ip_adapter_sd_image_encoder",
    description="IP Adapter SD Image Encoder",
    type=ModelType.CLIPVision,
)

# This is CLIP-ViT-bigG-14-laion2B-39B-b160k
ip_adapter_sdxl_image_encoder = StarterModel(
    name="IP Adapter SDXL Image Encoder",
    base=BaseModelType.Any,
    source="InvokeAI/ip_adapter_sdxl_image_encoder",
    description="IP Adapter SDXL Image Encoder",
    type=ModelType.CLIPVision,
)

# Note: This model is installed from the same source as the CLIPEmbed model below. The model contains both the image
# encoder and the text encoder, but we need separate model entries so that they get loaded correctly.
# Dependency-only (like the IP-Adapter encoders above): not listed in STARTER_MODELS,
# because its name collides with the same-named FLUX CLIPEmbed text encoder in the
# picker and it is no longer the imagemap index default. The FLUX IP-Adapter was
# trained against these exact OpenAI weights, so its dependency must stay this
# model, not the DFN2B one below.
clip_vit_l_image_encoder = StarterModel(
    name="clip-vit-large-patch14",
    base=BaseModelType.Any,
    source="InvokeAI/clip-vit-large-patch14",
    description="CLIP VIT-L Image Encoder ~1.7GB",
    type=ModelType.CLIPVision,
)
# Apple's DFN2B CLIP: same ViT-L-14 architecture as the model above, much
# stronger zero-shot weights (Data Filtering Networks, 39B samples seen). The
# imagemap index default: raw-cosine cluster labeling needs an encoder whose
# text/image geometry separates well, and the OpenAI weights let generic "hub"
# phrases (color words) outscore content phrases. The repo also carries an
# open_clip copy of the weights, which the HF file filter cannot distinguish
# from the transformers copy — hence the doubled download size in the blurb.
dfn2b_clip_vit_l_image_encoder = StarterModel(
    name="DFN2B-CLIP-ViT-L-14-39B",
    base=BaseModelType.Any,
    source="apple/DFN2B-CLIP-ViT-L-14-39B",
    description="DFN2B CLIP ViT-L Image Encoder (used by the imagemap index) ~3.4GB download",
    type=ModelType.CLIPVision,
)

# region TextEncoders
t5_base_encoder = StarterModel(
    name="t5_base_encoder",
    base=BaseModelType.Any,
    source="InvokeAI/t5-v1_1-xxl::bfloat16",
    description="T5-XXL text encoder (used in FLUX pipelines). ~9.5GB",
    type=ModelType.T5Encoder,
)

t5_8b_quantized_encoder = StarterModel(
    name="t5_bnb_int8_quantized_encoder",
    base=BaseModelType.Any,
    source="InvokeAI/t5-v1_1-xxl::bnb_llm_int8",
    description="T5-XXL text encoder with bitsandbytes LLM.int8() quantization (used in FLUX pipelines). ~5GB",
    type=ModelType.T5Encoder,
    format=ModelFormat.BnbQuantizedLlmInt8b,
)

t5_gguf_q3_k_s_encoder = StarterModel(
    name="t5_gguf_q3_k_s_encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q3_K_S.gguf",
    description="T5-XXL text encoder, GGUF Q3_K_S quantized (used in FLUX pipelines). Smallest size for low VRAM, lower quality. ~2.1GB",
    type=ModelType.T5Encoder,
    format=ModelFormat.GGUFQuantized,
)

t5_gguf_q6_k_encoder = StarterModel(
    name="t5_gguf_q6_k_encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q6_K.gguf",
    description="T5-XXL text encoder, GGUF Q6_K quantized (used in FLUX pipelines). Near-lossless quality. ~3.9GB",
    type=ModelType.T5Encoder,
    format=ModelFormat.GGUFQuantized,
)

clip_l_encoder = StarterModel(
    name="clip-vit-large-patch14",
    base=BaseModelType.Any,
    source="InvokeAI/clip-vit-large-patch14-text-encoder::bfloat16",
    description="CLIP-L text encoder (used in FLUX pipelines). ~250MB",
    type=ModelType.CLIPEmbed,
)

# region PiD (Pixel Diffusion Decoder)
# PiD's pretrained decoders condition on Gemma-2-2b-it caption embeddings (2304-dim). NVIDIA references the ungated
# mirror Efficient-Large-Model/gemma-2-2b-it. It is shared across all PiD backbones, so it is a dependency of each
# decoder below (and offered standalone here so it can be installed once).
gemma2_2b_encoder = StarterModel(
    name="Gemma 2 2B (PiD caption encoder)",
    base=BaseModelType.Any,
    source="Efficient-Large-Model/gemma-2-2b-it",
    description="Gemma-2-2b-it text encoder that PiD uses to condition its diffusion decode on a caption. ~5GB",
    type=ModelType.Gemma2Encoder,
    format=ModelFormat.Gemma2Encoder,
)

# endregion
# region SpandrelImageToImage
animesharp_v4_rcan = StarterModel(
    name="2x-AnimeSharpV4_RCAN",
    base=BaseModelType.Any,
    source="https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors",
    description="A 2x upscaling model (optimized for anime images).",
    type=ModelType.SpandrelImageToImage,
)

realesrgan_x4 = StarterModel(
    name="RealESRGAN_x4plus",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    description="A Real-ESRGAN 4x upscaling model (general-purpose).",
    type=ModelType.SpandrelImageToImage,
)

esrgan_srx4 = StarterModel(
    name="ESRGAN_SRx4_DF2KOST_official",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    description="The official ESRGAN 4x upscaling model.",
    type=ModelType.SpandrelImageToImage,
)

realesrgan_x2 = StarterModel(
    name="RealESRGAN_x2plus",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    description="A Real-ESRGAN 2x upscaling model (general-purpose).",
    type=ModelType.SpandrelImageToImage,
)

swinir = StarterModel(
    name="SwinIR - realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN",
    base=BaseModelType.Any,
    source="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth",
    description="A SwinIR 4x upscaling model.",
    type=ModelType.SpandrelImageToImage,
)

qwen_vl_encoder_fp8 = StarterModel(
    name="Qwen2.5-VL Encoder (fp8 scaled)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    description="ComfyUI's single-file FP8-scaled Qwen2.5-VL 7B encoder. Bundles the language model and "
    "visual tower; its tokenizer and config ship with InvokeAI. (~7GB)",
    type=ModelType.QwenVLEncoder,
    format=ModelFormat.Checkpoint,
)

qwen_vl_encoder_nvfp4 = StarterModel(
    name="Qwen2.5-VL Encoder (NVFP4)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors",
    description="ComfyUI's single-file nvfp4/fp8-mixed Qwen2.5-VL 7B encoder. Bundles the language model and visual "
    "tower; its tokenizer and config ship with InvokeAI. Its nvfp4 layers stay packed, about 6.7GB "
    "once loaded. ~5.7GB download",
    type=ModelType.QwenVLEncoder,
    format=ModelFormat.Checkpoint,
)

qwen_vl_encoder_diffusers = StarterModel(
    name="Qwen2.5-VL Encoder (Diffusers)",
    base=BaseModelType.Any,
    source="Qwen/Qwen-Image-Edit-2511::text_encoder+tokenizer+processor",
    description="Full-precision Qwen2.5-VL 7B encoder in Diffusers folder layout (text_encoder + tokenizer + processor). "
    "Larger than the fp8 variant but no on-the-fly dequantization. (~16GB)",
    type=ModelType.QwenVLEncoder,
    format=ModelFormat.QwenVLEncoder,
)

# region SigLIP
siglip = StarterModel(
    name="SigLIP - google/siglip-so400m-patch14-384",
    base=BaseModelType.Any,
    source="google/siglip-so400m-patch14-384",
    description="A SigLIP model (used by FLUX Redux).",
    type=ModelType.SigLIP,
)

# region LlavaOnevisionModel (vision-language models for Image-to-Prompt)
llava_onevision = StarterModel(
    name="LLaVA Onevision Qwen2 0.5B",
    base=BaseModelType.Any,
    source="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    description="LLaVA Onevision vision-language model (~1 GB). Lightweight default for the Image-to-Prompt feature.",
    type=ModelType.LlavaOnevision,
)

llava_onevision_7b = StarterModel(
    name="LLaVA Onevision Qwen2 7B",
    base=BaseModelType.Any,
    source="llava-hf/llava-onevision-qwen2-7b-ov-hf",
    description="LLaVA Onevision 7B vision-language model. Larger, higher-quality alternative for Image-to-Prompt. (~16 GB)",
    type=ModelType.LlavaOnevision,
)

# region TextLLM (causal language models for Prompt Expansion)
qwen2_5_1_5b_instruct = StarterModel(
    name="Qwen2.5-1.5B-Instruct",
    base=BaseModelType.Any,
    source="Qwen/Qwen2.5-1.5B-Instruct",
    description="Qwen2.5 1.5B instruction-tuned LLM. Recommended default for the Prompt Expansion feature — small and fast. (~3 GB)",
    type=ModelType.TextLLM,
)

qwen2_5_3b_instruct = StarterModel(
    name="Qwen2.5-3B-Instruct",
    base=BaseModelType.Any,
    source="Qwen/Qwen2.5-3B-Instruct",
    description="Qwen2.5 3B instruction-tuned LLM. Better prompt expansion quality at the cost of more VRAM. (~6 GB)",
    type=ModelType.TextLLM,
)

smollm2_1_7b_instruct = StarterModel(
    name="SmolLM2-1.7B-Instruct",
    base=BaseModelType.Any,
    source="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    description="SmolLM2 1.7B instruction-tuned LLM (Apache-2.0). Alternative to Qwen for prompt expansion. (~3 GB)",
    type=ModelType.TextLLM,
)

flux2_klein_qwen3_4b_encoder = StarterModel(
    name="FLUX.2 Klein Qwen3 4B Encoder",
    base=BaseModelType.Any,
    source="black-forest-labs/FLUX.2-klein-4B::text_encoder+tokenizer",
    description="Qwen3 4B text encoder for FLUX.2 Klein 4B (also compatible with Z-Image). ~8GB",
    type=ModelType.Qwen3Encoder,
)

flux2_klein_qwen3_8b_encoder = StarterModel(
    name="FLUX.2 Klein Qwen3 8B Encoder",
    base=BaseModelType.Any,
    source="black-forest-labs/FLUX.2-klein-9B::text_encoder+tokenizer",
    description="Qwen3 8B text encoder for FLUX.2 Klein 9B models. ~16GB",
    type=ModelType.Qwen3Encoder,
)

flux2_klein_qwen3_4b_encoder_fp4 = StarterModel(
    name="FLUX.2 Klein Qwen3 4B Encoder (FP4 mixed)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
    description="Comfy-Org's nvfp4/fp8-mixed Qwen3 4B text encoder for FLUX.2 Klein 4B. Its nvfp4 layers stay packed, about 5.2GB once loaded. ~3.6GB download",
    type=ModelType.Qwen3Encoder,
)

flux2_klein_qwen3_8b_encoder_fp4 = StarterModel(
    name="FLUX.2 Klein Qwen3 8B Encoder (FP4 mixed)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
    description="Comfy-Org's nvfp4/fp8-mixed Qwen3 8B text encoder for FLUX.2 Klein 9B. Its nvfp4 layers stay packed, about 9.4GB once loaded. ~6.3GB download",
    type=ModelType.Qwen3Encoder,
)

# A community repack -- Comfy-Org publishes no int8 build of this encoder. Pinned to a commit rather
# than `main` for the same reason the FLUX.1 and Klein 9B int8 entries are: the decode follows the
# file's own `comfy_quant` marker, which says `convrot: true` at group size 256. A reupload at the
# same path that changed or dropped that flag would be derotated wrongly, and the result loads
# cleanly, logs "Kept 252 of 252" and encodes noise.
flux2_klein_qwen3_8b_encoder_int8 = StarterModel(
    name="FLUX.2 Klein Qwen3 8B Encoder (int8)",
    base=BaseModelType.Any,
    source="https://huggingface.co/supermind/int8_convrot_models/resolve/eb6682675836a42aa91fd4effcd77d68fe60ff42/text_encoders/qwen_3_8b_int8_convrot.safetensors",
    description="Community int8_convrot Qwen3 8B text encoder for FLUX.2 Klein 9B. Its 252 projections stay int8 and dequantize per forward, so it occupies roughly what it downloads rather than the ~16GB of the BF16 release. ~9.4GB download",
    type=ModelType.Qwen3Encoder,
)

# Comfy-Org safetensors (single-file, 30-layer cow, with embedded Tekken tokenizer).
# Higher precision than the cow GGUFs and avoids the Tekken-via-HF-Hub fetch.
flux2_dev_comfy_mistral_fp8 = StarterModel(
    name="FLUX.2 [dev] Mistral Encoder (Comfy FP8)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
    description="Comfy-Org FP8 of BFL's 30-layer cow-mistral3-small. Best quality/size for prompt adherence; embeds Tekken tokenizer (no HF fetch needed). ~18GB",
    type=ModelType.MistralEncoder,
)

flux2_dev_comfy_mistral_bf16 = StarterModel(
    name="FLUX.2 [dev] Mistral Encoder (Comfy BF16)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
    description="Comfy-Org BF16 of BFL's 30-layer cow-mistral3-small. Reference precision; embeds Tekken tokenizer. ~35.6GB",
    type=ModelType.MistralEncoder,
)

flux2_dev_comfy_mistral_fp4 = StarterModel(
    name="FLUX.2 [dev] Mistral Encoder (Comfy FP4 mixed)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors",
    description="Comfy-Org FP4-mixed of BFL's 30-layer cow-mistral3-small; embeds Tekken tokenizer. Its nvfp4 layers stay packed, up to about 14.7GB once loaded. ~12.3GB download",
    type=ModelType.MistralEncoder,
)

# gguf-org cow GGUF variants (30-layer cow, llama.cpp packaging, also embed Tekken).
# Lower memory footprint than the Comfy safetensors but slightly lower fidelity.
flux2_dev_cow_mistral_q4 = StarterModel(
    name="FLUX.2 [dev] cow Mistral Encoder (GGUF Q4)",
    base=BaseModelType.Any,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/cow-mistral3-small-q4_0.gguf",
    description="cow-mistral3-small Q4_0 — 30-layer cow distillation BFL trained against. ~11.6GB",
    type=ModelType.MistralEncoder,
    format=ModelFormat.GGUFQuantized,
)

flux2_dev_cow_mistral_q8 = StarterModel(
    name="FLUX.2 [dev] cow Mistral Encoder (GGUF Q8)",
    base=BaseModelType.Any,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/cow-mistral3-small-q8_0.gguf",
    description="cow-mistral3-small Q8_0 — best prompt adherence among cow GGUF quants. ~20GB",
    type=ModelType.MistralEncoder,
    format=ModelFormat.GGUFQuantized,
)

flux2_dev_cow_mistral_iq4_xs = StarterModel(
    name="FLUX.2 [dev] cow Mistral Encoder (GGUF IQ4_XS)",
    base=BaseModelType.Any,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/cow-mistral3-small-iq4_xs.gguf",
    description="cow-mistral3-small IQ4_XS — smallest usable quant with reasonable adherence. ~11.1GB",
    type=ModelType.MistralEncoder,
    format=ModelFormat.GGUFQuantized,
)

# region Z-Image
z_image_qwen3_encoder = StarterModel(
    name="Z-Image Qwen3 Text Encoder",
    base=BaseModelType.Any,
    source="Tongyi-MAI/Z-Image-Turbo::text_encoder+tokenizer",
    description="Qwen3 4B text encoder with tokenizer for Z-Image (full precision). ~8GB",
    type=ModelType.Qwen3Encoder,
)

z_image_qwen3_encoder_quantized = StarterModel(
    name="Z-Image Qwen3 Text Encoder (quantized)",
    base=BaseModelType.Any,
    source="https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-Q6_K.gguf",
    description="Qwen3 4B text encoder for Z-Image quantized to GGUF Q6_K format. ~3.3GB",
    type=ModelType.Qwen3Encoder,
    format=ModelFormat.GGUFQuantized,
)

z_image_qwen3_encoder_fp4 = StarterModel(
    name="Z-Image Qwen3 Text Encoder (FP4 mixed)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors",
    description="Comfy-Org's nvfp4/fp8-mixed Qwen3 4B text encoder for Z-Image. Its nvfp4 layers stay packed, about 4.3GB once loaded. ~3.5GB download",
    type=ModelType.Qwen3Encoder,
)

# region Krea-2
# Standalone Qwen3-VL text encoder used by Krea-2 (distinct from the Qwen2.5-VL encoder above). Pair
# with single-file / GGUF Krea-2 transformers, which ship only the transformer. The Qwen-Image VAE
# dependency reuses the `qwen_image_vae` starter defined in the Qwen Image region.
qwen3_vl_encoder_4b = StarterModel(
    name="Qwen3-VL 4B Encoder (Diffusers)",
    base=BaseModelType.Any,
    source="Qwen/Qwen3-VL-4B-Instruct",
    description="Qwen3-VL 4B text encoder (Qwen3VLModel) used by Krea-2, in HuggingFace folder layout "
    "(includes tokenizer). Use with single-file / GGUF Krea-2 transformers. (~8GB)",
    type=ModelType.Qwen3VLEncoder,
    format=ModelFormat.Qwen3VLEncoder,
)

# region Wan 2.2 (local)
# Shared components — all Wan 2.2 variants use the UMT5-XXL text encoder. A14B
# (both T2V and I2V) uses a 16-channel VAE; TI2V-5B uses a 48-channel VAE. The
# two VAEs are not interchangeable.
wan_22_t5_encoder = StarterModel(
    name="Wan T5 Encoder (UMT5-XXL)",
    base=BaseModelType.Any,
    source="Wan-AI/Wan2.2-T2V-A14B-Diffusers::text_encoder+tokenizer",
    description="UMT5-XXL text encoder used by all Wan 2.2 variants (T2V/I2V A14B and TI2V-5B). "
    "Required when running a GGUF Wan main without a Diffusers Component Source. (~11GB)",
    type=ModelType.WanT5Encoder,
    format=ModelFormat.WanT5Encoder,
)

# DALL-E 2 removed — deprecated by OpenAI, shutdown May 12, 2026.
# region Anima
anima_qwen3_encoder = StarterModel(
    name="Anima Qwen3 0.6B Text Encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
    description="Qwen3 0.6B text encoder for Anima. ~1.2GB",
    type=ModelType.Qwen3Encoder,
    format=ModelFormat.Checkpoint,
)
