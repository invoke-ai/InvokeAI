"""Qwen-Image starter models."""

from invokeai.backend.model_manager.starter_models.common import gemma2_2b_encoder, qwen_vl_encoder_fp8
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
    QwenImageVariantType,
)

# Qwen-Image uses a 16-channel latent (ambiguous with FLUX/SD3). The config probe disambiguates via the checkpoint's
# directory name (`…official_qwenimage_distill…`); if the HF single-file download drops it, the explicit
# base=QwenImage override the installer sends is trusted instead (see pid_decoder.py::_validate_base). Only the
# 2K-to-4K preset exists.
pid_decoder_qwenimage_2kto4k = StarterModel(
    name="PiD Decoder Qwen-Image (2K to 4K)",
    base=BaseModelType.QwenImage,
    source="nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for Qwen-Image latents, 2K-to-4K preset (first-generation architecture; see PiD 1.5 for the newer one). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_1_5_decoder_qwenimage_2kto4k = StarterModel(
    name="PiD 1.5 Decoder Qwen-Image (2K to 4K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/PixelDiT/resolve/main/diffusion_models/pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
    description="NVIDIA PiD 1.5 4x super-resolution decoder for Qwen-Image latents, 2K-to-4K preset, in Comfy-Org's single-file build. ~2.8GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_1_5_decoder_qwenimage_2kto4k_int8 = StarterModel(
    name="PiD 1.5 Decoder Qwen-Image (2K to 4K, int8)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/PixelDiT/resolve/main/diffusion_models/pid_1.5_qwenimage_1024_to_4096_4step_int8_convrot.safetensors",
    description="PiD 1.5 Qwen-Image decoder in Comfy-Org's int8_convrot build: its transformer layers stay int8, about 2GB once loaded instead of 5.6GB, with output practically identical to the full-precision build. ~1.6GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

# region Qwen Image components (shared between Edit and txt2img variants)
qwen_image_vae = StarterModel(
    name="Qwen Image VAE",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-Edit-2511::vae/diffusion_pytorch_model.safetensors",
    description="Qwen Image VAE (AutoencoderKLQwenImage), shared between the Edit and txt2img variants. "
    "Use with GGUF transformers to avoid downloading the full ~40GB Diffusers pipeline. (~250MB)",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

# region Qwen Image Edit
qwen_image_edit = StarterModel(
    name="Qwen Image Edit 2511",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-Edit-2511",
    description="Qwen Image Edit 2511 full diffusers model. Supports text-guided image editing with multiple reference images. (~40GB)",
    type=ModelType.Main,
    variant=QwenImageVariantType.Edit,
)

qwen_image_edit_gguf_q4_k_m = StarterModel(
    name="Qwen Image Edit 2511 (Q4_K_M)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf",
    description="Qwen Image Edit 2511 - Q4_K_M quantized transformer. Good quality/size balance. (~13GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q2_k = StarterModel(
    name="Qwen Image Edit 2511 (Q2_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q2_K.gguf",
    description="Qwen Image Edit 2511 - Q2_K heavily quantized transformer. Smallest size, lower quality. (~7.5GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q6_k = StarterModel(
    name="Qwen Image Edit 2511 (Q6_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q6_K.gguf",
    description="Qwen Image Edit 2511 - Q6_K quantized transformer. Near-lossless quality. (~17GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q8_0 = StarterModel(
    name="Qwen Image Edit 2511 (Q8_0)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q8_0.gguf",
    description="Qwen Image Edit 2511 - Q8_0 quantized transformer. Highest quality quantization. (~22GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_lightning_4step = StarterModel(
    name="Qwen Image Edit Lightning (4-step, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image Edit — enables generation in just 4 steps. "
    "Settings: Steps=4, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

qwen_image_edit_lightning_8step = StarterModel(
    name="Qwen Image Edit Lightning (8-step, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image Edit — enables generation in 8 steps with better quality. "
    "Settings: Steps=8, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

# Qwen Image (txt2img)
qwen_image = StarterModel(
    name="Qwen Image 2512",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-2512",
    description="Qwen Image 2512 full diffusers model. High-quality text-to-image generation. (~40GB)",
    type=ModelType.Main,
)

qwen_image_gguf_q4_k_m = StarterModel(
    name="Qwen Image 2512 (Q4_K_M)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf",
    description="Qwen Image 2512 - Q4_K_M quantized transformer. Good quality/size balance. (~13GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q2_k = StarterModel(
    name="Qwen Image 2512 (Q2_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q2_K.gguf",
    description="Qwen Image 2512 - Q2_K heavily quantized transformer. Smallest size, lower quality. (~7.5GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q6_k = StarterModel(
    name="Qwen Image 2512 (Q6_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q6_K.gguf",
    description="Qwen Image 2512 - Q6_K quantized transformer. Near-lossless quality. (~17GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q8_0 = StarterModel(
    name="Qwen Image 2512 (Q8_0)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf",
    description="Qwen Image 2512 - Q8_0 quantized transformer. Highest quality quantization. (~22GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_lightning_4step = StarterModel(
    name="Qwen Image Lightning (4-step, V2.0, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image — enables generation in just 4 steps. "
    "Settings: Steps=4, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

qwen_image_lightning_8step = StarterModel(
    name="Qwen Image Lightning (8-step, V2.0, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image — enables generation in 8 steps with better quality. "
    "Settings: Steps=8, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)
