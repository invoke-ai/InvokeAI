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
    source="https://huggingface.co/Comfy-Org/PixelDiT/resolve/f1aa9118da0a0b53a4433e5de9a4de2e9aae2193/diffusion_models/pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
    description="NVIDIA PiD 1.5 4x super-resolution decoder for Qwen-Image latents, 2K-to-4K preset, in Comfy-Org's single-file build. ~2.8GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_1_5_decoder_qwenimage_2kto4k_int8 = StarterModel(
    name="PiD 1.5 Decoder Qwen-Image (2K to 4K, int8)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/PixelDiT/resolve/f1aa9118da0a0b53a4433e5de9a4de2e9aae2193/diffusion_models/pid_1.5_qwenimage_1024_to_4096_4step_int8_convrot.safetensors",
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

qwen_image_edit_fp8_mixed = StarterModel(
    name="Qwen Image Edit 2511 (fp8 mixed)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/Qwen-Image-Edit/resolve/f68ace85e60b4a02a323e394253731947657b7d2/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
    description="Qwen Image Edit 2511 transformer in Comfy-Org's scaled fp8 mixed build. FP8 Storage is switched on "
    "when it is installed, so its fp8 layers stay fp8, about 20.5GB once loaded, which fills most of a 24GB card; "
    "with the working memory a generation needs, partial loading may keep part of it in RAM. Ships only the "
    "transformer, so the Qwen Image VAE and the Qwen2.5-VL encoder are installed as dependencies. (~20.5GB)",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
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

qwen_image_fp8 = StarterModel(
    name="Qwen Image 2512 (fp8)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/Qwen-Image/resolve/1f12b17be14c89b026c51a91d67c32f84bb047bc/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
    description="Qwen Image 2512 transformer in Comfy-Org's fp8 build (plain e4m3, no scales). FP8 Storage is "
    "switched on when it is installed, so its weights are held in fp8, about 20.5GB once loaded, which fills most "
    "of a 24GB card; with the working memory a generation needs, partial loading may keep part of it in RAM. "
    "Loading briefly needs about 41GB of free system RAM, because the file is widened to bf16 before FP8 Storage "
    "casts it back. Ships only the transformer, so the Qwen Image VAE and the Qwen2.5-VL encoder are installed as "
    "dependencies. (~20.4GB)",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
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

# region Qwen Image ControlNet
qwen_image_fun_controlnet_union = StarterModel(
    name="Qwen Image ControlNet Union",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union/resolve/main/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
    description="Unified ControlNet for Qwen Image 2512 supporting Canny, HED, Depth, Pose, MLSD, Scribble and Gray "
    "control images, plus inpainting. Works best at weights 0.70-0.95. Apache-2.0. ~3.5GB",
    type=ModelType.ControlNet,
)
qwen_image_instantx_controlnet_union = StarterModel(
    name="Qwen Image InstantX ControlNet Union",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/main/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors",
    description="InstantX's ControlNet for Qwen Image supporting Canny, Soft Edge, Depth and Pose control images; "
    "no inpainting. Trained on the original Qwen Image: on Qwen Image 2512 use weights of about 1.5-2.0. Apache-2.0. "
    "~3.5GB",
    type=ModelType.ControlNet,
)
