"""MiniMax H3 starter models."""

from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

minimax_h3_components = StarterModel(
    name="MiniMax H3 Components",
    base=BaseModelType.MiniMaxH3,
    source="MiniMaxAI/MiniMax-H3::modular_model_index.json+transformer/config.json+tokenizer+processor+vae+audio_vae",
    description="MiniMax H3 shared components: tokenizer, processor and video/audio VAEs, without "
    "transformer or text-encoder weights (~11 GB). Pair with the MiniMax H3 single-file transformer "
    "and text encoder. NOTE: This model is distributed under a restrictive license that forbids its "
    "use in certain territories. Please see https://huggingface.co/MiniMaxAI/MiniMax-H3 for details.",
    type=ModelType.Main,
    format=ModelFormat.Diffusers,
)

minimax_h3_int8_text_encoder = StarterModel(
    name="MiniMax H3 Text Encoder (int8)",
    base=BaseModelType.MiniMaxH3,
    source="Comfy-Org/MiniMax-H3::text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    description="Truncated Qwen3-VL-32B conditioning encoder for MiniMax H3, int8 quantized (~27 GB). "
    "Select it in the MiniMax H3 Model Loader's text encoder field. NOTE: This model is distributed "
    "under a restrictive license that forbids its use in certain territories. Please see "
    "https://huggingface.co/MiniMaxAI/MiniMax-H3 for details.",
    type=ModelType.Qwen3VLEncoder,
    format=ModelFormat.Checkpoint,
)

minimax_h3_int8_transformer = StarterModel(
    name="MiniMax H3 FL2VA Transformer (int8, pruned)",
    base=BaseModelType.MiniMaxH3,
    source="Comfy-Org/MiniMax-H3::diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    description="MiniMax H3 video+audio generation. AdaLN-pruned int8 single-file transformer (~21 GB); "
    "select it in the MiniMax H3 Model Loader's transformer field. Total size with dependencies: ~59 GB. "
    "NOTE: This model is distributed under a restrictive license that forbids its use in certain "
    "territories. Please see https://huggingface.co/MiniMaxAI/MiniMax-H3 for details.",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[minimax_h3_components, minimax_h3_int8_text_encoder],
)

minimax_h3_ref2va_int8_transformer = StarterModel(
    name="MiniMax H3 Ref2VA Transformer (int8, pruned)",
    base=BaseModelType.MiniMaxH3,
    source="Comfy-Org/MiniMax-H3::diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    description="MiniMax H3 reference-conditioned video+audio generation: up to 3 video and 9 image "
    "references. AdaLN-pruned int8 single-file transformer (~21 GB); select it in the MiniMax H3 "
    "Model Loader's transformer field. Total size with dependencies: ~59 GB. NOTE: This model is "
    "distributed under a restrictive license that forbids its use in certain territories. Please "
    "see https://huggingface.co/MiniMaxAI/MiniMax-H3 for details.",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[minimax_h3_components, minimax_h3_int8_text_encoder],
)

minimax_h3_turbo_lora = StarterModel(
    name="MiniMax H3 Turbo LoRA",
    base=BaseModelType.MiniMaxH3,
    source="larryvrh/MiniMax-H3-Turbo-Lora::minimax_h3_turbo_v4_step600_ema.safetensors",
    description="Step-distillation LoRA for MiniMax H3 (Apache 2.0): renders video+audio in 4-8 "
    "denoising steps instead of ~50. Apply at strength 1.0 and lower Steps to 6-8. Works with the "
    "full and the pruned int8 transformers.",
    type=ModelType.LoRA,
    format=ModelFormat.LyCORIS,
)

# Only the *_comfyui_* files in the LightX2V repo are installable: they carry the native
# fused-key layout (diffusion_model.blocks.N.attn.qkv_proj) that the H3 LoRA probe accepts.
# The non-comfyui files use already-diffusers keys (transformer_blocks.N.attn.to_q with
# PEFT ".default" adapter names), which the probe rejects as a different architecture.
minimax_h3_lightx2v_turbo_lora = StarterModel(
    name="MiniMax H3 LightX2V Turbo LoRA",
    base=BaseModelType.MiniMaxH3,
    source="lightx2v/Minimax-h3-Turbo::minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    description="LightX2V step-distillation LoRA for MiniMax H3 (Apache 2.0), 8-step v1.0 (~2 GB): "
    "renders video+audio in ~8 denoising steps instead of ~50. Apply at strength 1.0 and lower "
    "Steps to 8. Works with the full and the pruned int8 transformers.",
    type=ModelType.LoRA,
    format=ModelFormat.LyCORIS,
)

# The LightX2V 8-step v1.0 (768p) Ref2V distillation replaces the earlier 4-step v0.1 repack
# (Comfy-Org/MiniMax-H3::loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors):
# v0.1 carries a strong rightward camera-pan bias whenever a VIDEO reference is conditioned
# on, which neither the undistilled 30-step base nor this v1.0 release shows (verified
# 2026-09-07). Same fused-key comfyui layout as the FL2VA LightX2V file above.
minimax_h3_lightx2v_ref2v_turbo_lora = StarterModel(
    name="MiniMax H3 LightX2V Ref2V Turbo LoRA",
    base=BaseModelType.MiniMaxH3,
    source="lightx2v/Minimax-h3-Turbo::minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    description="LightX2V step-distillation LoRA for the MiniMax H3 Ref2VA transformer (Apache 2.0), "
    "8-step v1.0 768p (~2 GB): renders reference-conditioned video+audio in ~8 denoising steps "
    "instead of ~50. Apply at strength 1.0 and lower Steps to 8. Trained against the Ref2VA "
    "transformer; not intended for FL2VA. Supersedes the 4-step v0.1 Ref2V Turbo repack, which "
    "pans the camera rightward whenever a video reference is used.",
    type=ModelType.LoRA,
    format=ModelFormat.LyCORIS,
)
