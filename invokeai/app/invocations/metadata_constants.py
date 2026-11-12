from enum import Enum
from typing import Literal

__all__ = [
    "CUSTOM_LABEL",
    "MetadataType",
    "CORE_METADATA_REGISTRY",
    "CORE_STRING_LABELS",
    "CORE_INTEGER_LABELS",
    "CORE_FLOAT_LABELS",
    "CORE_BOOL_LABELS",
    "CORE_SCHEDULER_LABELS",
    "CORE_MODEL_LABELS_TUPLE",
    "CORE_VAE_LABELS_TUPLE",
    "CORE_LABELS_TUPLE",
    "CORE_LABELS",
    "CORE_LABELS_STRING",
    "CORE_LABELS_INTEGER",
    "CORE_LABELS_FLOAT",
    "CORE_LABELS_BOOL",
    "CORE_LABELS_SCHEDULER",
    "CORE_LABELS_MODEL",
    "CORE_LABELS_VAE",
    "CORE_MODEL_LABELS",
]


class MetadataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    SCHEDULER = "scheduler"
    MODEL = "model"
    VAE = "vae"


T = MetadataType

CUSTOM_LABEL: str = "* CUSTOM LABEL *"

# Single canonical registry of all core metadata keys and their allowed value types.
# Order is intentionally human-curated:
# 1. Primary generation keys (generation_mode, prompts, dimensions, seed, cfg, steps, scheduler, model, vae)
# 2. Common feature extensions and denoise parameters
# 3. Architecture-specific keys grouped alphabetically by model prefix
CORE_METADATA_REGISTRY: tuple[tuple[str, MetadataType | tuple[MetadataType, ...]], ...] = (
    # 1. Primary generation keys
    ("generation_mode", T.STRING),
    ("positive_prompt", T.STRING),
    ("positive_prompts", T.STRING),
    ("positive_style_prompt", T.STRING),
    ("negative_prompt", T.STRING),
    ("negative_prompts", T.STRING),
    ("negative_style_prompt", T.STRING),
    ("width", T.INTEGER),
    ("height", T.INTEGER),
    ("seed", T.INTEGER),
    ("rand_device", T.STRING),
    ("cfg_scale", T.FLOAT),
    ("cfg_rescale_multiplier", T.FLOAT),
    ("steps", T.INTEGER),
    ("scheduler", T.SCHEDULER),
    ("clip_skip", T.INTEGER),
    ("model", T.MODEL),
    ("vae", T.VAE),
    # 2. Common feature extensions and denoise parameters
    ("seamless_x", T.BOOLEAN),
    ("seamless_y", T.BOOLEAN),
    ("guidance", T.FLOAT),
    ("cfg_scale_start_step", T.INTEGER),
    ("cfg_scale_end_step", T.INTEGER),
    ("denoising_start", T.FLOAT),
    ("denoising_end", T.FLOAT),
    ("strength", T.FLOAT),
    ("init_image", T.STRING),
    ("add_noise", T.BOOLEAN),
    ("shift", T.FLOAT),
    ("dype_preset", T.STRING),
    ("dype_scale", T.FLOAT),
    ("dype_exponent", T.FLOAT),
    ("freeu_s1", T.FLOAT),
    ("freeu_s2", T.FLOAT),
    ("freeu_b1", T.FLOAT),
    ("freeu_b2", T.FLOAT),
    ("hidiffusion", T.BOOLEAN),
    ("hidiffusion_raunet", T.BOOLEAN),
    ("hidiffusion_window_attn", T.BOOLEAN),
    ("hidiffusion_t1_ratio", T.FLOAT),
    ("hidiffusion_t2_ratio", T.FLOAT),
    ("hrf_enabled", T.BOOLEAN),
    ("hrf_method", T.STRING),
    ("hrf_strength", T.FLOAT),
    ("refiner_model", T.MODEL),
    ("refiner_cfg_scale", T.FLOAT),
    ("refiner_steps", T.INTEGER),
    ("refiner_scheduler", T.SCHEDULER),
    ("refiner_positive_aesthetic_score", T.FLOAT),
    ("refiner_negative_aesthetic_score", T.FLOAT),
    ("refiner_start", T.FLOAT),
    ("pid_mode", T.STRING),
    ("pid_steps", T.INTEGER),
    ("pid_decoder", T.MODEL),
    ("gemma2_encoder", T.MODEL),
    ("tile_width", T.INTEGER),
    ("tile_height", T.INTEGER),
    ("tile_overlap", T.INTEGER),
    ("num_frames", T.INTEGER),
    ("fps", (T.INTEGER, T.FLOAT)),
    ("source_video_start_frame", T.INTEGER),
    ("source_video_end_frame", T.INTEGER),
    ("t5_encoder", T.MODEL),
    ("t5_encoder_model", T.MODEL),
    ("clip_embed_model", T.MODEL),
    ("clip_l_model", T.MODEL),
    ("clip_g_model", T.MODEL),
    ("mistral_encoder", T.MODEL),
    ("mistral_encoder_model", T.MODEL),
    ("qwen3_encoder", T.MODEL),
    ("qwen3_encoder_model", T.MODEL),
    ("qwen3_source", T.MODEL),
    ("qwen3_5_encoder", T.MODEL),
    ("qwen3_vl_encoder", T.MODEL),
    ("qwen_vl_encoder", T.MODEL),
    ("glm_encoder", T.MODEL),
    # 3. Architecture-specific keys (grouped alphabetically by model prefix)
    # Flux
    ("flux3_caption", T.STRING),
    # Ideogram
    ("ideogram4_caption", T.STRING),
    ("ideogram4_color_palette", T.STRING),
    ("ideogram4_guidance_scale", T.FLOAT),
    ("ideogram4_mu", T.FLOAT),
    ("ideogram4_sampler_preset", T.STRING),
    ("ideogram4_steps", T.INTEGER),
    ("ideogram4_unconditional_model", T.MODEL),
    # Krea
    ("krea2_rebalance_enabled", T.BOOLEAN),
    ("krea2_rebalance_multiplier", T.FLOAT),
    ("krea2_rebalance_weights", T.STRING),
    ("krea2_reference_kv_cache", T.BOOLEAN),
    ("krea2_seed_variance_enabled", T.BOOLEAN),
    ("krea2_seed_variance_randomize_percent", T.INTEGER),
    ("krea2_seed_variance_strength", T.FLOAT),
    ("krea2_style_strength", T.FLOAT),
    # LTX-Video
    ("ltx2_audio_cfg_scale", T.FLOAT),
    ("ltx2_base_height", T.INTEGER),
    ("ltx2_base_width", T.INTEGER),
    ("ltx2_component_source", T.MODEL),
    ("ltx2_conditioning_role", T.STRING),
    ("ltx2_context_frames", T.INTEGER),
    ("ltx2_guidance_rescale", T.FLOAT),
    ("ltx2_modality_scale", T.FLOAT),
    ("ltx2_noise_scale", T.FLOAT),
    ("ltx2_schedule", T.STRING),
    ("ltx2_stg_scale", T.FLOAT),
    ("ltx2_text_encoder_model", T.MODEL),
    ("ltx2_two_stage", T.BOOLEAN),
    # MiniMax H3
    ("minimax_h3_component_source", T.MODEL),
    ("minimax_h3_hybrid_base_model", T.MODEL),
    ("minimax_h3_hybrid_start_block", T.INTEGER),
    ("minimax_h3_text_encoder_model", T.MODEL),
    ("minimax_h3_transformer_model", T.MODEL),
    # Moss Image
    ("moss_image_layout_prompt", T.STRING),
    ("moss_image_uncond_skip_layers", T.INTEGER),
    # Qwen Image
    ("qwen_image_2_1_prefix_kv_cache", T.BOOLEAN),
    ("qwen_image_component_source", T.MODEL),
    ("qwen_image_quantization", T.STRING),
    ("qwen_image_qwen_vl_encoder", T.MODEL),
    ("qwen_image_shift", (T.STRING, T.FLOAT)),
    ("qwen_image_vae", T.VAE),
    # Wan
    ("wan_component_source", T.MODEL),
    ("wan_guidance_scale_low_noise", T.FLOAT),
    ("wan_t5_encoder_model", T.MODEL),
    ("wan_transformer_low_noise", T.MODEL),
    # Z-Image
    ("z_image_seed_variance_enabled", T.BOOLEAN),
    ("z_image_seed_variance_randomize_percent", T.INTEGER),
    ("z_image_seed_variance_strength", T.FLOAT),
    ("z_image_shift", (T.STRING, T.FLOAT)),
)


def _filter_labels(target: MetadataType) -> tuple[str, ...]:
    return tuple(
        label
        for label, types in CORE_METADATA_REGISTRY
        if types == target or (isinstance(types, tuple) and target in types)
    )


CORE_STRING_LABELS: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.STRING))
CORE_INTEGER_LABELS: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.INTEGER))
CORE_FLOAT_LABELS: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.FLOAT))
CORE_BOOL_LABELS: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.BOOLEAN))
CORE_SCHEDULER_LABELS: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.SCHEDULER))
CORE_MODEL_LABELS_TUPLE: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.MODEL))
CORE_VAE_LABELS_TUPLE: tuple[str, ...] = (CUSTOM_LABEL, *_filter_labels(T.VAE))

CORE_LABELS_TUPLE: tuple[str, ...] = (CUSTOM_LABEL, *(label for label, _ in CORE_METADATA_REGISTRY))

CORE_LABELS = Literal[CORE_LABELS_TUPLE]
CORE_LABELS_STRING = Literal[CORE_STRING_LABELS]
CORE_LABELS_INTEGER = Literal[CORE_INTEGER_LABELS]
CORE_LABELS_FLOAT = Literal[CORE_FLOAT_LABELS]
CORE_LABELS_BOOL = Literal[CORE_BOOL_LABELS]
CORE_LABELS_SCHEDULER = Literal[CORE_SCHEDULER_LABELS]
CORE_LABELS_MODEL = Literal[CORE_MODEL_LABELS_TUPLE]
CORE_LABELS_VAE = Literal[CORE_VAE_LABELS_TUPLE]

CORE_MODEL_LABELS: set[str] = {
    label
    for label, types in CORE_METADATA_REGISTRY
    if types in (T.MODEL, T.VAE) or (isinstance(types, tuple) and any(t in (T.MODEL, T.VAE) for t in types))
}

# Module-level sanity assertions
_all_labels = [label for label, _ in CORE_METADATA_REGISTRY]
assert len(_all_labels) == len(set(_all_labels)), "Duplicate label detected in CORE_METADATA_REGISTRY"
for _lbl, _t in CORE_METADATA_REGISTRY:
    _t_seq = _t if isinstance(_t, tuple) else (_t,)
    assert all(isinstance(x, MetadataType) for x in _t_seq), f"Invalid MetadataType in {_lbl}"
