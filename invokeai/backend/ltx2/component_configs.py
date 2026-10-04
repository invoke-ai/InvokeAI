"""The diffusers configs and key maps for the released LTX-2.5 components.

The official files embed a config in their safetensors header, but it is the *official* code's
config (``AVTransformer3DModel``, ``CausalVideoAutoencoder``, ...), not diffusers', and the
diffusers default configs describe LTX-2.0. Each mapping here was derived by instantiating the
diffusers class on the meta device and matching every parameter name and shape against the released
2.5 file (0 missing, 0 unexpected, 0 shape mismatches for every component); the loaders re-check the
shapes on the real tensors, so a differently built file fails at load time with a named tensor
rather than at the first forward.

Key maps go from the official layout (``model.diffusion_model.*``, ``vae.*``, ...) to diffusers
attribute paths. Diffusers ships converters for the transformer and the two VAEs
(``diffusers.loaders.single_file_utils.convert_ltx2_*``); they are reused where they are complete and
patched where they stop short of 2.5 (the transformer's prompt AdaLN, the VAE's fourth decoder
stage). The connectors, text projection and vocoder have no diffusers converter at all.
"""

import re
from collections.abc import Mapping
from typing import Any, Final

# --- Transformer ---------------------------------------------------------------------------------

LTX2_5_TRANSFORMER_CONFIG: Final[dict[str, Any]] = {
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "cross_attention_dim": 4096,
    "vae_scale_factors": (8, 32, 32),
    "pos_embed_max_pos": 20,
    "base_height": 2048,
    "base_width": 2048,
    # LTX-2.3+ gated self-attention and prompt-KV modulation, on both streams.
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_in_channels": 128,
    "audio_out_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_hop_length": 160,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "num_layers": 48,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 3840,
    "attention_bias": True,
    "attention_out_bias": True,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_offset": 1,
    "timestep_scale_multiplier": 1000,
    "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split",
    # 2.3+ project the caption in the connectors, not in the transformer.
    "use_prompt_embeddings": False,
    # Selects the self-attention processor that can skip a block for spatio-temporal guidance.
    # It carries no weights of its own and is arithmetically identical to the plain processor
    # when no perturbation is asked for, so it is built in unconditionally rather than swapped
    # onto a shared cached model for the one pass per step that uses it.
    "perturbed_attn": True,
    # 2.5 dropped the video feed-forward biases but kept the audio ones.
    "ff_bias": False,
    "audio_ff_bias": True,
    "use_prompt_adaln_single": True,
    "use_keyframes_abs_pos_embedding": True,
}

# What ``convert_ltx2_transformer_to_diffusers`` leaves in the official spelling: the prompt
# cross-attention AdaLN heads (its rename table only covers the ``adaln_single`` pair).
TRANSFORMER_RENAMES_AFTER_DIFFUSERS: Final[tuple[tuple[str, str], ...]] = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
)

# Components the official transformer files may bundle beside the transformer; dropped by the
# transformer loader, they are loaded from the component folder instead.
TRANSFORMER_FOREIGN_PREFIXES: Final[tuple[str, ...]] = (
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
    "vae.",
    "audio_vae.",
    "vocoder.",
    "text_embedding_projection.",
)


def finish_transformer_keys(sd: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the renames diffusers' transformer converter misses; returns (sd, key_map) where
    ``key_map`` records every key that changed (old -> new)."""
    key_map: dict[str, str] = {}
    out: dict[str, Any] = {}
    for key, value in sd.items():
        new_key = key
        for old, new in TRANSFORMER_RENAMES_AFTER_DIFFUSERS:
            if new_key.startswith(old):
                new_key = new + new_key[len(old) :]
                break
        if new_key != key:
            key_map[key] = new_key
        out[new_key] = value
    return out, key_map


# --- Video VAE -----------------------------------------------------------------------------------

LTX2_5_VIDEO_VAE_CONFIG: Final[dict[str, Any]] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    # Encoder: 4 down blocks (128->256 spatial, ->512 temporal, ->1024 spatiotemporal,
    # ->1024 spatiotemporal) then a 1024-wide mid block; 4/6/4/2 resnets per block, 2 in the mid.
    "block_out_channels": (256, 512, 1024, 1024),
    "layers_per_block": (4, 6, 4, 2, 2),
    "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    "spatio_temporal_scaling": (True, True, True, True),
    # Decoder (outermost first, as diffusers lists it): 128 <-spatial- 256 <-temporal- 512
    # <-all(x1, channel preserving)- 512 <-all(x2)- 1024 mid. The third stage keeps its width, which
    # is what the unit upsample factor encodes.
    "decoder_block_out_channels": (256, 512, 512, 1024),
    "decoder_layers_per_block": (4, 6, 4, 2, 2),
    "decoder_spatio_temporal_scaling": (True, True, True, True),
    "decoder_inject_noise": (False, False, False, False, False),
    "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    "upsample_residual": (False, False, False, False),
    "upsample_factor": (2, 2, 1, 2),
    "timestep_conditioning": False,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "scaling_factor": 1.0,
    "encoder_causal": True,
    # The 2.5 header says ``causal_decoder: false`` and ``spatial_padding_mode: zeros`` for both halves.
    "decoder_causal": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "zeros",
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
}

# Official flat block indices -> diffusers nested paths. The encoder half is what diffusers' own
# converter does; the decoder half extends it by the fourth upsampling stage 2.5 added.
_VIDEO_VAE_BLOCK_RENAMES: Final[dict[str, str]] = {
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.down_blocks.1": "encoder.down_blocks.0.downsamplers.0",
    "encoder.down_blocks.2": "encoder.down_blocks.1",
    "encoder.down_blocks.3": "encoder.down_blocks.1.downsamplers.0",
    "encoder.down_blocks.4": "encoder.down_blocks.2",
    "encoder.down_blocks.5": "encoder.down_blocks.2.downsamplers.0",
    "encoder.down_blocks.6": "encoder.down_blocks.3",
    "encoder.down_blocks.7": "encoder.down_blocks.3.downsamplers.0",
    "encoder.down_blocks.8": "encoder.mid_block",
    "decoder.up_blocks.0": "decoder.mid_block",
    "decoder.up_blocks.1": "decoder.up_blocks.0.upsamplers.0",
    "decoder.up_blocks.2": "decoder.up_blocks.0",
    "decoder.up_blocks.3": "decoder.up_blocks.1.upsamplers.0",
    "decoder.up_blocks.4": "decoder.up_blocks.1",
    "decoder.up_blocks.5": "decoder.up_blocks.2.upsamplers.0",
    "decoder.up_blocks.6": "decoder.up_blocks.2",
    "decoder.up_blocks.7": "decoder.up_blocks.3.upsamplers.0",
    "decoder.up_blocks.8": "decoder.up_blocks.3",
}
_VIDEO_VAE_BLOCK_KEY = re.compile(r"^((?:encoder\.down_blocks|decoder\.up_blocks)\.\d+)(\..*)$")
_VIDEO_VAE_STATS_RENAMES: Final[dict[str, str]] = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}
_VIDEO_VAE_DROPPED_KEYS: Final[frozenset[str]] = frozenset(
    {"per_channel_statistics.channel", "per_channel_statistics.mean-of-stds"}
)


def convert_video_vae_keys(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Official video VAE keys (bare or under ``vae.``) -> diffusers ``AutoencoderKLLTX2Video`` keys."""
    out: dict[str, Any] = {}
    for key, value in sd.items():
        key = key[len("vae.") :] if key.startswith("vae.") else key
        if key in _VIDEO_VAE_DROPPED_KEYS:
            continue
        if key in _VIDEO_VAE_STATS_RENAMES:
            out[_VIDEO_VAE_STATS_RENAMES[key]] = value
            continue
        match = _VIDEO_VAE_BLOCK_KEY.match(key)
        if match:
            block, rest = match.groups()
            key = _VIDEO_VAE_BLOCK_RENAMES[block] + rest
        out[key.replace(".res_blocks.", ".resnets.")] = value
    return out


# --- Audio VAE -----------------------------------------------------------------------------------

# The released audio VAE is exactly diffusers' ``AutoencoderKLLTX2Audio`` default (8 latent
# channels, 64 mel bins, 16 kHz, hop 160, causal along the time axis).
LTX2_5_AUDIO_VAE_CONFIG: Final[dict[str, Any]] = {}


def convert_audio_vae_keys(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Official ``audio_vae.*`` keys -> diffusers ``AutoencoderKLLTX2Audio`` keys."""
    out: dict[str, Any] = {}
    for key, value in sd.items():
        key = key[len("audio_vae.") :] if key.startswith("audio_vae.") else key
        out[_VIDEO_VAE_STATS_RENAMES.get(key, key)] = value
    return out


# --- Vocoder -------------------------------------------------------------------------------------

# 2.5 ships the 48 kHz vocoder: the 24 kHz HiFi-GAN followed by a bandwidth-extension generator,
# which is diffusers' ``LTX2VocoderWithBWE`` default.
LTX2_5_VOCODER_CONFIG: Final[dict[str, Any]] = {}

_VOCODER_RENAMES: Final[tuple[tuple[str, str], ...]] = (
    (".conv_pre.", ".conv_in."),
    (".conv_post.", ".conv_out."),
    (".act_post.", ".act_out."),
    (".ups.", ".upsamplers."),
    (".resblocks.", ".resnets."),
    (".downsample.lowpass.filter", ".downsample.filter"),
)


def convert_vocoder_keys(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Official ``vocoder.{vocoder,bwe_generator,mel_stft}.*`` keys -> diffusers ``LTX2VocoderWithBWE``.

    The anti-aliasing ``resampler.filter`` buffer is computed by the constructor and is not stored.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        key = key[len("vocoder.") :] if key.startswith("vocoder.") else key
        key = "." + key
        for old, new in _VOCODER_RENAMES:
            key = key.replace(old, new)
        out[key[1:]] = value
    return out


# --- Text connectors -----------------------------------------------------------------------------

LTX2_5_CONNECTORS_CONFIG: Final[dict[str, Any]] = {
    "caption_channels": 3840,
    # 48 Gemma-4 layers + the embedding output, all stacked.
    "text_proj_in_factor": 49,
    "video_connector_num_attention_heads": 32,
    "video_connector_attention_head_dim": 128,
    "video_connector_num_layers": 8,
    "video_connector_num_learnable_registers": 128,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 32,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_layers": 8,
    "audio_connector_num_learnable_registers": 128,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 4096,
    "audio_hidden_dim": 2048,
    "proj_bias": True,
}

_CONNECTOR_RENAMES: Final[tuple[tuple[str, str], ...]] = (
    ("video_embeddings_connector.", "video_connector."),
    ("audio_embeddings_connector.", "audio_connector."),
    ("text_embedding_projection.video_aggregate_embed.", "video_text_proj_in."),
    ("text_embedding_projection.audio_aggregate_embed.", "audio_text_proj_in."),
)
_CONNECTOR_INNER_RENAMES: Final[tuple[tuple[str, str], ...]] = (
    (".transformer_1d_blocks.", ".transformer_blocks."),
    (".q_norm.", ".norm_q."),
    (".k_norm.", ".norm_k."),
)


def convert_connector_keys(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Official connector / projection keys (any of the three files) -> ``LTX2TextConnectors`` keys.

    Keys that belong to none of the connector prefixes are dropped: the official connector files
    carry nothing else, and a bundled transformer file's other tensors are not this component's.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        key = key[len("model.diffusion_model.") :] if key.startswith("model.diffusion_model.") else key
        for old, new in _CONNECTOR_RENAMES:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        else:
            continue
        for old, new in _CONNECTOR_INNER_RENAMES:
            key = key.replace(old, new)
        out[key] = value
    return out


# --- Latent upsamplers ---------------------------------------------------------------------------

# Both x2 upsamplers use diffusers' own key names; only their configs differ.
LTX2_5_SPATIAL_UPSAMPLER_CONFIG: Final[dict[str, Any]] = {
    "in_channels": 128,
    "mid_channels": 1024,
    "num_blocks_per_stage": 4,
    "dims": 3,
    "spatial_upsample": True,
    "temporal_upsample": False,
    "rational_spatial_scale": 2.0,
    "use_rational_resampler": False,
}
LTX2_5_TEMPORAL_UPSAMPLER_CONFIG: Final[dict[str, Any]] = {
    "in_channels": 128,
    "mid_channels": 512,
    "num_blocks_per_stage": 4,
    "dims": 3,
    "spatial_upsample": False,
    "temporal_upsample": True,
    "rational_spatial_scale": 1.0,
    "use_rational_resampler": True,
}
