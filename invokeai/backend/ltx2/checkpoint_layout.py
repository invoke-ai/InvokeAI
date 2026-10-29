"""How the released LTX-2 single files are told apart.

Lightricks and the mirrors that repack their weights (WanGP's ``DeepBeepMeep/LTX-2``, the ComfyUI
repacks) ship one safetensors file per component, all in the official (non-diffusers) key layout:
the transformer under ``model.diffusion_model.*``, the text connectors under the same prefix, the
video VAE bare (``encoder.*`` / ``decoder.*``), the audio VAE under ``audio_vae.*`` and the vocoder
under ``vocoder.*``. Some releases bundle several of these into one file. This module classifies a
file by its key names alone -- the header is read, never the tensors -- so both identification
(``configs/main.py``) and the loaders share one notion of "which component is this".

Nothing here imports torch or diffusers: identification runs in the installer's probe, which must
stay cheap and import-light.
"""

import json
from collections.abc import Iterable, Mapping
from typing import Any, Final

TRANSFORMER_PREFIX: Final = "model.diffusion_model."
"""Prefix the official layout puts on transformer and connector keys."""

# Component roles a single file can carry. Values double as the keys of a folder record's
# ``components`` map, so they are persisted in the model database: never rename one.
ROLE_TRANSFORMER: Final = "transformer"
ROLE_VIDEO_CONNECTOR: Final = "video_connector"
ROLE_AUDIO_CONNECTOR: Final = "audio_connector"
ROLE_TEXT_PROJECTION: Final = "text_projection"
ROLE_VIDEO_VAE: Final = "video_vae"
ROLE_DIFFUSION_VIDEO_VAE: Final = "diffusion_video_vae"
ROLE_AUDIO_VAE: Final = "audio_vae"
ROLE_VOCODER: Final = "vocoder"
ROLE_SPATIAL_UPSAMPLER: Final = "spatial_upsampler"
ROLE_TEMPORAL_UPSAMPLER: Final = "temporal_upsampler"

ALL_ROLES: Final = frozenset(
    {
        ROLE_TRANSFORMER,
        ROLE_VIDEO_CONNECTOR,
        ROLE_AUDIO_CONNECTOR,
        ROLE_TEXT_PROJECTION,
        ROLE_VIDEO_VAE,
        ROLE_DIFFUSION_VIDEO_VAE,
        ROLE_AUDIO_VAE,
        ROLE_VOCODER,
        ROLE_SPATIAL_UPSAMPLER,
        ROLE_TEMPORAL_UPSAMPLER,
    }
)

# The transformer fingerprint: the audio->video cross-attention gate's timestep embedder exists in
# no other family probed here. Diffusers' single-file detector keys on the same tensor.
TRANSFORMER_FINGERPRINT_KEY: Final = "av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight"

_VIDEO_VAE_MARKER: Final = "encoder.conv_in.conv.weight"
_AUDIO_VAE_PREFIX: Final = "audio_vae."
_VOCODER_PREFIX: Final = "vocoder."
_VIDEO_CONNECTOR_PREFIX: Final = "video_embeddings_connector."
_AUDIO_CONNECTOR_PREFIX: Final = "audio_embeddings_connector."
_TEXT_PROJECTION_PREFIX: Final = "text_embedding_projection."
_UPSAMPLER_MARKER: Final = "initial_conv.weight"


def strip_transformer_prefix(key: str) -> str:
    """``model.diffusion_model.x`` -> ``x``; other keys pass through."""
    return key[len(TRANSFORMER_PREFIX) :] if key.startswith(TRANSFORMER_PREFIX) else key


def header_config(metadata: Mapping[str, str] | None) -> dict[str, Any]:
    """The ``config`` JSON the official files embed in their safetensors header, or ``{}``."""
    raw = (metadata or {}).get("config")
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def header_model_version(metadata: Mapping[str, str] | None) -> str | None:
    """The ``model_version`` the official files embed (``"2.5.0"``), or None."""
    version = (metadata or {}).get("model_version")
    return version if isinstance(version, str) and version else None


def generation_from_version(version: str | None) -> str | None:
    """``"2.5.0"`` -> ``"2.5"``; None when the string is not ``major.minor[...]``."""
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None
    return f"{parts[0]}.{parts[1]}"


def classify_roles(
    state_dict: Mapping[str, Any] | Iterable[str], metadata: Mapping[str, str] | None = None
) -> set[str]:
    """Which components a file holds, from its key names.

    ``state_dict`` may be a real or header-only (meta tensor) state dict, or just its keys; tensor
    ranks are consulted only to tell the two x2 latent upsamplers apart when the header cannot.
    A file may carry several roles (the bundled 2.0/2.3 releases hold transformer, VAEs and more in
    one file); an unrelated file yields the empty set.
    """
    values: Mapping[str, Any] = state_dict if isinstance(state_dict, Mapping) else {}
    keys = state_dict.keys() if isinstance(state_dict, Mapping) else state_dict
    stripped = {strip_transformer_prefix(k) for k in keys if isinstance(k, str)}
    roles: set[str] = set()

    if TRANSFORMER_FINGERPRINT_KEY in stripped and any(k.startswith("transformer_blocks.") for k in stripped):
        roles.add(ROLE_TRANSFORMER)
    if any(k.startswith(_VIDEO_CONNECTOR_PREFIX) for k in stripped):
        roles.add(ROLE_VIDEO_CONNECTOR)
    if any(k.startswith(_AUDIO_CONNECTOR_PREFIX) for k in stripped):
        roles.add(ROLE_AUDIO_CONNECTOR)
    if any(k.startswith(_TEXT_PROJECTION_PREFIX) for k in stripped):
        roles.add(ROLE_TEXT_PROJECTION)
    if any(k.startswith(_AUDIO_VAE_PREFIX) for k in stripped):
        roles.add(ROLE_AUDIO_VAE)
    if any(k.startswith(_VOCODER_PREFIX) for k in stripped):
        roles.add(ROLE_VOCODER)

    # The video VAE is bare in the split files and under ``vae.`` in the bundled ones.
    if _VIDEO_VAE_MARKER in stripped or f"vae.{_VIDEO_VAE_MARKER}" in stripped:
        if any(k.startswith(("decoder.diff_blocks.", "vae.decoder.diff_blocks.")) for k in stripped):
            roles.add(ROLE_DIFFUSION_VIDEO_VAE)
        else:
            roles.add(ROLE_VIDEO_VAE)

    if _UPSAMPLER_MARKER in stripped and "res_blocks.0.conv1.weight" in stripped and "final_conv.weight" in stripped:
        config = header_config(metadata)
        if config.get("temporal_upsample") is True:
            roles.add(ROLE_TEMPORAL_UPSAMPLER)
        elif config.get("spatial_upsample") is True:
            roles.add(ROLE_SPATIAL_UPSAMPLER)
        else:
            # Without the header flags the two x2 files are told apart by their upsampling conv: the
            # spatial x2 file's pixel-shuffle conv is 2-D (kernel 3x3), the temporal one's is 3-D.
            conv = values.get("upsampler.0.weight")
            roles.add(ROLE_TEMPORAL_UPSAMPLER if getattr(conv, "ndim", 4) == 5 else ROLE_SPATIAL_UPSAMPLER)

    return roles


def is_distilled_filename(name: str) -> bool:
    """Dev and Distilled transformers are key-for-key identical; the filename is the classifier."""
    return "distilled" in name.lower()
