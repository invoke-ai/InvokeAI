"""Dynamic Position Extrapolation (DyPE) for FLUX models.

DyPE enables high-resolution image generation (4K+) with pretrained FLUX models
by dynamically scaling RoPE position embeddings during the denoising process.

Based on: https://github.com/wildminder/ComfyUI-DyPE
"""

from invokeai.backend.flux.dype.base import DyPEConfig
from invokeai.backend.flux.dype.embed import DyPEEmbedND
from invokeai.backend.flux.dype.presets import (
    DYPE_PRESET_4K,
    DYPE_PRESET_AREA,
    DYPE_PRESET_AUTO,
    DYPE_PRESET_LABELS,
    DYPE_PRESET_MANUAL,
    DYPE_PRESET_OFF,
    DyPEPreset,
    get_dype_config_for_area,
    get_dype_config_for_resolution,
)

__all__ = [
    "DyPEConfig",
    "DyPEEmbedND",
    "DyPEPreset",
    "DYPE_PRESET_OFF",
    "DYPE_PRESET_MANUAL",
    "DYPE_PRESET_AUTO",
    "DYPE_PRESET_AREA",
    "DYPE_PRESET_4K",
    "DYPE_PRESET_LABELS",
    "get_dype_config_for_area",
    "get_dype_config_for_resolution",
]
