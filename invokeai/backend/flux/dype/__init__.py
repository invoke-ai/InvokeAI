"""Dynamic Position Extrapolation (DyPE) for FLUX models.

DyPE enables high-resolution image generation (4K+) with pretrained FLUX models
by dynamically scaling RoPE position embeddings during the denoising process.

Based on: https://github.com/wildminder/ComfyUI-DyPE
"""

from invokeai.backend.flux.dype.base import DyPEConfig
from invokeai.backend.flux.dype.embed import DyPEEmbedND
from invokeai.backend.flux.dype.presets import DyPEPreset, get_dype_config_for_resolution

__all__ = [
    "DyPEConfig",
    "DyPEEmbedND",
    "DyPEPreset",
    "get_dype_config_for_resolution",
]
