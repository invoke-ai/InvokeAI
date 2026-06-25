"""Ideogram 4 backend.

The model modules (``modeling_ideogram4``, ``autoencoder``, ``latent_norm``,
``scheduler``, ``sampler_configs``, ``constants``, ``quantized_loading``) are
adapted from the Apache-2.0 Ideogram 4 reference implementation
(https://github.com/ideogram-oss/ideogram4). See ``NOTICE.md``. The remaining
modules wrap that model for InvokeAI invocations.

``quantized_loading`` is intentionally not re-exported here so that importing this
package does not eagerly import ``bitsandbytes``; import it directly where needed.
"""

from invokeai.backend.ideogram4.denoise import run_ideogram4_denoise
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.sampler_configs import PRESETS
from invokeai.backend.ideogram4.sampling_utils import (
    AE_SCALE_FACTOR,
    LATENT_DIM,
    PATCH_SIZE,
    PIXELS_PER_IMAGE_TOKEN,
    build_denoise_inputs,
    pack_latents_to_grid,
    unpatchify_and_denormalize,
    validate_dimensions,
)
from invokeai.backend.ideogram4.text_encoding import MAX_TEXT_TOKENS, encode_qwen3vl_prompt

__all__ = [
    "Ideogram4Config",
    "Ideogram4Transformer",
    "PRESETS",
    "run_ideogram4_denoise",
    "encode_qwen3vl_prompt",
    "MAX_TEXT_TOKENS",
    "build_denoise_inputs",
    "pack_latents_to_grid",
    "unpatchify_and_denormalize",
    "validate_dimensions",
    "AE_SCALE_FACTOR",
    "LATENT_DIM",
    "PATCH_SIZE",
    "PIXELS_PER_IMAGE_TOKEN",
]
