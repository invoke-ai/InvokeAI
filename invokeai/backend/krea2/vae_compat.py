"""Compatibility helpers for the Qwen-Image VAE used by Krea-2.

Krea-2 (and Qwen-Image) decode/encode with ``AutoencoderKLQwenImage``. A standalone single-file
``qwen_image_vae.safetensors`` in the native (ComfyUI/Wan) layout is byte-identical to the Anima VAE
and therefore classified with the Anima base, which loads it as ``AutoencoderKLWan``. The two classes
share the exact same diffusers state-dict (identical keys and shapes), so a Wan-loaded VAE can be
used through the same encode/decode path without rebuilding it. Both default configs carry the same
Qwen-Image ``latents_mean`` / ``latents_std`` / ``z_dim`` values read by the Qwen encode/decode nodes.
"""

from typing import Any

from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

QwenImageCompatibleVAE = AutoencoderKLQwenImage | AutoencoderKLWan


def as_qwen_image_vae(model: Any) -> QwenImageCompatibleVAE:
    """Return a cache-preserving VAE compatible with the Qwen-Image encode/decode path.

    The only expected non-matching input is ``AutoencoderKLWan`` (the same weights loaded via the
    Anima single-file path). The Wan and Qwen-Image classes have identical encode/decode behavior,
    state-dict layouts, and default latent statistics, so the cached module can be used directly.

    Returning the original object is important: the model cache injects custom modules for partial
    loading before this helper is called, and rebuilding the module from its state dict would discard
    those modules along with any hooks or layerwise-casting configuration.
    """
    if isinstance(model, AutoencoderKLQwenImage):
        return model
    if not isinstance(model, AutoencoderKLWan):
        raise TypeError(f"Expected AutoencoderKLQwenImage or AutoencoderKLWan, got {type(model).__name__}.")

    return model
