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


# The Qwen-Image VAE is a 16-channel, 8x-spatial, non-patchified autoencoder. A Wan VAE can only share the
# Qwen-Image encode/decode path if it has that exact geometry. Wan 2.2's VAE is a *different* architecture
# (48 latent channels, patchified) that happens to load as AutoencoderKLWan too, but would fail on the
# 16-vs-48 per-channel normalization (or silently produce incompatible latents), so it must be rejected.
_QWEN_IMAGE_VAE_Z_DIM = 16
_QWEN_IMAGE_VAE_SPATIAL_SCALE = 8


def as_qwen_image_vae(model: Any) -> QwenImageCompatibleVAE:
    """Return a cache-preserving VAE compatible with the Qwen-Image encode/decode path.

    The only expected non-matching input is ``AutoencoderKLWan`` (the same weights loaded via the
    Anima single-file path). A Wan VAE with the Qwen-Image geometry (16 latent channels, 8x spatial, no
    patchification) has identical encode/decode behavior, state-dict layout, and default latent statistics,
    so the cached module can be used directly. A Wan VAE with any other geometry (e.g. Wan 2.2's 48-channel,
    patchified VAE) is rejected here rather than failing deeper in normalization/decode.

    Returning the original object is important: the model cache injects custom modules for partial
    loading before this helper is called, and rebuilding the module from its state dict would discard
    those modules along with any hooks or layerwise-casting configuration.
    """
    if isinstance(model, AutoencoderKLQwenImage):
        return model
    if not isinstance(model, AutoencoderKLWan):
        raise TypeError(f"Expected AutoencoderKLQwenImage or AutoencoderKLWan, got {type(model).__name__}.")

    config = model.config
    z_dim = getattr(config, "z_dim", None)
    patch_size = getattr(config, "patch_size", None)
    spatial_scale = getattr(config, "scale_factor_spatial", _QWEN_IMAGE_VAE_SPATIAL_SCALE)
    if z_dim != _QWEN_IMAGE_VAE_Z_DIM or patch_size is not None or spatial_scale != _QWEN_IMAGE_VAE_SPATIAL_SCALE:
        raise ValueError(
            "AutoencoderKLWan is not Qwen-Image-compatible "
            f"(z_dim={z_dim}, patch_size={patch_size}, scale_factor_spatial={spatial_scale}); "
            f"expected {_QWEN_IMAGE_VAE_Z_DIM} latent channels, {_QWEN_IMAGE_VAE_SPATIAL_SCALE}x spatial, "
            "and no patchification."
        )

    return model
