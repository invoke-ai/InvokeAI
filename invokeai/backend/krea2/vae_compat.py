"""Compatibility helpers for the Qwen-Image VAE used by Krea-2.

Krea-2 (and Qwen-Image) decode/encode with ``AutoencoderKLQwenImage``. A standalone single-file
``qwen_image_vae.safetensors`` in the native (ComfyUI/Wan) layout is byte-identical to the Anima VAE
and therefore classified with the Anima base, which loads it as ``AutoencoderKLWan``. The two classes
share the exact same diffusers state-dict (identical keys and shapes), so a Wan-loaded VAE can be
reinterpreted as ``AutoencoderKLQwenImage`` losslessly — and the default ``AutoencoderKLQwenImage``
config carries the correct Qwen-Image ``latents_mean`` / ``latents_std`` / ``z_dim`` that the qwen
encode/decode nodes read.
"""

from typing import Any

import accelerate
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


def as_qwen_image_vae(model: Any) -> AutoencoderKLQwenImage:
    """Return ``model`` if it is already an ``AutoencoderKLQwenImage``, else reinterpret it as one.

    The only expected non-matching input is ``AutoencoderKLWan`` (the same weights loaded via the
    Anima single-file path). Its state dict is loaded — with ``assign=True`` so no tensors are copied
    and device/dtype are preserved — into a freshly built ``AutoencoderKLQwenImage`` whose default
    config provides the correct Qwen-Image latent statistics.
    """
    if isinstance(model, AutoencoderKLQwenImage):
        return model

    src_state_dict = model.state_dict()
    with accelerate.init_empty_weights():
        qwen_vae = AutoencoderKLQwenImage()
    # assign=True shares the source tensors (no copy) and keeps their device/dtype.
    qwen_vae.load_state_dict(src_state_dict, strict=True, assign=True)
    # Match the eval/grad state of a normally-loaded VAE.
    qwen_vae.eval()
    qwen_vae.requires_grad_(False)
    return qwen_vae
