"""The FLUX.2 VAE's latent normalization, shared by every architecture that decodes with that VAE.

The FLUX.2 autoencoder normalizes its 2x2-packed, 128-channel latents with a BatchNorm whose running
statistics ship in the checkpoint (``affine=False``, so there is no learned weight or bias). The
transformers that decode with it (FLUX.2, FLUX.3) denoise in the normalized space, while latents travel
between nodes un-normalized: a denoise node normalizes what it takes in and denormalizes what it returns.
"""

import torch

_DEFAULT_BN_EPS = 1e-4  # what BFL's autoencoder uses


def read_vae_bn_stats(vae: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(mean, std)`` of the VAE's latent BatchNorm, each of shape ``(128,)``, or None if it has none.

    The layer sits at a different attribute depending on the format the VAE was loaded from.
    """
    vae.eval()  # never update the running statistics
    bn_layer = None
    if hasattr(vae, "bn"):
        bn_layer = vae.bn
    elif hasattr(vae, "batch_norm"):
        bn_layer = vae.batch_norm
    elif hasattr(vae, "encoder") and hasattr(vae.encoder, "bn"):
        bn_layer = vae.encoder.bn

    if bn_layer is None or bn_layer.running_mean is None or bn_layer.running_var is None:
        return None

    bn_mean = bn_layer.running_mean.clone()
    bn_var = bn_layer.running_var.clone()
    bn_eps = bn_layer.eps if hasattr(bn_layer, "eps") else _DEFAULT_BN_EPS
    return bn_mean, torch.sqrt(bn_var + bn_eps)


def bn_normalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
    """``(x - mean) / std`` over the last (channel) axis of packed ``(B, seq, 128)`` latents."""
    return (x - bn_mean.to(x.device, x.dtype)) / bn_std.to(x.device, x.dtype)


def bn_denormalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
    """``x * std + mean``, the inverse of :func:`bn_normalize`."""
    return x * bn_std.to(x.device, x.dtype) + bn_mean.to(x.device, x.dtype)
