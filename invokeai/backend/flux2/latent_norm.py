"""The FLUX.2 VAE's latent normalization, shared by every architecture that decodes with that VAE.

The FLUX.2 autoencoder normalizes its 2x2-packed, 128-channel latents with a BatchNorm whose running
statistics ship in the checkpoint (``affine=False``, so there is no learned weight or bias). The
transformers that decode with it (FLUX.2, FLUX.3) denoise in the normalized space, while latents travel
between nodes un-normalized: a denoise node normalizes what it takes in and denormalizes what it returns.
The statistics come from ``invokeai.backend.flux2.sampling_utils.get_flux2_bn_stats``.
"""

import torch


def bn_normalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
    """``(x - mean) / std`` over the last (channel) axis of packed ``(B, seq, 128)`` latents."""
    return (x - bn_mean.to(x.device, x.dtype)) / bn_std.to(x.device, x.dtype)


def bn_denormalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
    """``x * std + mean``, the inverse of :func:`bn_normalize`."""
    return x * bn_std.to(x.device, x.dtype) + bn_mean.to(x.device, x.dtype)
