"""What preview rendering needs from a Tiny AutoEncoder decoder, whichever family it belongs to."""

import torch
from torch import nn


class TinyAutoencoderDecoder(nn.Module):
    """A TAE decoder: `(B, latent_channels, h, w)` latents in, `(B, image channels, H, W)` images out.

    `H` and `W` are `h` and `w` times `spatial_compression`; values are about [0, 1], and callers clamp.
    Video decoders (TAEHV) decode each latent as a single frame, and a window of latent frames through
    `decode_window`.
    """

    latent_channels: int
    spatial_compression: int
    max_preview_edge: int
    """Longest preview edge, in pixels, this decoder renders; a larger latent is downsampled first."""

    def decode_window(self, window: torch.Tensor) -> torch.Tensor:
        """The image of the last frame of a `(B, C, T, h, w)` latent window, the earlier ones as its context."""
        raise ValueError(f"{type(self).__name__} decodes single latents, not a (B, C, T, h, w) window of latent frames")
