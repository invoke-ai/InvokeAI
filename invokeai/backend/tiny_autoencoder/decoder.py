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

    def preview_working_memory_bytes(self, height: int, width: int, frames: int) -> int:
        """What decoding a `(height, width)` latent, or a window of `frames` latent frames, takes on the compute device
        besides the weights: an upper bound on its live activations and its convolutions' workspace, at the
        decoder's element size."""
        raise NotImplementedError

    def decode_window(self, window: torch.Tensor) -> torch.Tensor:
        """The image of the last frame of a `(B, C, T, h, w)` latent window, the earlier ones as its context."""
        raise ValueError(f"{type(self).__name__} decodes single latents, not a (B, C, T, h, w) window of latent frames")
