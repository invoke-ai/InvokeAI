"""What preview rendering needs from a Tiny AutoEncoder decoder, whichever family it belongs to."""

from typing import Final

import torch
from torch import nn

PREVIEW_FIXED_ALLOWANCE_BYTES: Final = 5 * 2**20
"""What every preview decode's working-memory bound adds whatever its size: the small allocations beside the
activations and the workspace. On an RTX 4090 the largest came from TAEF2, whose midblock pooling adds its own: 1.07 MiB
at its 1024 cap and up to 4.07 MiB with the cap off (a 2048x2048 decode peaked at 2564.07 MiB against 2560)."""


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
        decoder's element size, plus `PREVIEW_FIXED_ALLOWANCE_BYTES`."""
        raise NotImplementedError

    def decode_window(self, window: torch.Tensor) -> torch.Tensor:
        """The image of the last frame of a `(B, C, T, h, w)` latent window, the earlier ones as its context."""
        raise ValueError(f"{type(self).__name__} decodes single latents, not a (B, C, T, h, w) window of latent frames")
