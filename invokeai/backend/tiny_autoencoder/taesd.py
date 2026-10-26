"""Tiny AutoEncoder for Stable Diffusion (TAESD): the decoder half, used for denoising previews.

Adapted from madebyollin's TAESD, pinned to:
https://github.com/madebyollin/taesd/blob/fdbae2df989db95761f482efb8977e1182fb68d2/taesd.py

Kept: `conv`, `Clamp`, `Block` (with the `flux_2` mid-block pooling), the default 8x `Decoder` and the
16x `F16Decoder` that TAEQI2.1 uses. Left out: the encoders, the 32x Sana decoder, and `SuperBlock`
(the "tinier" TAEQI2.1), none of which InvokeAI loads. The decoders return a `TaesdDecoder` -- the
upstream `nn.Sequential` plus the two facts a caller needs to feed it -- instead of a bare Sequential.

The `nn.Sequential` indices are the state-dict contract with the released decoder-only checkpoints
(`1.weight`, `3.conv.0.weight`, ...): do not reorder, insert or remove layers. Output is upstream's
convention: image channels in roughly [0, 1], unclamped.

Original license:

MIT License

Copyright (c) 2023 Ollin Boer Bohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn

from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder


def conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_midblock_gn: bool = False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                nn.Conv2d(n_in, n_gn, 1, bias=False),
                nn.GroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_gn, n_in, 1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


class TaesdDecoder(TinyAutoencoderDecoder, nn.Sequential):
    """An upstream decoder `nn.Sequential`, and what its input and output look like."""

    def __init__(self, *layers: nn.Module, latent_channels: int, spatial_compression: int):
        super().__init__(*layers)
        self.latent_channels = latent_channels
        self.spatial_compression = spatial_compression
        # At 1024 a decode costs ~640 MiB of VRAM for the 8x TAEs (~160 MiB for TAEQI2.1) and ~25 ms (RTX
        # 4090, fp16) at any generation size; uncapped, 2048 costs ~2.5 GiB and ~100 ms. Decoding at full
        # size and scaling down would look better (SDXL 36.3 vs 30.7 dB against the VAE) but carries that
        # uncapped cost, so `decode_preview` area-downsamples the latent instead.
        self.max_preview_edge = 1024


def Decoder(latent_channels: int = 4, use_midblock_gn: bool = False, image_channels: int = 3) -> TaesdDecoder:
    mb_kw = {"use_midblock_gn": use_midblock_gn}
    return TaesdDecoder(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels),
        latent_channels=latent_channels,
        spatial_compression=8,
    )  # fmt: skip


def F16Decoder(latent_channels: int = 64, image_channels: int = 3) -> TaesdDecoder:
    """Decoder variant with 16x spatial upscaling, wider low-res stages, and 2x2 output patchify."""
    return TaesdDecoder(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels * 4), nn.PixelShuffle(2),
        latent_channels=latent_channels,
        spatial_compression=16,
    )  # fmt: skip
