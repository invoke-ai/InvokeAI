"""Tiny AutoEncoder for Hunyuan Video (TAEHV): the decoder half, used for denoising previews.

Adapted from madebyollin's TAEHV, pinned to:
https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/taehv.py

Kept: `conv`, `Clamp`, `MemBlock`, `TGrow`, the base decoder `nn.Sequential`, the parallel
memory-block application and `decode_video`'s trim. Left out: the encoder, the "super" decoder
(`SuperMemBlock`), the sequential and streaming paths, and `patch_tgrow_layers`, which no supported
checkpoint needs. The `decoder.N` indices are the state-dict contract with the released checkpoints:
do not reorder, insert or remove layers.

Original license:

MIT License

Copyright (c) 2025 Ollin Boer Bohan

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
import torch.nn.functional as F

from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder


def conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block whose conv also sees the previous frame's input (the temporal memory)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Temporal upsampling: a 1x1 conv to `stride` channel groups, reshaped into frames."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class TaehvDecoder(TinyAutoencoderDecoder):
    """The base TAEHV decoder for one latent space.

    Called as a module, it decodes `(B, C, h, w)` latents as single-frame videos and returns one RGB
    frame each; `decode_window` returns the last frame of a window of latent frames. `decode_video`
    decodes `(N, T, C, h, w)` latents as upstream does.
    """

    def __init__(
        self,
        latent_channels: int,
        patch_size: int = 1,
        time_upscale: tuple[bool, bool, bool] = (False, True, True),
        space_upscale: tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        n_f = (256, 128, 64, 64)
        t = [2 if up else 1 for up in time_upscale]
        s = [2 if up else 1 for up in space_upscale]
        self.decoder = nn.Sequential(
            Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=s[0]), TGrow(n_f[0], t[0]), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=s[1]), TGrow(n_f[1], t[1]), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=s[2]), TGrow(n_f[2], t[2]), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], 3 * patch_size**2),
        )  # fmt: skip
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.spatial_compression = s[0] * s[1] * s[2] * patch_size
        self.t_upscale = t[0] * t[1] * t[2]
        self.frames_to_trim = self.t_upscale - 1
        # Every latent frame decodes `t_upscale` frames through the full-resolution stages, which is what an
        # 8x decoder pays for: taew2_1 needs 2.1 GB and 63 ms at 1024 but 536 MB and 22 ms at 512 (RTX 4090,
        # fp16). The 16x and 32x decoders run those stages at a half or a quarter of the edge and stay at
        # 536 MB (taew2_2) and 266 MB (taeltx2_3) at 1024, under TAESD's ~640 MB.
        self.max_preview_edge = 512 if patch_size == 1 else 1024

    def _decode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, T, C, h, w)` latents to `(N, T * t_upscale, 3, H, W)` frames in [0, 1], untrimmed.

        Parallel over time: each memory block sees the previous frame's input, frame 0 sees zeros.
        """
        n = x.shape[0]
        x = x.flatten(0, 1)
        for block in self.decoder:
            if isinstance(block, MemBlock):
                nt, c, h, w = x.shape
                t = nt // n
                past = F.pad(x.view(n, t, c, h, w), (0, 0, 0, 0, 0, 0, 1, 0))[:, :t].reshape(x.shape)
                x = block(x, past)
            else:
                x = block(x)
        frames = F.pixel_shuffle(x, self.patch_size).clamp_(0, 1)
        return frames.view(n, -1, *frames.shape[1:])

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """`(N, T, C, h, w)` latents to `(N, T * t_upscale - frames_to_trim, 3, H, W)` frames.

        The first `frames_to_trim` outputs are causal warm-up that the reference VAEs do not produce.
        """
        return self._decode_frames(latents)[:, self.frames_to_trim :]

    def decode_window(self, window: torch.Tensor) -> torch.Tensor:
        """The last output frame of a `(B, C, T, h, w)` latent window, decoded after the frames before it."""
        return self.decode_video(window.transpose(1, 2))[:, -1]

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # One latent frame decodes to `t_upscale` frames, of which exactly one survives the trim.
        return self.decode_window(latents.unsqueeze(2))
