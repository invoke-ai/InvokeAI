"""Tiny AutoEncoder for video latents (TAEHV): the decoder half, run on one latent frame for image previews.

Adapted from madebyollin's TAEHV, pinned to:
https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/taehv.py

Kept: the default (non-`_super`) decoder at patch size 1, which taew2_1 uses. `TaehvImageDecoder` decodes one
latent frame into the last frame upstream's `TAEHV.decode_video` returns for it: exactly that frame once clamped to
[0, 1]. Like the TAESD decoders, its output is unclamped; `decode_preview` clamps. Upstream's `TGrow`s turn the frame
into several; each `MemBlock` gives frame t the block input of frame t-1 as memory and frame 0 zeros
(`apply_model_with_memblocks_parallel`). From the last `MemBlock` on every layer's output for the last frame depends
only on that frame and its memory, so the decoder runs that block on the last frame alone, and the `TGrow` that
follows is built at stride 1 with its last-timestep output channels (`checkpoint.taesd_decoder_state_dict`, after
upstream's `patch_tgrow_layers`). The full-resolution stages therefore run on one frame. Left out: the encoder, `TPool`, `SuperMemBlock`, pixel-shuffle patching and the
streaming drivers.

The `nn.Sequential` indices are the state-dict contract with the released checkpoints' `decoder.N.*` keys: do not
reorder, insert or remove layers. Upstream applies no latent scale or shift: the decoder reads the latents the
diffusion model denoises.

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

from invokeai.backend.tiny_autoencoder.taesd import Clamp, TaesdDecoder, conv


class MemBlock(nn.Module):
    """Upstream's `MemBlock`: a residual block whose convolution also reads the previous frame's input."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))  # fmt: skip
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Upstream's `TGrow`: a 1x1 convolution to `stride` groups of channels, one per output frame."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class TaehvImageDecoder(TaesdDecoder):
    """`(N, C, h, w)` latent frames in, `(N, 3, 8h, 8w)` images out: each upstream's last decoded frame, unclamped."""

    def __init__(self, *layers: nn.Module, latent_channels: int, spatial_compression: int):
        super().__init__(*layers, latent_channels=latent_channels, spatial_compression=spatial_compression)
        self._last_memblock = max(i for i, block in enumerate(self) if isinstance(block, MemBlock))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        for i, block in enumerate(self):
            if not isinstance(block, MemBlock):
                x = block(x)
                continue
            frames = x.view(n, -1, *x.shape[1:])
            past = F.pad(frames, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1]
            if i == self._last_memblock:
                x = block(frames[:, -1], past[:, -1])
            else:
                x = block(x, past.reshape(x.shape))
        return x


def ImageDecoder(latent_channels: int = 16) -> TaehvImageDecoder:
    """The patch-1 TAEHV decoder at upstream's default `decoder_time_upscale=(False, True, True)`.

    The last `TGrow` comes after the last `MemBlock`, where only the last frame is kept, so it is built at stride 1.
    """
    n_f = [256, 128, 64, 64]
    return TaehvImageDecoder(
        Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
        MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
        MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2), TGrow(n_f[1], 2), conv(n_f[1], n_f[2], bias=False),
        MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2), TGrow(n_f[2], 1), conv(n_f[2], n_f[3], bias=False),
        nn.ReLU(inplace=True), conv(n_f[3], 3),
        latent_channels=latent_channels,
        spatial_compression=8,
    )  # fmt: skip
