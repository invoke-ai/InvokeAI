"""Tiny AutoEncoder decoder for MiniMax H3 latents ("taeh3"), used for denoising previews.

Adapted from madebyollin's TAEHV (MIT license), decoder half only, pinned to the same commit
as the published H3 weights:
https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/taehv.py

The decoder consumes latents in the NORMALIZED space — ``(z - latents_mean) / latents_std``,
i.e. exactly the space the denoise loop operates in — and emits RGB frames in ``[0, 1]``.
Verified against the real H3 video VAE: a real-VAE encode round-tripped through this decoder
reconstructs at ~21 dB PSNR, while feeding *unnormalized* latents produces garbage, so no
mean/std conversion belongs in the preview path.

Scale factors mirror the full VAE: 16x spatial (8x upsampling + 2x pixel shuffle), 4x
temporal (TGrow strides 1/2/2), 24 latent channels. The ``decoder.*`` ``nn.Sequential``
indices are a state-dict contract with the released ``taeh3.safetensors`` — do not reorder.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Weights are pinned to the commit whose taehv.py this module mirrors. The GitHub "raw"
# redirect resolves to the LFS object; the download cache keys on this URL.
TAEH3_PREVIEW_MODEL_URL = (
    "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors"
)

TAEH3_LATENT_CHANNELS = 24
TAEH3_TEMPORAL_UPSCALE = 4
_PIXEL_SHUFFLE_PATCH = 2


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """Residual block whose conv also sees the previous frame's input (the temporal memory)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    """Temporal upsampling: a 1x1 conv to ``stride`` channel groups, reshaped into frames."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class TAEH3Decoder(nn.Module):
    """The taeh3 decoder: ``[N, 24, T, h, w]`` normalized latents -> ``[N, 4T, 3, 16h, 16w]`` RGB."""

    def __init__(self) -> None:
        super().__init__()
        n_f = (256, 128, 64, 64)
        self.decoder = nn.Sequential(
            _Clamp(),
            nn.Conv2d(TAEH3_LATENT_CHANNELS, n_f[0], 3, padding=1),
            nn.ReLU(inplace=True),
            _MemBlock(n_f[0], n_f[0]),
            _MemBlock(n_f[0], n_f[0]),
            _MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[0], 1),
            nn.Conv2d(n_f[0], n_f[1], 3, padding=1, bias=False),
            _MemBlock(n_f[1], n_f[1]),
            _MemBlock(n_f[1], n_f[1]),
            _MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[1], 2),
            nn.Conv2d(n_f[1], n_f[2], 3, padding=1, bias=False),
            _MemBlock(n_f[2], n_f[2]),
            _MemBlock(n_f[2], n_f[2]),
            _MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2),
            _TGrow(n_f[2], 2),
            nn.Conv2d(n_f[2], n_f[3], 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_f[3], 3 * _PIXEL_SHUFFLE_PATCH**2, 3, padding=1),
        )

    @classmethod
    def load_model(cls, path: Path) -> "TAEH3Decoder":
        """Loader for ``context.models.load_remote_model``: reads ``taeh3.safetensors``.

        The file also carries the (unused) encoder; only ``decoder.*`` keys are consumed,
        strictly. Weights stay in their stored float16.
        """
        from safetensors.torch import load_file

        state_dict = {k: v for k, v in load_file(path).items() if k.startswith("decoder.")}
        model = cls()
        model.to(torch.float16)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[N, C, T, h, w]`` normalized latents to ``[N, 4T, 3, 16h, 16w]`` RGB in [0, 1].

        Straight decoder pass with parallel temporal-memory handling — none of the full VAE's
        17-frame chunk alignment, so the leading ``4T - (4T - 3)``-ish warmup frames decode
        against empty memory: for previews, read the LAST frame, not the first.
        """
        if latents.ndim != 5 or latents.shape[1] != TAEH3_LATENT_CHANNELS:
            raise ValueError(f"Expected [N, {TAEH3_LATENT_CHANNELS}, T, h, w] latents, got {list(latents.shape)}.")
        weight_dtype = self.decoder[1].weight.dtype
        x = latents.permute(0, 2, 1, 3, 4).to(weight_dtype)  # NTCHW
        n = x.shape[0]
        x = x.flatten(0, 1)
        for block in self.decoder:
            if isinstance(block, _MemBlock):
                nt, c, h, w = x.shape
                t = nt // n
                # Each frame's memory is the previous frame's block input; frame 0 sees zeros.
                past = F.pad(x.view(n, t, c, h, w), (0, 0, 0, 0, 0, 0, 1, 0))[:, :t].reshape(x.shape)
                x = block(x, past)
            else:
                x = block(x)
        frames = F.pixel_shuffle(x, _PIXEL_SHUFFLE_PATCH).clamp_(0, 1)
        return frames.view(n, -1, *frames.shape[1:])

    def decode_preview_frame(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a small latent window and return the best single frame, ``[3, H, W]`` in [0, 1].

        The last frame of the window has the most temporal-memory context (the earlier frames
        are causal warmup), so it is the one worth showing.
        """
        return self.decode(latents)[0, -1]
