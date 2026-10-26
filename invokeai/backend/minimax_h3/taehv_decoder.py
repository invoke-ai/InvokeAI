"""Tiny AutoEncoder decoder for MiniMax H3 latents ("taeh3"), used for denoising previews.

The vendored TAEHV decoder (`invokeai/backend/tiny_autoencoder/taehv.py`, madebyollin, MIT) at the
same commit as the published H3 weights:
https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/taehv.py

The decoder consumes latents in the NORMALIZED space — ``(z - latents_mean) / latents_std``,
i.e. exactly the space the denoise loop operates in — and emits RGB frames in ``[0, 1]``.
Verified against the real H3 video VAE: a real-VAE encode round-tripped through this decoder
reconstructs at ~21 dB PSNR, while feeding *unnormalized* latents produces garbage, so no
mean/std conversion belongs in the preview path.

Scale factors mirror the full VAE: 16x spatial (8x upsampling + 2x pixel shuffle), 4x
temporal (TGrow strides 1/2/2), 24 latent channels. The ``decoder.*`` ``nn.Sequential``
indices are a state-dict contract with the released ``taeh3.safetensors`` — do not reorder.

H3 keeps its own preview path (``load_remote_model``, ``decode_preview_frame`` over a latent window)
rather than the ``tiny_autoencoder`` model type: its frames are taken untrimmed from a window, not as
one trimmed frame per latent.
"""

from pathlib import Path

import torch

from invokeai.backend.tiny_autoencoder.taehv import TaehvDecoder

# Weights are pinned to the commit whose taehv.py this module mirrors. The GitHub "raw"
# redirect resolves to the LFS object; the download cache keys on this URL.
TAEH3_PREVIEW_MODEL_URL = (
    "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors"
)

TAEH3_LATENT_CHANNELS = 24
TAEH3_TEMPORAL_UPSCALE = 4
_PIXEL_SHUFFLE_PATCH = 2


class TAEH3Decoder(TaehvDecoder):
    """The taeh3 decoder: ``[N, 24, T, h, w]`` normalized latents -> ``[N, 4T, 3, 16h, 16w]`` RGB."""

    def __init__(self) -> None:
        super().__init__(TAEH3_LATENT_CHANNELS, patch_size=_PIXEL_SHUFFLE_PATCH)

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
        return self._decode_frames(latents.permute(0, 2, 1, 3, 4).to(weight_dtype))  # NTCHW

    def decode_preview_frame(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a small latent window and return the best single frame, ``[3, H, W]`` in [0, 1].

        The last frame of the window has the most temporal-memory context (the earlier frames
        are causal warmup), so it is the one worth showing.
        """
        return self.decode(latents)[0, -1]
