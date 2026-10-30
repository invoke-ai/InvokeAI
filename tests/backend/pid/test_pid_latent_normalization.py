"""The FLUX.2 PiD node must hand PiD the latent the upstream FLUX.2 tokenizer encodes to.

Upstream (nv-tlabs/PiD ``tokenizers/flux2_vae.py``) trains PiD on ``BatchNorm(patchify(raw))`` with the
VAE's running statistics. ``flux2_denoise`` stores the BN-*denormalized*, unpacked latent, so the node
has to undo exactly that.
"""

import pytest
import torch
from einops import rearrange
from torch import nn

from invokeai.backend.flux2.sampling_utils import get_flux2_bn_stats, unpack_flux2
from invokeai.backend.pid.decode import pack_flux2_latent_for_pid


class _Flux2LikeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(128, eps=1e-4, affine=False, track_running_stats=True)
        generator = torch.Generator().manual_seed(0)
        self.bn.running_mean.copy_(torch.randn(128, generator=generator))
        self.bn.running_var.copy_(torch.rand(128, generator=generator) * 4 + 0.1)


def test_pack_matches_upstream_flux2_tokenizer_encode() -> None:
    vae = _Flux2LikeVae()
    raw = torch.randn(2, 32, 12, 8, generator=torch.Generator().manual_seed(1))

    # The upstream tokenizer's encode, verbatim: patchify with channel order (c pi pj), then BatchNorm in eval mode.
    vae.bn.eval()
    expected = vae.bn(rearrange(raw, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2))

    bn_stats = get_flux2_bn_stats(vae)
    assert bn_stats is not None
    torch.testing.assert_close(pack_flux2_latent_for_pid(raw, *bn_stats), expected)


def test_pack_recovers_the_latent_flux2_denoise_produced() -> None:
    """Round trip through flux2_denoise's output path: denoised (normalized, packed) -> stored -> PiD input."""
    bn_mean, bn_std = get_flux2_bn_stats(_Flux2LikeVae())
    height, width = 192, 128
    packed_h, packed_w = height // 16, width // 16
    normalized = torch.randn(1, packed_h * packed_w, 128, generator=torch.Generator().manual_seed(2))

    # What flux2_denoise stores: BN-denormalize in (B, seq, 128), then unpack to (B, 32, H/8, W/8).
    stored = unpack_flux2(normalized * bn_std + bn_mean, height, width)

    expected = rearrange(normalized, "b (h w) c -> b c h w", h=packed_h, w=packed_w)
    torch.testing.assert_close(pack_flux2_latent_for_pid(stored, bn_mean, bn_std), expected)


def test_pack_rejects_a_non_flux2_latent() -> None:
    bn_mean, bn_std = torch.zeros(128), torch.ones(128)
    with pytest.raises(ValueError, match="32-channel"):
        pack_flux2_latent_for_pid(torch.zeros(1, 16, 8, 8), bn_mean, bn_std)


def test_pack_rejects_batchnorm_stats_of_another_vae() -> None:
    with pytest.raises(ValueError, match="128 channels"):
        pack_flux2_latent_for_pid(torch.zeros(1, 32, 8, 8), torch.zeros(64), torch.ones(64))


def test_bn_stats_absent_without_a_batchnorm_layer() -> None:
    assert get_flux2_bn_stats(nn.Conv2d(4, 4, 1)) is None
