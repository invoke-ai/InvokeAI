"""The FLUX.2 Fun control context is VideoX-Fun's, value for value.

`_videox_context` restates `pipeline_flux2_control.py` (the block that builds `control_context`): each part is encoded,
patchified 2x2 to 128 channels, batch-normalized only if it was provided, and packed; the keep mask is `1 - mask`
nearest-sampled to latent resolution and patchified to 4 features; the three are concatenated per token. The function
under test takes the other route -- normalize, put back to 32 channels, let the shared builder stack 65 channels, pack
once -- and must land on the same tensor.
"""

import pytest
import torch
import torch.nn.functional as F

from invokeai.backend.fun_controlnet.flux2 import FLUX2_FUN_CONTROL_IN_DIM, build_flux2_control_context

SIDE = 64
LATENT_SIDE = SIDE // 8
LATENT_CHANNELS = 32


def _encode(pixels: torch.Tensor) -> torch.Tensor:
    """Stands in for the VAE: an 8x average pool spread over 32 channels, each scaled differently so a channel that
    lands in the wrong feature shows up."""
    pooled = F.avg_pool2d(pixels, 8)
    channels = torch.cat([pooled] * 11, dim=1)[:, :LATENT_CHANNELS]
    return channels * torch.linspace(0.5, 2.0, LATENT_CHANNELS).view(1, -1, 1, 1)


def _bn_stats() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(4 * LATENT_CHANNELS, generator=generator) * 0.3
    std = torch.rand(4 * LATENT_CHANNELS, generator=generator) + 0.5
    return mean, std


def _image(seed: int) -> torch.Tensor:
    return torch.rand(1, 3, SIDE, SIDE, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def _soft_mask() -> torch.Tensor:
    """Bands below, at and above the 0.5 threshold, so thresholding and the nearest sampling both matter."""
    mask = torch.zeros(1, 1, SIDE, SIDE)
    mask[..., :16, :] = 0.5
    mask[..., 16:24, :] = 0.3
    mask[..., 24:40, : SIDE // 2] = 0.9
    return mask


# VideoX-Fun `Flux2ControlPipeline._patchify_latents` / `_pack_latents`.
def _patchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def _pack(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def _videox_context(bn_mean, bn_std, control_image=None, inpaint_image=None, inpaint_mask=None) -> torch.Tensor:
    mean, std = bn_mean.view(1, -1, 1, 1), bn_std.view(1, -1, 1, 1)
    if inpaint_mask is not None:
        mask_condition = torch.where(inpaint_mask >= 0.5, 1.0, 0.0).tile(1, 3, 1, 1)
    else:
        mask_condition = torch.ones(1, 3, SIDE, SIDE)
    if inpaint_image is not None:
        inpaint_latent = _encode(inpaint_image * (mask_condition < 0.5))
    else:
        inpaint_latent = torch.zeros(1, LATENT_CHANNELS, LATENT_SIDE, LATENT_SIDE)
    control_latents = _encode(control_image) if control_image is not None else torch.zeros_like(inpaint_latent)

    keep = F.interpolate(1 - mask_condition[:, :1], size=control_latents.shape[-2:], mode="nearest")
    keep = _pack(_patchify(keep))
    inpaint_latent = _patchify(inpaint_latent)
    if inpaint_image is not None:
        inpaint_latent = (inpaint_latent - mean) / std
    control_latents = _patchify(control_latents)
    if control_image is not None:
        control_latents = (control_latents - mean) / std
    return torch.cat([_pack(control_latents), keep, _pack(inpaint_latent)], dim=2)


@pytest.mark.parametrize(
    "parts",
    [
        {"control_image": _image(1)},
        {"inpaint_image": _image(2), "inpaint_mask": _soft_mask()},
        {"control_image": _image(3), "inpaint_image": _image(4), "inpaint_mask": _soft_mask()},
    ],
    ids=["control", "inpaint", "both"],
)
def test_the_context_is_videox_funs_value_for_value(parts) -> None:
    bn_mean, bn_std = _bn_stats()

    context = build_flux2_control_context(encode=_encode, bn_mean=bn_mean, bn_std=bn_std, **parts)

    assert context.shape == (1, (LATENT_SIDE // 2) ** 2, FLUX2_FUN_CONTROL_IN_DIM)
    assert torch.equal(context, _videox_context(bn_mean, bn_std, **parts))


def test_an_absent_part_stays_zero_rather_than_normalized_zero() -> None:
    """The normalization would turn an absent part into `-mean / std`. Zeros are what training feeds there most of
    the time: always for a dropped control image, and for 90 % of control-only samples' masked image (the rest, and
    VideoX-Fun's inference scripts, use the normalized encoding of the grey masked image)."""
    bn_mean, bn_std = _bn_stats()
    assert bn_mean.abs().min() > 0, "the stats must move zero for this test to mean anything"

    context = build_flux2_control_context(encode=_encode, bn_mean=bn_mean, bn_std=bn_std, control_image=_image(5))

    assert torch.equal(context[..., 128:], torch.zeros(1, (LATENT_SIDE // 2) ** 2, 132))
