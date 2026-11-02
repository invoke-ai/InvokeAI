"""MOSS-Image sigma schedule, initial noise and latent-layout conversions."""

import pytest
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.utils.torch_utils import randn_tensor

from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageScheduler
from invokeai.backend.moss_image.sampling import get_moss_noise, get_moss_sigmas, to_model_space, to_raw_latent


@pytest.fixture(scope="module")
def vae() -> AutoencoderKLFlux2:
    """A one-block AutoencoderKLFlux2: only its BatchNorm (128 channels, eps 1e-4) is used.

    The running variances span 1e-3..3, so the 1e-5 default of the ERNIE helpers would visibly shift the small ones,
    and the means are far from zero, so a wrong channel order or direction shows.
    """
    vae = AutoencoderKLFlux2(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        mid_block_add_attention=False,
    ).eval()
    generator = torch.Generator().manual_seed(0)
    vae.bn.running_mean.copy_(torch.randn(128, generator=generator) * 2)
    vae.bn.running_var.copy_(torch.logspace(-3, 0.5, 128)[torch.randperm(128, generator=generator)])
    return vae


@pytest.mark.parametrize("num_steps", [1, 50, 250])
def test_full_window_is_the_reference_schedule(num_steps: int) -> None:
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(num_steps)

    sigmas = get_moss_sigmas(num_steps)

    assert sigmas == scheduler.sigmas.tolist()
    if num_steps == 250:
        # Snapping to the training grid merges neighbouring sigmas here; the full window must keep the reference's
        # shorter schedule rather than re-spacing it to the requested count.
        assert len(sigmas) - 1 < num_steps


def test_partial_window_is_respaced_to_the_requested_steps_between_exact_endpoints() -> None:
    # Clipping the 50-step schedule to this window leaves 15 steps.
    sigmas = get_moss_sigmas(50, denoising_start=0.3, denoising_end=0.8765)

    assert len(sigmas) - 1 == 50
    assert sigmas[0] == 1.0 - 0.3
    assert sigmas[-1] == 1.0 - 0.8765
    # Even steps in the unshifted variable v of the scheduler's shift sigma = 5v / (1 + 4v).
    unshifted = [sigma / (5.0 - 4.0 * sigma) for sigma in sigmas]
    gaps = [a - b for a, b in zip(unshifted[:-1], unshifted[1:], strict=True)]
    assert gaps == pytest.approx([gaps[0]] * 50, rel=1e-9)
    assert gaps[0] > 0


def test_window_that_keeps_the_requested_steps_is_left_on_the_reference_grid() -> None:
    full = get_moss_sigmas(8)

    # Ending at sigma 0.01 keeps all 8 steps above it and appends the end point.
    sigmas = get_moss_sigmas(8, denoising_start=0.0, denoising_end=0.99)

    assert sigmas == full[:-1] + [1.0 - 0.99]


@pytest.mark.parametrize(("denoising_start", "denoising_end"), [(0.5, 0.5), (0.6, 0.4), (-0.1, 1.0), (0.0, 1.1)])
def test_invalid_window_is_rejected(denoising_start: float, denoising_end: float) -> None:
    with pytest.raises(ValueError, match="Invalid denoising window"):
        get_moss_sigmas(50, denoising_start, denoising_end)


def test_window_too_narrow_for_a_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="too narrow"):
        get_moss_sigmas(50, denoising_start=0.5, denoising_end=0.5000001)


def test_noise_is_the_reference_draw_for_the_seed() -> None:
    expected = randn_tensor(
        (1, 128, 1024 // 16, 768 // 16),
        generator=torch.Generator("cpu").manual_seed(42),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(get_moss_noise(42, height=1024, width=768), expected)


def test_model_space_is_flux2_packing_normalized_with_the_vae_batch_norm(vae: AutoencoderKLFlux2) -> None:
    raw = torch.randn(2, 32, 6, 10, generator=torch.Generator().manual_seed(1)) * 3
    bn = vae.bn

    # Flux2DenoiseInvocation: pack_flux2 to (B, h * w, 128), then (x - mean) / sqrt(var + eps) with the BN's own eps.
    sequence = (pack_flux2(raw) - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
    expected = sequence.reshape(2, 3, 5, 128).permute(0, 3, 1, 2)

    torch.testing.assert_close(to_model_space(raw, bn), expected)


def test_raw_latent_is_the_reference_decode_preparation(vae: AutoencoderKLFlux2) -> None:
    packed = torch.randn(2, 128, 3, 5, generator=torch.Generator().manual_seed(2))

    # MOSSImagePipeline.__call__, the lines between the denoise loop and vae.decode (float32 VAE).
    z = packed.to(dtype=vae.dtype)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    expected = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)

    assert torch.equal(to_raw_latent(packed, bn), expected)


def test_round_trip_through_model_space_is_lossless_in_float32(vae: AutoencoderKLFlux2) -> None:
    # A bf16 input is converted in float32: a bf16 round trip would be off by ~1e-2 here.
    raw = (torch.randn(1, 32, 8, 12, generator=torch.Generator().manual_seed(3)) * 3).bfloat16()

    round_trip = to_raw_latent(to_model_space(raw, vae.bn), vae.bn)

    torch.testing.assert_close(round_trip, raw.float())
