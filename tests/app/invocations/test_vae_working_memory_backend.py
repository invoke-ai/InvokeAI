"""Which convolution-backend coefficients the diffusers-AutoencoderKL estimators use, and what the SD estimator adds up.

SD1/SDXL, SD3 and CogView4 run the same AutoencoderKL convolution stack as the FLUX.1 autoencoder, so they take the
same measured per-backend coefficients: MIOpen needs ~1.64x what cuDNN does. The change is large enough to matter -- an
SD1.5 1024px fp32 decode reserves 9.49 GB on cuDNN and 15.36 GB on MIOpen -- and the branch is invisible on a CUDA
machine, so it is pinned here rather than left to whichever card the suite runs on.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.util.vae_working_memory import (
    VAE_COGVIEW4_SPEC,
    VAE_SD3_SPEC,
    VAE_SD_SPEC,
    estimate_vae_working_memory_cogview4,
    estimate_vae_working_memory_sd3,
    estimate_vae_working_memory_sd15_sdxl,
)
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.platform import Platform


def _vae(dtype: torch.dtype = torch.float16) -> MagicMock:
    vae = MagicMock(spec=AutoencoderKL)
    vae.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=dtype)])
    return vae


def _estimate(estimator, device: torch.device, hip: str | None) -> int:
    latents = torch.zeros(1, 4, 64, 64)  # 512px output
    with (
        patch("torch.version.hip", hip),
        # The mid-block score matrix is the other term in these estimates and has its own tests; hold it at zero so
        # the assertions below are about the convolution coefficient alone.
        patch("invokeai.backend.util.vae_working_memory._vae_mid_block_score_matrix_bytes", return_value=0),
    ):
        if estimator is estimate_vae_working_memory_sd15_sdxl:
            estimate = estimator(
                operation="decode", image_tensor=latents, vae=_vae(), tile_size=None, fp32=False, device=device
            )
        else:
            estimate = estimator(operation="decode", image_tensor=latents, vae=_vae(), device=device)
    return estimate.reservation_bytes


def _decode_bytes(spec, conv_backend: str) -> float:
    """A 512px fp16 decode at the shipped coefficient of that backend."""
    return 512 * 512 * 2 * shipped_coefficients(spec, Platform(conv_backend)).values["k_decode"]


ESTIMATORS = [
    pytest.param(estimate_vae_working_memory_sd15_sdxl, VAE_SD_SPEC, id="sd15-sdxl"),
    pytest.param(estimate_vae_working_memory_sd3, VAE_SD3_SPEC, id="sd3"),
    pytest.param(estimate_vae_working_memory_cogview4, VAE_COGVIEW4_SPEC, id="cogview4"),
]


@pytest.mark.parametrize("estimator, spec", ESTIMATORS)
def test_a_rocm_vae_reserves_the_miopen_coefficient(estimator, spec):
    assert _estimate(estimator, torch.device("cuda", 0), hip="7.2.0") == _decode_bytes(spec, "miopen")


@pytest.mark.parametrize("estimator, spec", ESTIMATORS)
def test_a_cuda_vae_reserves_the_cudnn_coefficient(estimator, spec):
    assert _estimate(estimator, torch.device("cuda", 0), hip=None) == _decode_bytes(spec, "cudnn")


@pytest.mark.parametrize("estimator, spec", ESTIMATORS)
def test_a_cpu_only_vae_reserves_the_cudnn_coefficient_even_on_a_rocm_build(estimator, spec):
    """A `cpu_only` VAE does not run on MIOpen, so naming its device has to change the answer."""
    assert _estimate(estimator, torch.device("cpu"), hip="7.2.0") == _decode_bytes(spec, "cudnn")


def _sd_fp32_decode(tile_size: int | None) -> int:
    latents = torch.zeros(1, 4, 96, 128)  # a 768x1024 output
    with patch("invokeai.backend.util.vae_working_memory._vae_mid_block_score_matrix_bytes", return_value=0):
        estimate = estimate_vae_working_memory_sd15_sdxl(
            operation="decode",
            image_tensor=latents,
            vae=_vae(),
            tile_size=tile_size,
            fp32=True,
            device=torch.device("cpu"),
        )
    return estimate.reservation_bytes


FP32_WEIGHTS = 250 * 2**20
"""What an fp32 decode adds for the upcast weights."""


def test_a_tiled_fp32_sd_decode_is_one_tile_with_its_margin_plus_the_fp32_weights():
    k = shipped_coefficients(VAE_SD_SPEC, Platform("cudnn")).values["k_decode"]
    assert _sd_fp32_decode(512) == int(512 * 512 * 4 * k * 1.25 + FP32_WEIGHTS)


def test_an_untiled_fp32_sd_decode_is_the_whole_image_plus_the_fp32_weights():
    k = shipped_coefficients(VAE_SD_SPEC, Platform("cudnn")).values["k_decode"]
    assert _sd_fp32_decode(None) == int(768 * 1024 * 4 * k + FP32_WEIGHTS)
