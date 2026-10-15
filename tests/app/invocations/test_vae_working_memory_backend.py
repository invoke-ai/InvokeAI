"""Which convolution-backend constant the diffusers-AutoencoderKL estimators use.

SD1/SDXL, SD3 and CogView4 run the same AutoencoderKL convolution stack as the FLUX.1 autoencoder, so they take the
same measured per-backend constants: MIOpen needs ~1.64x what cuDNN does. The change is large enough to matter -- an
SD1.5 1024px fp32 decode reserves 9.49 GB on cuDNN and 15.36 GB on MIOpen -- and the branch is invisible on a CUDA
machine, so it is pinned here rather than left to whichever card the suite runs on.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.util.vae_working_memory import (
    estimate_vae_working_memory_cogview4,
    estimate_vae_working_memory_sd3,
    estimate_vae_working_memory_sd15_sdxl,
)


def _vae(dtype: torch.dtype = torch.float16) -> MagicMock:
    vae = MagicMock(spec=AutoencoderKL)
    vae.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=dtype)])
    return vae


def _estimate(estimator, device: torch.device, hip: str | None):
    latents = torch.zeros(1, 4, 64, 64)  # 512px output
    with (
        patch("torch.version.hip", hip),
        # The mid-block score matrix is the other term in these estimates and has its own tests; hold it at zero so
        # the assertions below are about the convolution constant alone.
        patch("invokeai.backend.util.vae_working_memory._vae_mid_block_score_matrix_bytes", return_value=0),
    ):
        if estimator is estimate_vae_working_memory_sd15_sdxl:
            return estimator(
                operation="decode", image_tensor=latents, vae=_vae(), tile_size=None, fp32=False, device=device
            )
        return estimator(operation="decode", image_tensor=latents, vae=_vae(), device=device)


ESTIMATORS = [
    pytest.param(estimate_vae_working_memory_sd15_sdxl, id="sd15-sdxl"),
    pytest.param(estimate_vae_working_memory_sd3, id="sd3"),
    pytest.param(estimate_vae_working_memory_cogview4, id="cogview4"),
]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_a_rocm_vae_reserves_the_miopen_constant(estimator):
    assert _estimate(estimator, torch.device("cuda", 0), hip="7.2.0") == 512 * 512 * 2 * 3600


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_a_cuda_vae_reserves_the_cudnn_constant(estimator):
    assert _estimate(estimator, torch.device("cuda", 0), hip=None) == 512 * 512 * 2 * 2200


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_a_cpu_only_vae_reserves_the_cudnn_constant_even_on_a_rocm_build(estimator):
    """A `cpu_only` VAE does not run on MIOpen, so naming its device has to change the answer."""
    assert _estimate(estimator, torch.device("cpu"), hip="7.2.0") == 512 * 512 * 2 * 2200
