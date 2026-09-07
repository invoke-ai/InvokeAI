"""Regression tests for the FLUX.2 img2img/inpainting latent normalization (see #8964).

The FLUX.2 transformer operates on BN-normalized latents, while the VAE encode node emits raw
latents. The img2img preblend used to be computed in raw space and normalized afterwards, which
divided the noise term by bn_std (~1.77 for the FLUX.2 VAE) as well. The start latents then carried
only ~57% of the noise implied by their timestep, the model over-denoised, and fine detail collapsed
into posterized patches -- progressively worse the higher the denoise strength.
"""

import pytest
import torch

from invokeai.app.invocations.flux2_denoise import Flux2DenoiseInvocation

# Measured on the BFL FLUX.2 VAE (bn.running_mean / bn.running_var): mean ~0, var ~3.13.
BN_STD_VALUE = 1.7676
PACKED_CHANNELS = 128


def _bn_stats() -> tuple[torch.Tensor, torch.Tensor]:
    bn_mean = torch.full((PACKED_CHANNELS,), 0.05)
    bn_std = torch.full((PACKED_CHANNELS,), BN_STD_VALUE)
    return bn_mean, bn_std


def _packed(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 64, PACKED_CHANNELS, generator=generator)


@pytest.mark.parametrize("t_0", [0.2, 0.5, 0.85, 1.0])
def test_noise_keeps_unit_scale_in_normalized_start_latents(t_0: float) -> None:
    """The noise term must survive the preblend at full N(0, 1) scale, for every denoise strength."""
    bn_mean, bn_std = _bn_stats()
    init_latents = _packed(1)
    noise = _packed(2)
    invocation = Flux2DenoiseInvocation.model_construct(add_noise=True)

    x, normalized_init = invocation._prepare_normalized_start_latents(
        init_latents_packed=init_latents,
        noise_packed=noise,
        t_0=t_0,
        bn_mean=bn_mean,
        bn_std=bn_std,
    )

    # x == t_0 * noise + (1 - t_0) * normalize(init), so peeling off the init term must recover the
    # noise unscaled -- not noise / bn_std, which is what normalizing the raw mixture produced.
    recovered_noise = (x - (1.0 - t_0) * normalized_init) / t_0
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_start_latents_differ_from_normalizing_the_raw_mixture() -> None:
    """Guard against a regression back to normalizing the blended tensor as a whole."""
    bn_mean, bn_std = _bn_stats()
    init_latents = _packed(3)
    noise = _packed(4)
    t_0 = 0.6
    invocation = Flux2DenoiseInvocation.model_construct(add_noise=True)

    x, _ = invocation._prepare_normalized_start_latents(
        init_latents_packed=init_latents,
        noise_packed=noise,
        t_0=t_0,
        bn_mean=bn_mean,
        bn_std=bn_std,
    )

    buggy = invocation._bn_normalize(t_0 * noise + (1.0 - t_0) * init_latents, bn_mean, bn_std)
    # The buggy variant attenuates the noise term by 1 / bn_std.
    assert not torch.allclose(x, buggy, atol=1e-3)
    assert torch.allclose(x - buggy, t_0 * (noise - (noise - bn_mean) / bn_std), atol=1e-5)


def test_init_latents_are_normalized() -> None:
    bn_mean, bn_std = _bn_stats()
    init_latents = _packed(5)
    invocation = Flux2DenoiseInvocation.model_construct(add_noise=True)

    _, normalized_init = invocation._prepare_normalized_start_latents(
        init_latents_packed=init_latents,
        noise_packed=_packed(6),
        t_0=0.4,
        bn_mean=bn_mean,
        bn_std=bn_std,
    )

    assert torch.allclose(normalized_init, (init_latents - bn_mean) / bn_std, atol=1e-6)


def test_without_add_noise_start_latents_are_the_normalized_init_latents() -> None:
    bn_mean, bn_std = _bn_stats()
    init_latents = _packed(7)
    invocation = Flux2DenoiseInvocation.model_construct(add_noise=False)

    x, normalized_init = invocation._prepare_normalized_start_latents(
        init_latents_packed=init_latents,
        noise_packed=_packed(8),
        t_0=0.4,
        bn_mean=bn_mean,
        bn_std=bn_std,
    )

    assert torch.allclose(x, normalized_init, atol=1e-6)
    assert torch.allclose(x, (init_latents - bn_mean) / bn_std, atol=1e-6)


def test_without_bn_stats_the_raw_preblend_is_preserved() -> None:
    """VAE formats that expose no BN stats keep the previous raw-space behaviour."""
    init_latents = _packed(9)
    noise = _packed(10)
    t_0 = 0.3
    invocation = Flux2DenoiseInvocation.model_construct(add_noise=True)

    x, normalized_init = invocation._prepare_normalized_start_latents(
        init_latents_packed=init_latents,
        noise_packed=noise,
        t_0=t_0,
        bn_mean=None,
        bn_std=None,
    )

    assert torch.allclose(normalized_init, init_latents, atol=1e-6)
    assert torch.allclose(x, t_0 * noise + (1.0 - t_0) * init_latents, atol=1e-6)
