import pytest
import torch

from invokeai.backend.krea2.style_reference_rf import build_linear_reference_latents


def test_linear_schedule_returns_one_latent_per_sigma() -> None:
    reference = torch.randn(1, 12, 64)
    sigmas = [1.0, 0.75, 0.5, 0.25]
    assert len(build_linear_reference_latents(reference, sigmas)) == len(sigmas)


def test_linear_schedule_matches_the_closed_form() -> None:
    reference = torch.randn(1, 12, 64)
    sigmas = [1.0, 0.5, 0.0]
    latents = build_linear_reference_latents(reference, sigmas)

    # At sigma 0 the reference is untouched; at sigma 1 it is pure noise.
    assert torch.allclose(latents[2], reference)
    noise = latents[0]
    assert torch.allclose(latents[1], 0.5 * reference + 0.5 * noise, atol=1e-6)


def test_linear_schedule_reuses_a_single_noise_draw() -> None:
    """Upstream draws eps once and reuses it for every sigma.

    Re-sampling per step would make the reference's features jitter from step to step, which is exactly
    the thing the styled attention must not do.
    """
    reference = torch.randn(1, 12, 64)
    latents = build_linear_reference_latents(reference, [0.8, 0.4])

    # Recover eps from each point: z = (1 - s) * ref + s * eps  =>  eps = (z - (1 - s) * ref) / s
    eps_from_first = (latents[0] - 0.2 * reference) / 0.8
    eps_from_second = (latents[1] - 0.6 * reference) / 0.4
    assert torch.allclose(eps_from_first, eps_from_second, atol=1e-4)


def test_linear_schedule_is_deterministic_for_a_fixed_seed() -> None:
    reference = torch.randn(1, 12, 64)
    first = build_linear_reference_latents(reference, [0.7])
    second = build_linear_reference_latents(reference, [0.7])
    assert torch.equal(first[0], second[0])


def test_linear_schedule_responds_to_the_seed() -> None:
    reference = torch.randn(1, 12, 64)
    first = build_linear_reference_latents(reference, [0.7], seed=1)
    second = build_linear_reference_latents(reference, [0.7], seed=2)
    assert not torch.allclose(first[0], second[0])


def test_linear_schedule_clamps_sigmas_into_range() -> None:
    reference = torch.randn(1, 12, 64)
    latents = build_linear_reference_latents(reference, [-0.5, 1.5])
    assert torch.allclose(latents[0], reference)


def test_linear_schedule_preserves_dtype_and_shape() -> None:
    reference = torch.randn(1, 12, 64, dtype=torch.float16)
    latent = build_linear_reference_latents(reference, [0.5])[0]
    assert latent.shape == reference.shape
    assert latent.dtype == torch.float16


def test_linear_schedule_rejects_an_empty_schedule() -> None:
    with pytest.raises(ValueError, match="sigma schedule is empty"):
        build_linear_reference_latents(torch.randn(1, 12, 64), [])
