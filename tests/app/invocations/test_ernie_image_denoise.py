"""Tests for `ErnieImageDenoiseInvocation`'s initial-latent handling.

ERNIE denoises on the rectified-flow path, so the loop assumes its input already sits at the first
sigma of the schedule. Both directions of that contract are easy to get silently wrong:

- init latents passed through *unnoised* tell the model the sample is at `sigma_0` when it is at 0
- pure noise at a *reduced* `sigma_0` tells the model it is already partly denoised

Neither produces an error at runtime -- just a bad image -- so they are pinned here.
"""

import pytest
import torch

from invokeai.app.invocations.ernie_image_denoise import ErnieImageDenoiseInvocation


def _invocation(denoising_start: float = 0.0) -> ErnieImageDenoiseInvocation:
    return ErnieImageDenoiseInvocation.model_construct(denoising_start=denoising_start)


def test_txt2img_returns_the_noise_unchanged() -> None:
    noise = torch.randn(1, 128, 4, 4)

    result = _invocation()._prepare_initial_latents(noise, None, first_sigma=1.0)

    assert result is noise


def test_denoising_start_without_init_latents_is_rejected() -> None:
    """Starting at sigma 0.5 from pure noise is silently wrong, so it must raise.

    `get_schedule` happily produces a window starting at 0.5, and nothing downstream can tell that
    the sample handed to it is full-magnitude noise rather than a half-denoised latent.
    """
    noise = torch.randn(1, 128, 4, 4)

    with pytest.raises(ValueError, match="denoising_start must be 0"):
        _invocation(denoising_start=0.5)._prepare_initial_latents(noise, None, first_sigma=0.5)


def test_init_latents_are_blended_with_noise_at_the_first_sigma() -> None:
    """Image-to-image must preblend; passing init latents through clean is the bug this pins."""
    noise = torch.ones(1, 128, 4, 4)
    init = torch.zeros(1, 128, 4, 4)

    result = _invocation(denoising_start=0.25)._prepare_initial_latents(noise, init, first_sigma=0.75)

    # 0.75 * 1 + 0.25 * 0 -- and critically NOT `init` itself.
    assert torch.allclose(result, torch.full_like(noise, 0.75))
    assert not torch.equal(result, init)


def test_full_range_init_latents_collapse_to_pure_noise() -> None:
    """At `denoising_start=0` the first sigma is 1.0, so the init latents contribute nothing.

    That is the correct rectified-flow reading of "start from scratch" and matches Z-Image; pinned
    so the blend weights cannot be silently inverted.
    """
    noise = torch.randn(1, 128, 4, 4)
    init = torch.randn(1, 128, 4, 4)

    result = _invocation()._prepare_initial_latents(noise, init, first_sigma=1.0)

    assert torch.allclose(result, noise)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 4, 4),  # unpatchified -- 32 latent channels instead of the patched 128
        (1, 128, 8, 8),  # right channels, wrong spatial grid
        (2, 128, 4, 4),  # batched, but the text conditioning is built for batch 1
    ],
)
def test_mismatched_init_latents_are_rejected(shape: tuple[int, ...]) -> None:
    """A mismatch must fail loudly here rather than broadcast or blow up inside the transformer."""
    noise = torch.randn(1, 128, 4, 4)
    init = torch.randn(*shape)

    with pytest.raises(ValueError, match="patchified"):
        _invocation()._prepare_initial_latents(noise, init, first_sigma=1.0)
