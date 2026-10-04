"""Tests for `ErnieImageDenoiseInvocation`'s initial-latent handling.

ERNIE denoises on the rectified-flow path, so the loop assumes its input already sits at the first
sigma of the schedule. Both directions of that contract are easy to get silently wrong:

- clean init latents passed through *unnoised* tell the model the sample is at `sigma_0` when it is
  at 0 (guarded by `add_noise=True`, with the blend itself done in `denoise()`, which is the only
  layer that knows the post-shift sigma)
- pure noise at a *reduced* `sigma_0` tells the model it is already partly denoised

Neither produces an error at runtime -- just a bad image -- so they are pinned here.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.ernie_image.ernie_image_denoise import ErnieImageDenoiseInvocation
from invokeai.app.invocations.fields import LatentsField


def _invocation(*, denoising_start: float = 0.0, latents: LatentsField | None = None) -> ErnieImageDenoiseInvocation:
    return ErnieImageDenoiseInvocation.model_construct(denoising_start=denoising_start, latents=latents)


def _context(tensor: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(tensors=SimpleNamespace(load=lambda _name: tensor))


def _load(invocation: ErnieImageDenoiseInvocation, context: SimpleNamespace, noise: torch.Tensor):
    return invocation._load_init_latents(context, noise, device=torch.device("cpu"), dtype=torch.float32)  # type: ignore[arg-type]


def test_txt2img_reports_no_init_latents() -> None:
    noise = torch.randn(1, 128, 4, 4)

    assert _load(_invocation(), _context(), noise) is None


def test_denoising_start_without_init_latents_is_rejected() -> None:
    """Starting at sigma 0.5 from pure noise is silently wrong, so it must raise.

    `get_schedule` happily produces a window starting at 0.5, and nothing downstream can tell that
    the sample handed to it is full-magnitude noise rather than a half-denoised latent.
    """
    noise = torch.randn(1, 128, 4, 4)

    with pytest.raises(ValueError, match="denoising_start must be 0"):
        _load(_invocation(denoising_start=0.5), _context(), noise)


def test_init_latents_are_returned_for_the_loop_to_blend() -> None:
    """The invocation validates and hands them over; it must not blend them itself.

    Blending here would have to use the raw schedule sigma, since the scheduler's `shift` is only
    applied later inside `set_timesteps`.
    """
    noise = torch.randn(1, 128, 4, 4)
    init = torch.randn(1, 128, 4, 4)
    invocation = _invocation(denoising_start=0.25, latents=LatentsField(latents_name="init"))

    result = _load(invocation, _context(init), noise)

    assert result is not None
    assert torch.equal(result, init)


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
    invocation = _invocation(latents=LatentsField(latents_name="init"))

    with pytest.raises(ValueError, match="patchified"):
        _load(invocation, _context(torch.randn(*shape)), noise)


def test_add_noise_defaults_to_on() -> None:
    """A clean VAE-encoded latent is the common case, and it has to be noised to be usable."""
    assert ErnieImageDenoiseInvocation.model_fields["add_noise"].default is True
