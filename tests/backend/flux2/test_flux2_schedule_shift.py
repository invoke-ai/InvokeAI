"""Regression tests for the FLUX.2 schedule shift shared by the txt2img and img2img paths.

``get_schedule_flux2()`` returns an unshifted linear schedule because the txt2img scheduler applies
the exponential shift from ``mu`` itself. img2img and inpainting step the schedule manually and must
apply the same shift, otherwise they run the model on a sigma trajectory txt2img never visits -- with
9 steps the shifted schedule bottoms out at 0.485 while the linear one runs down to 0.111, and a
distilled model like FLUX.2 Klein leaves a grainy residue down there.
"""

import numpy as np
import pytest
from diffusers import FlowMatchEulerDiscreteScheduler

from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux2.sampling_utils import (
    FLUX2_TXT2IMG_SCHEDULER_KWARGS,
    compute_empirical_mu,
    get_schedule_flux2,
    redensify_schedule_flux2,
    time_shift_flux2,
    unshift_flux2,
)


def _txt2img_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """The scheduler exactly as Flux2DenoiseInvocation builds it for txt2img.

    Built from the same config the node uses rather than a copy of it, so the two cannot drift apart
    without this test noticing.
    """
    return FlowMatchEulerDiscreteScheduler(**FLUX2_TXT2IMG_SCHEDULER_KWARGS)


@pytest.mark.parametrize("num_steps", [4, 9, 20, 30])
def test_shift_matches_the_txt2img_scheduler(num_steps: int) -> None:
    """The manual shift must reproduce the sigmas the txt2img scheduler produces."""
    image_seq_len = 64 * 64
    timesteps = get_schedule_flux2(num_steps=num_steps, image_seq_len=image_seq_len)
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)

    scheduler = _txt2img_scheduler()
    scheduler.set_timesteps(sigmas=timesteps[:-1], mu=mu)

    shifted = time_shift_flux2(timesteps, mu)
    np.testing.assert_allclose(shifted, [float(s) for s in scheduler.sigmas], rtol=0, atol=1e-6)


def test_shift_is_a_no_op_at_the_endpoints() -> None:
    """1.0 and 0.0 are fixed points, and 0.0 must not divide by zero."""
    shifted = time_shift_flux2([1.0, 0.5, 0.0], mu=2.02)
    assert shifted[0] == 1.0
    assert shifted[-1] == 0.0


def test_shift_is_strictly_decreasing_and_bounded() -> None:
    timesteps = get_schedule_flux2(num_steps=30, image_seq_len=64 * 64)
    shifted = time_shift_flux2(timesteps, mu=compute_empirical_mu(image_seq_len=64 * 64, num_steps=30))

    assert all(0.0 <= s <= 1.0 for s in shifted)
    assert all(a > b for a, b in zip(shifted[:-1], shifted[1:], strict=True))


def test_shift_raises_the_schedule_floor_above_the_linear_one() -> None:
    """The property that matters: the model is never asked for the low sigmas of the linear schedule."""
    num_steps = 9
    linear = get_schedule_flux2(num_steps=num_steps, image_seq_len=64 * 64)
    shifted = time_shift_flux2(linear, mu=compute_empirical_mu(image_seq_len=64 * 64, num_steps=num_steps))

    # Lowest sigma the model is actually evaluated at (the final 0.0 entry is the step target, not a
    # timestep the model is called with).
    assert linear[-2] == pytest.approx(1 / num_steps)
    assert shifted[-2] > 0.45
    assert all(s >= lin for s, lin in zip(shifted, linear, strict=True))


def test_unshift_inverts_the_shift() -> None:
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=9)
    sigmas = [i / 200 for i in range(201)]

    assert unshift_flux2(time_shift_flux2(sigmas, mu), mu) == pytest.approx(sigmas, abs=1e-12)


@pytest.mark.parametrize("num_steps", [4, 9, 20])
@pytest.mark.parametrize("strength", [0.05, 0.2, 0.5, 0.75])
def test_redensify_restores_the_requested_step_count(num_steps: int, strength: float) -> None:
    """A clipped window is stepped with the number of steps that was actually requested.

    This is the img2img case: the canvas maps UI strength to ``denoising_start = 1 - s**0.2``, and
    clipping the shifted schedule to that window drops most of the steps.
    """
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=num_steps)
    shifted = time_shift_flux2(get_schedule_flux2(num_steps=num_steps, image_seq_len=64 * 64), mu)
    clipped = clip_timestep_schedule_fractional(shifted, 1 - strength**0.2, 1.0)
    assert len(clipped) - 1 < num_steps, "precondition: clipping must have dropped steps"

    respaced = redensify_schedule_flux2(clipped, num_steps, mu)

    assert len(respaced) - 1 == num_steps
    # The endpoints define the denoising range and the sigma the start latents are blended at, so
    # they must come back bit-exact, not merely close.
    assert respaced[0] == clipped[0]
    assert respaced[-1] == clipped[-1]
    assert all(a > b for a, b in zip(respaced[:-1], respaced[1:], strict=True))


def test_redensify_follows_the_full_schedule_through_the_window() -> None:
    """Respacing samples the shifted curve, it does not draw a straight line across the window.

    Spacing the interior points evenly in shifted space instead of unshifted space would still give
    the right number of steps and the right endpoints, and every other test here would still pass --
    but the sigmas in between would be wrong. This pins them: the unshifted schedule is uniform, so
    respacing the window between two of its sigmas has to reproduce the full schedule's own sigmas
    in between, exactly.
    """
    num_steps = 8
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=num_steps)
    linear = get_schedule_flux2(num_steps=num_steps, image_seq_len=64 * 64)
    assert np.allclose(np.diff(linear), np.diff(linear)[0]), "precondition: the linear schedule is evenly spaced"
    full = time_shift_flux2(linear, mu)

    # The bare endpoints of the schedule's second half, respaced back to the 4 steps that span it.
    respaced = redensify_schedule_flux2([full[4], full[-1]], 4, mu)

    assert respaced == pytest.approx(full[4:], abs=1e-12)
    # Sanity: the interior is genuinely curved, so a straight line across the window would differ.
    straight = np.linspace(full[4], full[-1], 5)
    assert not np.allclose(respaced, straight, atol=1e-6)


@pytest.mark.parametrize("num_steps", [4, 9, 20])
def test_redensify_leaves_an_unclipped_schedule_alone(num_steps: int) -> None:
    """Strength 1.0 / txt2img must be untouched: the fix may only ever add steps back."""
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=num_steps)
    shifted = time_shift_flux2(get_schedule_flux2(num_steps=num_steps, image_seq_len=64 * 64), mu)
    clipped = clip_timestep_schedule_fractional(shifted, 0.0, 1.0)

    assert redensify_schedule_flux2(clipped, num_steps, mu) == clipped


def test_redensify_never_makes_a_schedule_sparser() -> None:
    """A window that already holds enough steps keeps every one of them."""
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=4)
    dense = [0.9, 0.8, 0.7, 0.6, 0.5, 0.0]

    assert redensify_schedule_flux2(dense, 4, mu) == dense


@pytest.mark.parametrize("degenerate", [[], [0.5], [0.5, 0.5]])
def test_redensify_passes_degenerate_schedules_through(degenerate: list[float]) -> None:
    """denoising_start == denoising_end leaves nothing to space out; the caller handles that case."""
    mu = compute_empirical_mu(image_seq_len=64 * 64, num_steps=4)

    assert redensify_schedule_flux2(degenerate, 4, mu) == degenerate
