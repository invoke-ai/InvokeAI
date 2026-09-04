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

from invokeai.backend.flux2.sampling_utils import compute_empirical_mu, get_schedule_flux2, time_shift_flux2


def _txt2img_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """The scheduler exactly as Flux2DenoiseInvocation builds it for txt2img."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        time_shift_type="exponential",
    )


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
