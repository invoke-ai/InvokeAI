import math

import pytest
import torch

from invokeai.backend.flux.sampling_utils import (
    MU_FIT_MAX_SEQ_LEN,
    MU_FIT_MIN_SEQ_LEN,
    clip_timestep_schedule,
    clip_timestep_schedule_fractional,
    get_lin_function,
    get_schedule,
)


def float_lists_almost_equal(list1: list[float], list2: list[float], tol: float = 1e-6) -> bool:
    return all(abs(a - b) < tol for a, b in zip(list1, list2, strict=True))


@pytest.mark.parametrize(
    ["denoising_start", "denoising_end", "expected_timesteps", "raises"],
    [
        (0.0, 1.0, [1.0, 0.75, 0.5, 0.25, 0.0], False),  # Default case.
        (-0.1, 1.0, [], True),  # Negative denoising_start should raise.
        (0.0, 1.1, [], True),  # denoising_end > 1 should raise.
        (0.5, 0.0, [], True),  # denoising_start > denoising_end should raise.
        (0.0, 0.0, [1.0], False),  # denoising_end == 0.
        (1.0, 1.0, [0.0], False),  # denoising_start == 1.
        (0.2, 0.8, [1.0, 0.75, 0.5, 0.25], False),  # Middle of the schedule.
        # If we denoise from 0.0 to x, then from x to 1.0, it is important that denoise_end = x and denoise_start = x
        # map to the same timestep. We test this first when x is equal to a timestep, then when it falls between two
        # timesteps.
        # x = 0.5
        (0.0, 0.5, [1.0, 0.75, 0.5], False),
        (0.5, 1.0, [0.5, 0.25, 0.0], False),
        # x = 0.3
        (0.0, 0.3, [1.0, 0.75], False),
        (0.3, 1.0, [0.75, 0.5, 0.25, 0.0], False),
    ],
)
def test_clip_timestep_schedule(
    denoising_start: float, denoising_end: float, expected_timesteps: list[float], raises: bool
):
    timesteps = torch.linspace(1, 0, 5).tolist()
    if raises:
        with pytest.raises(AssertionError):
            clip_timestep_schedule(timesteps, denoising_start, denoising_end)
    else:
        assert float_lists_almost_equal(
            clip_timestep_schedule(timesteps, denoising_start, denoising_end), expected_timesteps
        )


@pytest.mark.parametrize(
    ["denoising_start", "denoising_end", "expected_timesteps", "raises"],
    [
        (0.0, 1.0, [1.0, 0.75, 0.5, 0.25, 0.0], False),  # Default case.
        (-0.1, 1.0, [], True),  # Negative denoising_start should raise.
        (0.0, 1.1, [], True),  # denoising_end > 1 should raise.
        (0.5, 0.0, [], True),  # denoising_start > denoising_end should raise.
        (0.0, 0.0, [1.0], False),  # denoising_end == 0.
        (1.0, 1.0, [0.0], False),  # denoising_start == 1.
        (0.2, 0.8, [0.8, 0.75, 0.5, 0.25, 0.2], False),  # Middle of the schedule.
        # If we denoise from 0.0 to x, then from x to 1.0, it is important that denoise_end = x and denoise_start = x
        # map to the same timestep. We test this first when x is equal to a timestep, then when it falls between two
        # timesteps.
        # x = 0.5
        (0.0, 0.5, [1.0, 0.75, 0.5], False),
        (0.5, 1.0, [0.5, 0.25, 0.0], False),
        # x = 0.3
        (0.0, 0.3, [1.0, 0.75, 0.7], False),
        (0.3, 1.0, [0.7, 0.5, 0.25, 0.0], False),
    ],
)
def test_clip_timestep_schedule_fractional(
    denoising_start: float, denoising_end: float, expected_timesteps: list[float], raises: bool
):
    timesteps = torch.linspace(1, 0, 5).tolist()
    if raises:
        with pytest.raises(AssertionError):
            clip_timestep_schedule_fractional(timesteps, denoising_start, denoising_end)
    else:
        assert float_lists_almost_equal(
            clip_timestep_schedule_fractional(timesteps, denoising_start, denoising_end), expected_timesteps
        )


def mu_of(image_seq_len: int) -> float:
    """Recover the shift the schedule was built with, through the public function.

    `get_schedule(2, n)` is [1.0, x, 0.0] with x = exp(mu) / (exp(mu) + 1), because `time_shift`'s
    `(1 / t - 1) ** sigma` term is 1 at t = 0.5 for every sigma. That makes the readback exact but
    blind to the *shape* of the shift -- `test_the_shift_has_the_shape_this_readback_assumes` pins
    that separately, and without it a changed sigma would slip past every cell below.
    """
    x = get_schedule(num_steps=2, image_seq_len=image_seq_len, shift=True)[1]
    return math.log(x / (1.0 - x))


def test_the_shift_has_the_shape_this_readback_assumes():
    """One schedule pinned outright, computed by hand from exp(mu) / (exp(mu) + (1 / t - 1)).

    The interesting entries are the asymmetric ones: at t = 0.75 and t = 0.25 a sigma of 2 would
    give 0.966 and 0.260 instead, while t = 0.5 would be unchanged.
    """
    assert float_lists_almost_equal(
        get_schedule(num_steps=4, image_seq_len=MU_FIT_MAX_SEQ_LEN, shift=True),
        [1.0, 0.904531, 0.759511, 0.512844, 0.0],
        tol=1e-5,
    )


def test_the_anchors_are_where_the_clamp_assumes_they_are():
    """Written as literals on purpose.

    Asserting `line(MU_FIT_MAX_SEQ_LEN) == 1.15` would hold for any anchor value, because the
    constants are also `get_lin_function`'s own defaults -- it would pass with the anchor moved to
    8192, which is exactly the drift it is supposed to catch.
    """
    assert MU_FIT_MIN_SEQ_LEN == 256
    assert MU_FIT_MAX_SEQ_LEN == 4096
    assert get_lin_function()(4096) == pytest.approx(1.15)


@pytest.mark.parametrize(
    ["image_seq_len", "expected_mu"],
    [
        (MU_FIT_MIN_SEQ_LEN, 0.5),  # 256px: the lower point of the fit
        (1024, 0.63),  # 512px: interpolation, unaffected
        (MU_FIT_MAX_SEQ_LEN, 1.15),  # 1024px: the upper point, where the fit ends
        (9216, 1.15),  # 1536px: beyond the fit, held
        (16384, 1.15),  # 2048px: extrapolation would give 3.23
        (36864, 1.15),  # 3072px: extrapolation would give 6.70
        (65536, 1.15),  # 4096px: extrapolation would give 11.55
    ],
)
def test_the_shift_is_not_extrapolated_past_the_fit(image_seq_len: int, expected_mu: float):
    assert mu_of(image_seq_len) == pytest.approx(expected_mu, abs=1e-3)


def test_the_fit_is_still_live_below_its_upper_point():
    """Holding the fit at its endpoint is not the same as replacing it with a constant, and this
    names that choice: below one megapixel the schedule still adapts to the frame."""
    assert mu_of(512) < mu_of(1024) < mu_of(2048) < mu_of(MU_FIT_MAX_SEQ_LEN)


@pytest.mark.parametrize(
    ["image_seq_len", "why"],
    [
        (MU_FIT_MAX_SEQ_LEN, "1024px: the anchor, a control -- unchanged by the clamp either way"),
        (16384, "2048px: 2 of 30 steps survived before the clamp"),
        (36864, "3072px: 1 of 30 survived before the clamp"),
    ],
)
def test_the_clipped_schedule_keeps_the_same_steps_at_every_frame_size(image_seq_len: int, why: str):
    """An extrapolated shift pushed the schedule above the `denoising_start` line, so the clip left one
    or two of the 30 steps asked for here -- an img2img or upscale pass that silently did almost nothing.
    Asking for more barely helped: at 2048px, 100 steps kept 4 and 1000 kept 39.

    0.499 is not arbitrary: it is what the upscale widget's creativity slider sends at its middle
    position, ((0 * -1 + 10) * 4.99) / 100 (`features/upscale/core/graph.ts`).
    """
    timesteps = get_schedule(num_steps=30, image_seq_len=image_seq_len, shift=True)
    assert len(clip_timestep_schedule_fractional(timesteps, 0.499, 1.0)) - 1 == 8, why


def test_schnell_is_untouched_because_it_does_not_shift():
    unshifted = get_schedule(num_steps=4, image_seq_len=65536, shift=False)
    assert float_lists_almost_equal(unshifted, torch.linspace(1, 0, 5).tolist())
