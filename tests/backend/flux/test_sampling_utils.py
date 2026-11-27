import math

import pytest
import torch

from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule,
    clip_timestep_schedule_fractional,
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

    `get_schedule(2, n)` is [1.0, x, 0.0] with x = exp(mu) / (exp(mu) + 1), because `1 / t - 1` is 1 at
    t = 0.5, so any power of it is too. That makes the readback exact but blind to the *shape* of the
    shift -- `test_the_shift_has_the_shape_this_readback_assumes` pins that separately, and without it a
    changed form would slip past every cell below.
    """
    x = get_schedule(num_steps=2, image_seq_len=image_seq_len, shift=True)[1]
    return math.log(x / (1.0 - x))


def test_the_shift_has_the_shape_this_readback_assumes():
    """One schedule pinned outright, computed by hand from exp(mu) / (exp(mu) + (1 / t - 1)).

    The interesting entries are the asymmetric ones: at t = 0.75 and t = 0.25 a sigma of 2 would
    give 0.966 and 0.260 instead, while t = 0.5 would be unchanged.
    """
    assert float_lists_almost_equal(
        get_schedule(num_steps=4, image_seq_len=4096, shift=True),
        [1.0, 0.904531, 0.759511, 0.512844, 0.0],
        tol=1e-5,
    )


@pytest.mark.parametrize(
    ["image_seq_len", "expected_mu"],
    [
        (1, 0.5 - 0.65 * 255 / 3840),  # 16px: below the lower point the line keeps running (0.457)
        (256, 0.5),  # 256px: the lower point of the fit
        (1024, 0.63),  # 512px: interpolation
        (4096, 1.15),  # 1024px: the upper point, where the fit ends
        (16384, 1.15),  # 2048px: held; extrapolation would give 3.23
    ],
)
def test_the_shift_follows_its_anchors_and_holds_past_the_fit(image_seq_len: int, expected_mu: float):
    assert mu_of(image_seq_len) == pytest.approx(expected_mu, abs=1e-3)


@pytest.mark.parametrize(
    ["image_seq_len", "why"],
    [
        (4096, "1024px: the anchor, a control -- unchanged by the clamp either way"),
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
