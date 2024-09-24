import pytest
import torch

from invokeai.backend.flux.sampling_utils import clip_timestep_schedule, clip_timestep_schedule_fractional


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
