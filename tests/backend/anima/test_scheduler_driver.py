"""Tests for AnimaSchedulerDriver — the helper that hides per-scheduler API quirks
(sigmas= vs num_inference_steps=, Heun's doubled timestep array, set_begin_index)
behind a uniform iteration interface."""

import inspect

import pytest
import torch

from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift
from invokeai.backend.anima.scheduler_driver import AnimaSchedulerDriver
from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_MAP, ANIMA_SHIFT


def _anima_sigmas(num_steps: int) -> list[float]:
    return [loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / num_steps) for i in range(num_steps + 1)]


@pytest.mark.parametrize("scheduler_name", ["heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde"])
def test_driver_full_schedule_iteration_count(scheduler_name: str) -> None:
    """For full schedules (no clipping), the driver yields enough iterations to
    cover one full denoise. Heun yields 2N-1 iterations for N user steps."""
    num_steps = 8
    sigmas = _anima_sigmas(num_steps)
    driver = AnimaSchedulerDriver(
        scheduler_name=scheduler_name,
        sigmas=sigmas,
        steps=num_steps,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    iterations = list(driver.iterations())
    if driver.is_heun:
        assert len(iterations) == 2 * num_steps - 1
    else:
        assert len(iterations) == num_steps


@pytest.mark.parametrize("scheduler_name", ["dpmpp_2m", "dpmpp_2m_sde", "er_sde"])
def test_driver_single_step_schedulers_complete_user_step_every_iteration(scheduler_name: str) -> None:
    """Non-Heun schedulers report completes_user_step on every iteration."""
    num_steps = 6
    driver = AnimaSchedulerDriver(
        scheduler_name=scheduler_name,
        sigmas=_anima_sigmas(num_steps),
        steps=num_steps,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    user_step_count = sum(1 for it in driver.iterations() if it.completes_user_step)
    assert user_step_count == num_steps


def test_driver_heun_completes_user_step_on_second_order_and_terminal() -> None:
    """Heun yields one completion per user step: each pair's 2nd-order half plus
    the unpaired terminal 1st-order step (sigma_prev==0)."""
    num_steps = 4
    driver = AnimaSchedulerDriver(
        scheduler_name="heun",
        sigmas=_anima_sigmas(num_steps),
        steps=num_steps,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    # state_in_first_order only toggles once scheduler.step runs, so drive a fake
    # step per iteration to mirror production behaviour.
    completes_flags = []
    for it in driver.iterations():
        completes_flags.append(it.completes_user_step)
        driver.scheduler.step(
            model_output=torch.zeros((1, 1, 1, 4, 4)),
            timestep=it.sched_timestep,
            sample=torch.zeros((1, 1, 1, 4, 4)),
        )
    # N=4 → 7 iterations: indices 1, 3, 5 (SO halves) + 6 (terminal FO) = 4 completions.
    assert sum(completes_flags) == num_steps
    assert completes_flags[-1] is True, "terminal Heun first-order step must complete its user step"


@pytest.mark.parametrize(
    ("denoising_start", "denoising_end"),
    [(0.0, 1.0), (0.2, 1.0), (0.0, 0.8), (0.2, 0.8), (0.5, 0.75)],
)
def test_driver_dpmpp_clipped_schedule_starts_at_correct_sigma(denoising_start: float, denoising_end: float) -> None:
    """DPM++ doesn't accept sigmas= on diffusers 0.35.1; the driver's set_begin_index
    fallback must expose a first iteration whose sigma matches the clipped Anima reference.

    DPM++ constructs its internal flow schedule via ``np.linspace(1, T, N+1)[:-1]`` rather
    than the closed-form Anima loglinear shift, so the leading-edge sigma is offset by up
    to ~2e-3 from the Anima reference. That offset is a property of the scheduler family,
    not the driver — same offset exists in the pre-refactor code path.
    """
    num_steps = 30
    full_sigmas = _anima_sigmas(num_steps)
    k_start = int(denoising_start * num_steps)
    expected_first_sigma = full_sigmas[k_start]

    cls, _ = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    accepts_sigmas = "sigmas" in inspect.signature(cls(num_train_timesteps=1000).set_timesteps).parameters

    driver = AnimaSchedulerDriver(
        scheduler_name="dpmpp_2m",
        sigmas=full_sigmas[k_start : int(denoising_end * num_steps) + 1] if accepts_sigmas else full_sigmas,
        steps=num_steps,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        device=torch.device("cpu"),
        seed=0,
    )
    first_iter = next(driver.iterations())
    assert abs(first_iter.sigma_curr - expected_first_sigma) < 2e-3


@pytest.mark.parametrize(
    ("denoising_start", "denoising_end", "steps"),
    [(0.2, 0.8, 30), (0.0, 0.8, 30), (0.2, 1.0, 30), (0.5, 0.75, 20)],
)
def test_driver_heun_clipped_schedule_iteration_count(denoising_start: float, denoising_end: float, steps: int) -> None:
    """Heun clipped schedule: iteration count is 2*(k_end-k_start), clamped so
    denoising_end=1.0 doesn't run past the 2N-1 array."""
    full_sigmas = _anima_sigmas(steps)
    k_start = int(denoising_start * steps)
    k_end = int(denoising_end * steps)

    driver = AnimaSchedulerDriver(
        scheduler_name="heun",
        sigmas=full_sigmas,
        steps=steps,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        device=torch.device("cpu"),
        seed=0,
    )

    # If Heun's set_timesteps accepts sigmas=, the driver will pass the full schedule directly
    # and yield 2*steps-1 iterations regardless of clipping. The set_begin_index path applies
    # only when sigmas= is unsupported.
    accepts_sigmas = "sigmas" in inspect.signature(driver.scheduler.set_timesteps).parameters
    if accepts_sigmas:
        # Driver took the sigma-passing path; sigmas were not pre-clipped here, so the count
        # reflects the full schedule.
        assert driver.num_iterations == 2 * steps - 1
        return

    expected = min(2 * (k_end - k_start), len(driver.scheduler.timesteps) - driver.begin_index)
    assert driver.num_iterations == expected
    assert driver.begin_index == 2 * k_start


def test_driver_terminal_sigma_prev_is_zero() -> None:
    """The last iteration's sigma_prev must be 0.0 (terminal noise level)."""
    driver = AnimaSchedulerDriver(
        scheduler_name="dpmpp_2m",
        sigmas=_anima_sigmas(8),
        steps=8,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    last_iter = list(driver.iterations())[-1]
    assert last_iter.sigma_prev == 0.0


def test_driver_seed_determinism() -> None:
    """Same seed → identical step_generator state → reproducible SDE noise."""
    sigmas = _anima_sigmas(8)
    driver_a = AnimaSchedulerDriver(
        scheduler_name="er_sde",
        sigmas=sigmas,
        steps=8,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=42,
    )
    driver_b = AnimaSchedulerDriver(
        scheduler_name="er_sde",
        sigmas=sigmas,
        steps=8,
        denoising_start=0.0,
        denoising_end=1.0,
        device=torch.device("cpu"),
        seed=42,
    )
    # Same seed → same first random draw.
    a = torch.randn((1, 4), generator=driver_a.step_generator)
    b = torch.randn((1, 4), generator=driver_b.step_generator)
    assert torch.equal(a, b)
