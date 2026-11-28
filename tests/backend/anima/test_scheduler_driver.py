"""Tests for AnimaSchedulerDriver — the helper that hides per-scheduler API quirks
(sigmas= vs num_inference_steps=, Heun's doubled timestep array, set_begin_index)
behind a uniform iteration interface."""

import inspect

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import loglinear_timestep_shift
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
    the unpaired terminal 1st-order step."""
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


def _clipped_driver(scheduler_name: str, denoising_start: float, denoising_end: float, steps: int, seed: int = 0):
    """Build the driver as `anima_denoise` does: it hands over the schedule already clipped by index."""
    full_sigmas = _anima_sigmas(steps)
    clipped = full_sigmas[int(denoising_start * steps) : int(denoising_end * steps) + 1]
    driver = AnimaSchedulerDriver(
        scheduler_name=scheduler_name,
        sigmas=clipped,
        steps=steps,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        device=torch.device("cpu"),
        seed=seed,
    )
    return driver, clipped


@pytest.mark.parametrize("scheduler_name", ["er_sde", "lcm"])
@pytest.mark.parametrize(
    ("denoising_start", "denoising_end"),
    [(0.0, 1.0), (0.3, 1.0), (0.5, 1.0), (0.0, 0.8), (0.5, 0.75)],
)
def test_driver_starts_at_the_first_sigma_of_the_clipped_schedule(
    scheduler_name: str, denoising_start: float, denoising_end: float
) -> None:
    """The img2img blend noises the latents to the clipped schedule's first sigma, so that is where every
    scheduler that takes sigmas has to start (Heun and LCM used to start at their own 1.0 / unshifted plan)."""
    driver, clipped = _clipped_driver(scheduler_name, denoising_start, denoising_end, steps=30)

    first = next(driver.iterations())

    assert first.sigma_curr == pytest.approx(clipped[0], abs=1e-6)


@pytest.mark.parametrize(
    ("denoising_start", "denoising_end"),
    [(0.0, 1.0), (0.3, 1.0), (0.5, 1.0), (0.0, 0.8), (0.5, 0.75)],
)
def test_driver_heun_runs_the_clipped_schedule_in_its_doubled_layout(
    denoising_start: float, denoising_end: float
) -> None:
    driver, clipped = _clipped_driver("heun", denoising_start, denoising_end, steps=30)
    user_steps = len(clipped) - 1

    iterations = list(driver.iterations())

    assert driver.num_iterations == len(iterations) == 2 * user_steps - 1
    assert [it.sigma_curr for it in iterations] == pytest.approx(
        [clipped[0]] + [s for s in clipped[1:-1] for _ in (0, 1)], abs=1e-6
    )
    # The last iteration ends at the requested terminal sigma, which is not 0 when denoising_end < 1.
    assert iterations[-1].sigma_prev == pytest.approx(clipped[-1], abs=1e-6)


@pytest.mark.parametrize("denoising_end", [1.0, 0.75])
def test_driver_heun_progress_reaches_every_user_step(denoising_end: float) -> None:
    """Fed through scheduler.step like production, Heun completes exactly one user step per interval,
    including the terminal first-order step when the pass ends before sigma 0."""
    driver, clipped = _clipped_driver("heun", 0.5, denoising_end, steps=20)
    completions = 0
    for it in driver.iterations():
        completions += it.completes_user_step
        driver.scheduler.step(
            model_output=torch.zeros((1, 1, 1, 4, 4)),
            timestep=it.sched_timestep,
            sample=torch.zeros((1, 1, 1, 4, 4)),
        )

    assert completions == len(clipped) - 1


@pytest.mark.parametrize(("denoising_start", "denoising_end"), [(0.0, 1.0), (0.5, 1.0), (0.5, 0.75)])
def test_driver_lcm_runs_the_clipped_schedule_to_its_terminal_sigma(
    denoising_start: float, denoising_end: float
) -> None:
    driver, clipped = _clipped_driver("lcm", denoising_start, denoising_end, steps=20)

    iterations = list(driver.iterations())

    assert [it.sigma_curr for it in iterations] == pytest.approx(clipped[:-1], abs=1e-6)
    assert iterations[-1].sigma_prev == pytest.approx(clipped[-1], abs=1e-6)
    assert all(it.completes_user_step for it in iterations)


def test_driver_lcm_step_noise_follows_the_seed() -> None:
    """LCM re-noises the sample on every step: it has to draw from the driver's seeded generator, not the global RNG."""

    def run(seed: int) -> torch.Tensor:
        torch.rand(64)  # advance the global RNG like a long-lived server does
        driver, _ = _clipped_driver("lcm", 0.5, 1.0, steps=8, seed=seed)
        sample = torch.zeros((1, 1, 1, 4, 4))
        for it in driver.iterations():
            sample = driver.step(torch.zeros_like(sample), it.sched_timestep, sample)
        return sample

    assert torch.equal(run(1234), run(1234))
    assert not torch.equal(run(1), run(2))


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
