"""The FLUX.2 Diffusers scheduler path runs the schedule it is handed.

The node passes the shifted, clipped schedule: the sigmas the model is evaluated at plus the terminal
sigma the last step ends at. Euler used to drop that terminal sigma (Diffusers appends 0, so a pass
with denoising_end < 1 finished the image instead of stopping), and Heun replaced the whole schedule
with its own static shift.
"""

from typing import Any

import pytest
import torch
from diffusers import FlowMatchHeunDiscreteScheduler

from invokeai.app.util.step_callback import calc_percentage
from invokeai.backend.flux.schedulers import _HAS_LCM, FLUX_SCHEDULER_MAP
from invokeai.backend.flux2.denoise import denoise

VELOCITY = 0.25


class _ConstantVelocityModel:
    """Predicts a constant velocity and records the sigma of every evaluation.

    With a constant velocity every first- or second-order flow-matching solver moves the sample by
    exactly (sigma_end - sigma_start) * velocity, so the result shows where the schedule ended.
    """

    def __init__(self) -> None:
        self.sigmas: list[float] = []

    def __call__(self, hidden_states: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        del kwargs
        self.sigmas.append(float(timestep[0]))
        return (torch.full_like(hidden_states, VELOCITY),)


class _SampleDependentModel:
    """Predicts from the sample, so each evaluation's clean estimate differs."""

    def __init__(self) -> None:
        self.evaluations: list[tuple[float, torch.Tensor]] = []

    def __call__(self, hidden_states: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        del kwargs
        sigma = float(timestep[0])
        self.evaluations.append((sigma, hidden_states.clone()))
        return (0.1 * hidden_states + 0.2 * sigma,)


def _run(scheduler, timesteps: list[float], model=None):
    model = model or _ConstantVelocityModel()
    img = torch.tensor([[[1.0], [-0.5]]])
    callbacks = []
    result = denoise(
        model=model,  # type: ignore[arg-type]
        img=img,
        img_ids=torch.zeros(1, 2, 4, dtype=torch.long),
        txt=torch.zeros(1, 3, 1),
        txt_ids=torch.zeros(1, 3, 4, dtype=torch.long),
        timesteps=timesteps,
        step_callback=callbacks.append,
        guidance=1.0,
        cfg_scale=[1.0] * (len(timesteps) - 1),
        scheduler=scheduler,
    )
    return result, img, model, callbacks


@pytest.mark.parametrize("terminal_sigma", [0.0, 0.4])
def test_euler_runs_the_given_sigmas_and_stops_at_the_terminal_sigma(terminal_sigma: float):
    schedule = [1.0, 0.95, 0.85, 0.6, terminal_sigma]
    scheduler = FLUX_SCHEDULER_MAP["euler"](num_train_timesteps=1000)

    result, img, model, callbacks = _run(scheduler, schedule)

    assert model.sigmas == pytest.approx(schedule[:-1])
    assert torch.equal(scheduler.sigmas[-1], torch.tensor(terminal_sigma))
    assert torch.allclose(result, img + (terminal_sigma - schedule[0]) * VELOCITY)
    assert [callback.step for callback in callbacks] == [1, 2, 3, 4]
    assert all(callback.order == 1 and callback.total_steps == 4 for callback in callbacks)


def test_heun_runs_the_given_sigmas_including_the_terminal_sigma():
    schedule = [1.0, 0.95, 0.85, 0.6, 0.4]
    scheduler = FLUX_SCHEDULER_MAP["heun"](num_train_timesteps=1000)

    result, img, model, callbacks = _run(scheduler, schedule)

    # Diffusers' Heun layout: the first sigma once, every later one twice, then the terminal sigma.
    expected_sigmas = [1.0, 0.95, 0.95, 0.85, 0.85, 0.6, 0.6, 0.4]
    assert scheduler.sigmas.tolist() == pytest.approx(expected_sigmas)
    assert model.sigmas == pytest.approx(expected_sigmas[:-1])
    assert torch.allclose(result, img + (schedule[-1] - schedule[0]) * VELOCITY, atol=1e-6, rtol=0)


@pytest.mark.parametrize("step_count", [4, 5])
def test_heun_progress_reaches_every_user_step(step_count: int):
    schedule = torch.linspace(1.0, 0.0, step_count + 1).tolist()

    _, _, _, callbacks = _run(FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000), schedule)

    assert [callback.step for callback in callbacks] == list(range(1, step_count + 1))
    assert [calc_percentage(callback) for callback in callbacks] == pytest.approx(
        [step / step_count for step in range(1, step_count + 1)]
    )


@pytest.mark.parametrize("scheduler_name", ["euler", "heun"])
def test_previews_use_the_sample_and_sigma_of_the_model_evaluation(scheduler_name: str):
    scheduler = FLUX_SCHEDULER_MAP[scheduler_name](num_train_timesteps=1000)

    _, img, _, callbacks = _run(scheduler, [1.0, 0.7, 0.35, 0.0])

    # With a constant velocity the clean estimate x - sigma * v is the same at every sigma.
    assert len(callbacks) == 3
    assert all(torch.allclose(callback.latents, img - VELOCITY) for callback in callbacks)


def test_heun_previews_come_from_each_intervals_second_order_evaluation():
    schedule = [0.9, 0.6, 0.3, 0.1]
    model = _SampleDependentModel()

    _, _, _, callbacks = _run(FLUX_SCHEDULER_MAP["heun"](num_train_timesteps=1000), schedule, model=model)

    # Evaluations: 0.9 (first order), 0.6 (second order), 0.6 (first order), 0.3 (second order), then the
    # final interval's first-order step from 0.3, which has no second-order evaluation.
    assert [sigma for sigma, _ in model.evaluations] == pytest.approx([0.9, 0.6, 0.6, 0.3, 0.3])
    for callback, evaluation_index in zip(callbacks, [1, 3, 4], strict=True):
        sigma, sample = model.evaluations[evaluation_index]
        assert torch.allclose(callback.latents, sample - sigma * (0.1 * sample + 0.2 * sigma))


@pytest.mark.skipif(not _HAS_LCM, reason="FlowMatchLCMScheduler not available in this Diffusers version")
@pytest.mark.parametrize("terminal_sigma", [0.0, 0.4])
def test_lcm_runs_the_given_sigmas_and_renoises_to_the_terminal_sigma(terminal_sigma: float):
    schedule = [1.0, 0.95, 0.85, 0.6, terminal_sigma]
    scheduler = FLUX_SCHEDULER_MAP["lcm"](num_train_timesteps=1000)

    _, _, model, callbacks = _run(scheduler, schedule)

    assert model.sigmas == pytest.approx(schedule[:-1])
    assert torch.equal(scheduler.sigmas[-1], torch.tensor(terminal_sigma))
    assert [callback.step for callback in callbacks] == [1, 2, 3, 4]


def test_an_unrelated_scheduler_is_rejected():
    class _UnrelatedScheduler:
        def set_timesteps(self, sigmas: list[float], device: torch.device) -> None:
            raise AssertionError("must not be configured")

    with pytest.raises(TypeError, match="Unsupported FLUX.2 scheduler"):
        _run(_UnrelatedScheduler(), [1.0, 0.5, 0.0])
