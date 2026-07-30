import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

from invokeai.backend.ernie_image.denoise import denoise
from invokeai.backend.ernie_image.sampling_utils import get_schedule
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


class _StubTransformer(torch.nn.Module):
    """Stands in for `ErnieImageTransformer2DModel`; returns a velocity of the right shape."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def _run(scheduler, steps: int, denoising_start: float = 0.0, denoising_end: float = 1.0):
    states: list[PipelineIntermediateState] = []
    # 128 channels is the patched channel count; the spatial dims are tiny to keep this fast.
    img = torch.zeros(1, 128, 2, 2)
    text_bth = torch.zeros(1, 4, 8)
    text_lens = torch.tensor([4])
    sigmas = get_schedule(steps, denoising_start=denoising_start, denoising_end=denoising_end)

    denoise(
        model=_StubTransformer(),
        img=img,
        text_bth=text_bth,
        text_lens=text_lens,
        timesteps=sigmas.tolist(),
        step_callback=states.append,
        cfg_scale=[1.0] * (len(sigmas) - 1),
        scheduler=scheduler,
    )
    return states, sigmas


def test_euler_progress_matches_requested_steps() -> None:
    states, _ = _run(FlowMatchEulerDiscreteScheduler(), steps=8)

    assert len(states) == 8
    # Progress must never overrun: the last emitted step equals the reported total.
    assert states[-1].step == states[-1].total_steps == 8


def test_heun_progress_does_not_overrun_total_steps() -> None:
    # Heun is 2nd order: `set_timesteps(8)` yields 15 timesteps, so the loop runs 15 times.
    # `total_steps` must reflect the real iteration count, otherwise progress runs past 100%.
    states, _ = _run(FlowMatchHeunDiscreteScheduler(), steps=8)

    assert len(states) > 8
    assert all(s.step <= s.total_steps for s in states)
    assert states[-1].step == states[-1].total_steps == len(states)


def test_heun_rejects_a_partial_denoise_range() -> None:
    # Heun's `set_timesteps` takes only a step count, so it cannot honor a custom sigma window.
    # Silently running a full denoise instead would be worse than refusing.
    with pytest.raises(ValueError, match="denoising_start/denoising_end"):
        _run(FlowMatchHeunDiscreteScheduler(), steps=8, denoising_start=0.5)


@pytest.mark.parametrize("denoising_end", [1.0, 0.75, 0.5])
def test_denoising_end_stops_at_the_requested_sigma(denoising_end: float) -> None:
    """The scheduler must terminate at the requested end sigma, not at 0.

    Every FlowMatch scheduler appends its own terminal 0 sigma in `set_timesteps`. If the driver
    hands it the window minus its last entry, that appended 0 silently replaces the requested end
    sigma -- so `denoising_end < 1.0` would run a *full* denoise in fewer, coarser steps instead of
    stopping early, which breaks every multi-stage handoff built on this field.
    """
    scheduler = FlowMatchEulerDiscreteScheduler()
    states, sigmas = _run(scheduler, steps=8, denoising_end=denoising_end)

    scheduler_sigmas = scheduler.sigmas
    assert scheduler_sigmas is not None
    assert float(scheduler_sigmas[-1]) == pytest.approx(float(sigmas[-1]), abs=1e-6)
    # One step per adjacent sigma pair, and no step is dropped or duplicated.
    assert len(states) == len(sigmas) - 1


def test_denoising_end_keeps_the_schedulers_own_shift() -> None:
    """The terminal sigma must be shifted by the scheduler, not spliced in raw.

    Truncating the scheduler's appended zero (rather than overwriting the last sigma by hand) is
    what keeps this true: with `shift=3`, the requested end sigma 0.25 has to land on
    `3 * 0.25 / (1 + 2 * 0.25) = 0.5`, matching the shift applied to every other sigma.
    """
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    _run(scheduler, steps=8, denoising_end=0.75)

    scheduler_sigmas = scheduler.sigmas
    assert scheduler_sigmas is not None
    assert float(scheduler_sigmas[-1]) == pytest.approx(0.5, abs=1e-6)


def test_degenerate_denoising_window_is_rejected() -> None:
    """A window that rounds down to a single sigma yields zero steps.

    The loop would then return its input untouched and the graph would decode raw noise with no
    error at all, so `get_schedule` has to refuse rather than hand back a one-entry schedule.
    """
    with pytest.raises(ValueError, match="rounds to zero steps"):
        get_schedule(8, denoising_end=0.1)
