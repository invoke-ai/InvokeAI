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
    return states


def test_euler_progress_matches_requested_steps() -> None:
    states = _run(FlowMatchEulerDiscreteScheduler(), steps=8)

    assert len(states) == 8
    # Progress must never overrun: the last emitted step equals the reported total.
    assert states[-1].step == states[-1].total_steps == 8


def test_heun_progress_does_not_overrun_total_steps() -> None:
    # Heun is 2nd order: `set_timesteps(8)` yields 15 timesteps, so the loop runs 15 times.
    # `total_steps` must reflect the real iteration count, otherwise progress runs past 100%.
    states = _run(FlowMatchHeunDiscreteScheduler(), steps=8)

    assert len(states) > 8
    assert all(s.step <= s.total_steps for s in states)
    assert states[-1].step == states[-1].total_steps == len(states)


def test_heun_rejects_a_partial_denoise_range() -> None:
    # Heun's `set_timesteps` takes only a step count, so it cannot honor a custom sigma window.
    # Silently running a full denoise instead would be worse than refusing.
    with pytest.raises(ValueError, match="denoising_start/denoising_end"):
        _run(FlowMatchHeunDiscreteScheduler(), steps=8, denoising_start=0.5)
