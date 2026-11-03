"""`denoise()`'s `before_step` seam: which user step each forward pass belongs to.

A per-step setting -- a ControlNet's begin/end range -- is set from this call, so every forward of a step, the
negative one and Heun's second-order one included, has to see the index of the step it belongs to.
"""

from typing import Any

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler

from invokeai.backend.flux2.denoise import denoise

TOKENS = 4
CHANNELS = 8
TIMESTEPS = [1.0, 0.75, 0.5, 0.25, 0.0]  # four user steps


class StepRecordingModel:
    """Records, per forward pass, the step `before_step` last announced, and predicts a constant velocity."""

    def __init__(self) -> None:
        self.current_step: int | None = None
        self.steps_seen: list[int | None] = []

    def announce(self, step_index: int) -> None:
        self.current_step = step_index

    def __call__(self, hidden_states: torch.Tensor, **_kwargs: Any) -> tuple[torch.Tensor]:
        self.steps_seen.append(self.current_step)
        return (torch.full_like(hidden_states, 0.5),)


def _run(model: StepRecordingModel, scheduler: Any, cfg_scale: float) -> None:
    generator = torch.Generator().manual_seed(0)
    denoise(
        model=model,  # type: ignore[arg-type]
        img=torch.randn(1, TOKENS, CHANNELS, generator=generator),
        img_ids=torch.zeros(1, TOKENS, 4, dtype=torch.long),
        txt=torch.randn(1, 3, CHANNELS, generator=generator),
        txt_ids=torch.zeros(1, 3, 4, dtype=torch.long),
        timesteps=TIMESTEPS,
        step_callback=lambda _state: None,
        guidance=1.0,
        cfg_scale=[cfg_scale] * (len(TIMESTEPS) - 1),
        neg_txt=torch.zeros(1, 3, CHANNELS),
        neg_txt_ids=torch.zeros(1, 3, 4, dtype=torch.long),
        scheduler=scheduler,
        before_step=model.announce,
    )


@pytest.mark.parametrize(
    "scheduler",
    [None, FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)],
    ids=["manual-euler", "scheduler-euler"],
)
def test_both_guidance_forwards_of_a_step_see_its_index(scheduler) -> None:
    model = StepRecordingModel()

    _run(model, scheduler, cfg_scale=2.0)

    assert model.steps_seen == [0, 0, 1, 1, 2, 2, 3, 3]


def test_heuns_two_forwards_of_a_step_share_its_index() -> None:
    """Heun evaluates the model at a step's start and at its end, except on the last step to sigma 0, which it takes
    as a single Euler evaluation: seven forwards for four steps, numbered by the step they advance."""
    model = StepRecordingModel()

    _run(model, FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=3.0), cfg_scale=1.0)

    assert model.steps_seen == [0, 0, 1, 1, 2, 2, 3]
