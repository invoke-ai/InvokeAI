"""The FLUX.3 sampling loop over a stand-in transformer that records each call.

What is asserted is what the loop owes the model: the flow-map target is the next sigma, CFG combines the two
predictions the reference way, references go into every forward unstepped, and the Euler update and preview
are computed from the prediction.
"""

import pytest
import torch

from invokeai.backend.flux3.denoise import denoise


class _Recorder(torch.nn.Module):
    """Predicts a velocity of `scale` (pos) or `neg_scale` (neg text), and records every call."""

    def __init__(self, neg_text: torch.Tensor | None = None):
        super().__init__()
        self.calls: list[dict] = []
        self.neg_text = neg_text

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        is_neg = self.neg_text is not None and kwargs["encoder_hidden_states"] is self.neg_text
        return (torch.full_like(kwargs["hidden_states"], -1.0 if is_neg else 1.0),)


def _run(model, schedule, **kwargs):
    img = torch.zeros(1, 4, 8)
    previews = []
    out = denoise(
        model=model,
        img=img,
        img_ids=torch.zeros(4, 4),
        txt=kwargs.pop("txt", torch.zeros(1, 2, 3)),
        txt_ids=torch.zeros(2, 4),
        timesteps=schedule,
        step_callback=lambda state: previews.append(state.latents.clone()),
        compute_dtype=torch.float32,
        **kwargs,
    )
    return out, previews


def test_the_flow_map_is_told_the_timestep_each_step_jumps_to() -> None:
    model = _Recorder()
    schedule = [1.0, 0.75, 0.3, 0.0]

    out, previews = _run(model, schedule, flow_map=True)

    targets = [call["target_timestep"].item() for call in model.calls]
    timesteps = [call["timestep"].item() for call in model.calls]
    assert timesteps == pytest.approx(schedule[:-1]) and targets == pytest.approx(schedule[1:])
    # Euler on a constant velocity of 1 from 0: x = sum(dt) = -1.
    assert torch.allclose(out, torch.full_like(out, -1.0))
    assert len(previews) == 3


def test_without_a_flow_map_no_target_is_passed() -> None:
    model = _Recorder()
    _run(model, [1.0, 0.5, 0.0], flow_map=False)
    assert all(call["target_timestep"] is None for call in model.calls)


def test_cfg_is_uncond_plus_scale_times_the_difference() -> None:
    neg = torch.zeros(1, 2, 3)
    model = _Recorder(neg_text=neg)

    out, _ = _run(model, [1.0, 0.0], flow_map=False, cfg_scale=3.5, neg_txt=neg, neg_txt_ids=torch.zeros(2, 4))

    # velocity = -1 + 3.5 * (1 - -1) = 6, one step of dt = -1.
    assert len(model.calls) == 2
    assert torch.allclose(out, torch.full_like(out, -6.0))


def test_a_flow_map_refuses_cfg() -> None:
    with pytest.raises(ValueError, match="flow-map"):
        _run(_Recorder(), [1.0, 0.0], flow_map=True, cfg_scale=2.0, neg_txt=torch.zeros(1, 2, 3))


def test_references_go_into_every_forward_unchanged() -> None:
    model = _Recorder()
    ref_tokens, ref_ids = torch.randn(1, 3, 8), torch.zeros(3, 4)

    _run(model, [1.0, 0.5, 0.0], flow_map=True, ref_tokens=ref_tokens, ref_ids=ref_ids)

    assert all(call["image_cond_hidden_states"] is ref_tokens for call in model.calls)
    assert all(call["image_cond_ids"] is ref_ids for call in model.calls)
