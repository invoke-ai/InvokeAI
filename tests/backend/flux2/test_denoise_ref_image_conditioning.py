"""Regression tests for FLUX.2 reference image conditioning during denoising.

Reference image latents are *context*: they are concatenated onto the latents for every forward
pass, but the sampler must never advance them. A previous implementation concatenated them onto
the sampled tensor once before the loop, so every step integrated the reference tokens along the
model's velocity field - the reference silently drifted away from the encoded image over the
schedule and dragged the generated image with it (visible as a reproducible spatial shift of the
edited result).
"""

from typing import Any

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.backend.flux2.denoise import denoise

BATCH = 1
GEN_SEQ_LEN = 6
REF_SEQ_LEN = 4
CHANNELS = 8
TXT_SEQ_LEN = 3


class RecordingModel:
    """A stand-in transformer that records its inputs and predicts a constant velocity.

    A constant non-zero prediction is enough to expose sampler updates leaking into the reference
    tokens: any integration of the reference part changes it away from the encoded latents.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor]:
        self.calls.append({"hidden_states": hidden_states.clone(), "img_ids": img_ids.clone()})
        return (torch.full_like(hidden_states, 0.5),)


def _inputs() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    return {
        "img": torch.randn(BATCH, GEN_SEQ_LEN, CHANNELS, generator=generator),
        "img_ids": torch.zeros(BATCH, GEN_SEQ_LEN, 4, dtype=torch.long),
        "txt": torch.randn(BATCH, TXT_SEQ_LEN, CHANNELS, generator=generator),
        "txt_ids": torch.zeros(BATCH, TXT_SEQ_LEN, 4, dtype=torch.long),
        "img_cond_seq": torch.randn(BATCH, REF_SEQ_LEN, CHANNELS, generator=generator),
        "img_cond_seq_ids": torch.full((BATCH, REF_SEQ_LEN, 4), 10, dtype=torch.long),
    }


def _scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)


@pytest.mark.parametrize("use_scheduler", [False, True], ids=["euler", "scheduler"])
def test_reference_conditioning_is_constant_across_steps(use_scheduler: bool):
    inputs = _inputs()
    model = RecordingModel()
    ref_latents = inputs["img_cond_seq"].clone()

    out = denoise(
        model=model,  # type: ignore[arg-type]
        img=inputs["img"],
        img_ids=inputs["img_ids"],
        txt=inputs["txt"],
        txt_ids=inputs["txt_ids"],
        timesteps=[1.0, 0.75, 0.5, 0.25, 0.0],
        step_callback=lambda state: None,
        guidance=1.0,
        cfg_scale=[1.0] * 4,
        scheduler=_scheduler() if use_scheduler else None,
        img_cond_seq=inputs["img_cond_seq"],
        img_cond_seq_ids=inputs["img_cond_seq_ids"],
    )

    assert len(model.calls) == 4

    expected_ids = torch.cat([inputs["img_ids"], inputs["img_cond_seq_ids"]], dim=1)
    for step, call in enumerate(model.calls):
        hidden_states = call["hidden_states"]
        assert hidden_states.shape == (BATCH, GEN_SEQ_LEN + REF_SEQ_LEN, CHANNELS)
        # The reference tokens must be bit-identical to the encoded latents at every step.
        assert torch.equal(hidden_states[:, GEN_SEQ_LEN:, :], ref_latents), f"reference drifted at step {step}"
        assert torch.equal(call["img_ids"], expected_ids)

    # The caller's tensor must not be mutated either.
    assert torch.equal(inputs["img_cond_seq"], ref_latents)

    # Only the generated tokens are returned, and they were actually denoised.
    assert out.shape == (BATCH, GEN_SEQ_LEN, CHANNELS)
    assert not torch.equal(out, inputs["img"])


@pytest.mark.parametrize("use_scheduler", [False, True], ids=["euler", "scheduler"])
def test_reference_conditioning_with_cfg(use_scheduler: bool):
    """With CFG the negative prediction must be sliced to the generated tokens as well."""
    inputs = _inputs()
    model = RecordingModel()
    ref_latents = inputs["img_cond_seq"].clone()

    out = denoise(
        model=model,  # type: ignore[arg-type]
        img=inputs["img"],
        img_ids=inputs["img_ids"],
        txt=inputs["txt"],
        txt_ids=inputs["txt_ids"],
        timesteps=[1.0, 0.5, 0.0],
        step_callback=lambda state: None,
        guidance=1.0,
        cfg_scale=[2.0] * 2,
        neg_txt=torch.zeros(BATCH, TXT_SEQ_LEN, CHANNELS),
        neg_txt_ids=torch.zeros(BATCH, TXT_SEQ_LEN, 4, dtype=torch.long),
        scheduler=_scheduler() if use_scheduler else None,
        img_cond_seq=inputs["img_cond_seq"],
        img_cond_seq_ids=inputs["img_cond_seq_ids"],
    )

    # One positive and one negative forward pass per step.
    assert len(model.calls) == 4
    for call in model.calls:
        assert torch.equal(call["hidden_states"][:, GEN_SEQ_LEN:, :], ref_latents)

    assert out.shape == (BATCH, GEN_SEQ_LEN, CHANNELS)


@pytest.mark.parametrize("use_scheduler", [False, True], ids=["euler", "scheduler"])
def test_without_reference_conditioning(use_scheduler: bool):
    inputs = _inputs()
    model = RecordingModel()

    out = denoise(
        model=model,  # type: ignore[arg-type]
        img=inputs["img"],
        img_ids=inputs["img_ids"],
        txt=inputs["txt"],
        txt_ids=inputs["txt_ids"],
        timesteps=[1.0, 0.5, 0.0],
        step_callback=lambda state: None,
        guidance=1.0,
        cfg_scale=[1.0] * 2,
        scheduler=_scheduler() if use_scheduler else None,
    )

    for call in model.calls:
        assert call["hidden_states"].shape == (BATCH, GEN_SEQ_LEN, CHANNELS)
        assert torch.equal(call["img_ids"], inputs["img_ids"])

    assert out.shape == (BATCH, GEN_SEQ_LEN, CHANNELS)
