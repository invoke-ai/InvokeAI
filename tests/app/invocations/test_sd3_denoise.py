"""The SD3 denoise node, run against stand-ins for the transformer and the context."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import SD3ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, SD3ConditioningInfo

DENOISE = "invokeai.app.invocations.sd3.sd3_denoise"


class _Transformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=16, joint_attention_dim=8)

    def forward(self, *, hidden_states: torch.Tensor, **_kwargs) -> tuple[torch.Tensor]:
        return (torch.zeros_like(hidden_states),)


class _TransformerInfo:
    def __init__(self, events: list[str]) -> None:
        self.model = _Transformer()
        self.events = events

    @contextmanager
    def model_on_device(self, **_kwargs):
        self.events.append("lock")
        yield ({}, self.model)
        self.events.append("unlock")


def _conditioning() -> ConditioningFieldData:
    info = SD3ConditioningInfo(
        clip_l_pooled_embeds=torch.zeros(1, 4),
        clip_l_embeds=torch.zeros(1, 2, 4),
        clip_g_pooled_embeds=torch.zeros(1, 4),
        clip_g_embeds=torch.zeros(1, 2, 4),
        t5_embeds=torch.zeros(1, 3, 8),
    )
    return ConditioningFieldData(conditionings=[info])


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_torch_dtype", lambda: torch.float32)
    monkeypatch.setattr(f"{DENOISE}.SD3Transformer2DModel", _Transformer)


def test_every_preview_is_sent_while_the_transformer_is_locked() -> None:
    """A Tiny AutoEncoder preview places its TAE next to the locked model; before the lock there is none to fit."""
    events: list[str] = []
    context = SimpleNamespace(
        models=SimpleNamespace(load=lambda _identifier: _TransformerInfo(events)),
        conditioning=SimpleNamespace(load=lambda _name: _conditioning()),
        util=SimpleNamespace(sd_step_callback=lambda *_args: events.append("preview")),
    )
    model = ModelIdentifierField(key="m", hash="h", name="m", base=BaseModelType.StableDiffusion3, type=ModelType.Main)
    invocation = SD3DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=model, loras=[]),
        positive_conditioning=SD3ConditioningField(conditioning_name="pos"),
        negative_conditioning=SD3ConditioningField(conditioning_name="neg"),
        latents=None,
        noise=None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        cfg_scale=3.5,
        width=64,
        height=64,
        steps=3,
        seed=0,
    )

    invocation._run_diffusion(context)  # type: ignore[arg-type]

    assert events == ["lock", "preview", "preview", "preview", "preview", "unlock"]
