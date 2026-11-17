"""The SD3 denoise node, run against stand-ins for the transformer and the context."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import LatentsField, SD3ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, SD3ConditioningInfo

DENOISE = "invokeai.app.invocations.sd3.sd3_denoise"


class _Transformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=16, joint_attention_dim=8)

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[float] = []

    def forward(self, *, hidden_states: torch.Tensor, timestep: torch.Tensor, **_kwargs) -> tuple[torch.Tensor]:
        self.timesteps.append(timestep[0].item())
        return (torch.zeros_like(hidden_states),)


class _TransformerInfo:
    def __init__(self, events: list[str] | None = None) -> None:
        self.model = _Transformer()
        self.events = events if events is not None else []

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


def _context(transformer_info: _TransformerInfo, model_path: Path, **namespaces: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: transformer_info,
            get_config=lambda _identifier: SimpleNamespace(path=str(model_path)),
            get_absolute_path=lambda config: Path(config.path),
        ),
        conditioning=SimpleNamespace(load=lambda _name: _conditioning()),
        util=SimpleNamespace(sd_step_callback=lambda *_args: transformer_info.events.append("preview")),
        **namespaces,
    )


def _invocation(**fields) -> SD3DenoiseInvocation:
    model = ModelIdentifierField(key="m", hash="h", name="m", base=BaseModelType.StableDiffusion3, type=ModelType.Main)
    defaults = {
        "transformer": TransformerField(transformer=model, loras=[]),
        "positive_conditioning": SD3ConditioningField(conditioning_name="pos"),
        "negative_conditioning": SD3ConditioningField(conditioning_name="neg"),
        "latents": None,
        "noise": None,
        "denoise_mask": None,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
        "cfg_scale": 3.5,
        "width": 64,
        "height": 64,
        "steps": 3,
        "seed": 0,
    }
    return SD3DenoiseInvocation.model_construct(**(defaults | fields))


def test_every_preview_is_sent_while_the_transformer_is_locked(tmp_path: Path) -> None:
    """A Tiny AutoEncoder preview places its TAE next to the locked model; before the lock there is none to fit."""
    events: list[str] = []
    context = _context(_TransformerInfo(events), tmp_path / "sd3.gguf")

    _invocation()._run_diffusion(context)  # type: ignore[arg-type]

    assert events == ["lock", "preview", "preview", "preview", "preview", "unlock"]


# Stability's SD3.5 sampling at shift 3.0 and 4 steps: timesteps from 1000 to the shifted first training timestep,
# evenly spaced, each shifted by `3t / (1 + 2t)`.
SHIFTED_4_STEPS = [1000.0, 857.69, 602.15, 8.93]


def test_a_transformer_without_a_scheduler_config_samples_with_the_sd35_shift(tmp_path: Path) -> None:
    """A GGUF build carries no scheduler, so it gets the shift every SD3 release ships."""
    transformer_info = _TransformerInfo()

    _invocation(steps=4)._run_diffusion(_context(transformer_info, tmp_path / "sd3.5_large_turbo-Q4_0.gguf"))  # type: ignore[arg-type]

    assert transformer_info.model.timesteps == pytest.approx(SHIFTED_4_STEPS, abs=0.01)


def test_a_diffusers_model_samples_with_its_own_scheduler_config(tmp_path: Path) -> None:
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 1.0})
    )
    transformer_info = _TransformerInfo()

    _invocation(steps=4)._run_diffusion(_context(transformer_info, tmp_path))  # type: ignore[arg-type]

    assert transformer_info.model.timesteps == pytest.approx([1000.0, 667.0, 334.0, 1.0], abs=0.01)


def test_image_to_image_starts_at_the_noise_level_denoising_start_names(tmp_path: Path) -> None:
    """The shifted schedule is clipped, not the unshifted one: a pass at denoising_start 0.3 starts at sigma 0.7 and
    keeps the shifted steps below it."""
    transformer_info = _TransformerInfo()
    tensors = {"init": torch.ones(1, 16, 8, 8), "noise": torch.full((1, 16, 8, 8), 10.0)}
    context = _context(transformer_info, tmp_path / "sd3.gguf", tensors=SimpleNamespace(load=tensors.__getitem__))
    invocation = _invocation(
        steps=4,
        denoising_start=0.3,
        latents=LatentsField(latents_name="init"),
        noise=LatentsField(latents_name="noise"),
    )

    result = invocation._run_diffusion(context)  # type: ignore[arg-type]

    assert transformer_info.model.timesteps == pytest.approx([700.0, *SHIFTED_4_STEPS[2:]], abs=0.01)
    # The stand-in predicts zero velocity, so the result is the input as noised for the first step.
    assert torch.allclose(result, torch.full((1, 16, 8, 8), 0.7 * 10.0 + 0.3 * 1.0))
