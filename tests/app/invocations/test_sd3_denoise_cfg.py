"""CFG handling in SD3DenoiseInvocation._run_diffusion."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import SD3ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, SD3ConditioningInfo

DENOISE = "invokeai.app.invocations.sd3.sd3_denoise"


class _RecordingTransformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=16, joint_attention_dim=8, patch_size=2)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor]:
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "encoder_hidden_states": encoder_hidden_states.detach().clone(),
                "pooled_projections": pooled_projections.detach().clone(),
            }
        )
        fill = pooled_projections[:, 0].view(-1, 1, 1, 1)
        return (fill.expand_as(hidden_states).clone(),)


class _TransformerInfo:
    def __init__(self, transformer: _RecordingTransformer) -> None:
        self.model = transformer

    @contextmanager
    def model_on_device(self, **_kwargs):
        yield ({}, self.model)


def _conditioning(fill: float) -> ConditioningFieldData:
    info = SD3ConditioningInfo(
        clip_l_pooled_embeds=torch.full((1, 4), fill),
        clip_l_embeds=torch.full((1, 2, 4), fill),
        clip_g_pooled_embeds=torch.full((1, 4), fill),
        clip_g_embeds=torch.full((1, 2, 4), fill),
        t5_embeds=torch.full((1, 3, 8), fill),
    )
    return ConditioningFieldData(conditionings=[info])


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_torch_dtype", lambda: torch.float32)
    monkeypatch.setattr(f"{DENOISE}.SD3Transformer2DModel", _RecordingTransformer)


# The SD3.5 shift-3.0 schedule at two steps runs sigma 1 -> 1/112 -> 0 (the GGUF stand-in has no scheduler config).
FIRST_STEP = -111 / 112
SECOND_STEP = -1 / 112


def _run_denoise(
    cfg_scale: float | list[float], steps: int = 2, **fields
) -> tuple[_RecordingTransformer, list[str], torch.Tensor, torch.Tensor]:
    transformer = _RecordingTransformer()
    loaded: list[str] = []

    def _load_cond(name: str) -> ConditioningFieldData:
        loaded.append(name)
        return _conditioning(1.0 if name == "pos" else -1.0)

    context = SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: SimpleNamespace(path="sd3.gguf"),
            get_absolute_path=lambda config: Path(config.path),
        ),
        conditioning=SimpleNamespace(load=_load_cond),
        util=SimpleNamespace(sd_step_callback=lambda *_args: None),
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
        cfg_scale=cfg_scale,
        width=64,
        height=64,
        steps=steps,
        seed=0,
    )
    initial_noise = invocation._get_noise(1, 16, 64, 64, torch.float32, torch.device("cpu"), 0)
    latents = invocation._run_diffusion(context)  # type: ignore[arg-type]
    return transformer, loaded, initial_noise, latents


def test_cfg_scale_one_skips_negative_conditioning_and_unconditional_pass() -> None:
    transformer, loaded, initial_noise, latents = _run_denoise(cfg_scale=1.0, steps=2)
    assert loaded == ["pos"]
    assert [c["hidden_states"].shape[0] for c in transformer.calls] == [1, 1]
    assert all(c["timestep"].shape == (1,) and torch.all(c["pooled_projections"] == 1.0) for c in transformer.calls)
    assert torch.allclose(latents, initial_noise + (FIRST_STEP + SECOND_STEP), atol=1e-5)


def test_cfg_scale_above_one_loads_negative_conditioning_and_batches_uncond_with_cond() -> None:
    transformer, loaded, initial_noise, latents = _run_denoise(cfg_scale=3.5, steps=2)
    assert loaded == ["pos", "neg"]
    assert [c["hidden_states"].shape[0] for c in transformer.calls] == [2, 2]
    assert all(
        torch.all(c["pooled_projections"][0] == -1.0) and torch.all(c["pooled_projections"][1] == 1.0)
        for c in transformer.calls
    )
    assert torch.allclose(latents, initial_noise + 6.0 * (FIRST_STEP + SECOND_STEP), atol=1e-5)


def test_per_step_cfg_scale_skips_unconditional_pass_only_on_cfg_one_steps() -> None:
    transformer, loaded, initial_noise, latents = _run_denoise(cfg_scale=[3.5, 1.0], steps=2)
    assert loaded == ["pos", "neg"]
    assert [c["hidden_states"].shape[0] for c in transformer.calls] == [2, 1]
    assert [c["encoder_hidden_states"].shape[0] for c in transformer.calls] == [2, 1]
    assert torch.all(transformer.calls[1]["pooled_projections"] == 1.0)
    # The guided step predicts -1 + 3.5 * 2 = 6; the CFG 1 step predicts the positive prompt's 1.
    assert torch.allclose(latents, initial_noise + 6.0 * FIRST_STEP + 1.0 * SECOND_STEP, atol=1e-5)
