import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import DenoiseMaskField, Krea2ConditioningField, LatentsField
from invokeai.app.invocations.krea2_denoise import KREA2_LATENT_CHANNELS, Krea2DenoiseInvocation
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.backend.model_manager.taxonomy import BaseModelType, Krea2VariantType, ModelFormat, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, Krea2ConditioningInfo


@pytest.mark.parametrize(("denoising_start", "denoising_end"), [(0.75, 0.25), (0.5, 0.5)])
def test_validate_inputs_rejects_empty_or_inverted_denoising_range(
    denoising_start: float, denoising_end: float
) -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=denoising_start, denoising_end=denoising_end, denoise_mask=None
    )

    with pytest.raises(ValueError, match="denoising_start must be less than denoising_end"):
        invocation._validate_inputs()


def test_validate_inputs_rejects_denoise_mask_without_latents() -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=0.0,
        denoising_end=1.0,
        denoise_mask=DenoiseMaskField(mask_name="mask"),
        latents=None,
    )

    with pytest.raises(ValueError, match="Initial latents are required when a denoise mask is provided"):
        invocation._validate_inputs()


def test_validate_inputs_accepts_a_valid_configuration() -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=0.0, denoising_end=1.0, denoise_mask=None, latents=None
    )
    # A full-range denoise with no mask is valid and must not raise.
    invocation._validate_inputs()


def _validation_payload() -> dict:
    model = ModelIdentifierField(
        key="krea-model",
        hash="model-hash",
        name="Krea Model",
        base=BaseModelType.Krea2,
        type=ModelType.Main,
    )
    return {
        "transformer": TransformerField(transformer=model, loras=[]),
        "positive_conditioning": Krea2ConditioningField(conditioning_name="positive"),
    }


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("width", -16), ("height", 0), ("height", -16)])
def test_model_validation_rejects_non_positive_dimensions(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        Krea2DenoiseInvocation(**_validation_payload(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cfg_scale", math.nan),
        ("cfg_scale", math.inf),
        ("cfg_scale", [1.0, math.nan]),
        ("shift", math.nan),
        ("shift", math.inf),
    ],
)
def test_model_validation_rejects_non_finite_sampling_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        Krea2DenoiseInvocation(**_validation_payload(), **{field: value})


def test_model_validation_accepts_positive_dimensions_and_finite_sampling_values() -> None:
    invocation = Krea2DenoiseInvocation(**_validation_payload(), width=16, height=32, cfg_scale=[1.0] * 8, shift=1.15)
    assert invocation.width == 16
    assert invocation.height == 32


class TestPrepareCfgScale:
    def test_scalar_is_broadcast_to_the_step_count(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=4.5)
        assert invocation._prepare_cfg_scale(8) == [4.5] * 8

    def test_list_of_matching_length_is_returned_unchanged(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[4.0, 3.0, 2.0])
        assert invocation._prepare_cfg_scale(3) == [4.0, 3.0, 2.0]

    def test_list_of_wrong_length_raises(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[4.0, 3.0, 2.0])
        with pytest.raises(ValueError, match="cfg_scale list has 3 values but the model is configured for 8 steps"):
            invocation._prepare_cfg_scale(8)


class TestCfgForStep:
    def test_scale_above_one_uses_cfg_when_negative_conditioning_is_available(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct()
        assert invocation._should_apply_cfg_for_step(4.0, has_negative_conditioning=True) is True

    @pytest.mark.parametrize("cfg_scale", [1.0, 0.5])
    def test_scale_at_or_below_one_does_not_use_cfg(self, cfg_scale: float) -> None:
        invocation = Krea2DenoiseInvocation.model_construct()
        assert invocation._should_apply_cfg_for_step(cfg_scale, has_negative_conditioning=True) is False

    def test_missing_negative_conditioning_does_not_use_cfg(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct()
        assert invocation._should_apply_cfg_for_step(4.0, has_negative_conditioning=False) is False


class TestEffectiveScheduleValidation:
    def test_rejects_a_fractional_range_that_rounds_to_zero_steps(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct()
        with pytest.raises(ValueError, match="does not contain any effective denoising steps"):
            invocation._validate_effective_schedule(start_idx=0, end_idx=0)

    def test_accepts_a_range_with_at_least_one_effective_step(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct()
        invocation._validate_effective_schedule(start_idx=0, end_idx=1)

    def test_invalid_type_raises(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale="nonsense")
        with pytest.raises(ValueError, match="Invalid CFG scale type"):
            invocation._prepare_cfg_scale(8)


def test_cfg_scale_list_is_built_against_full_step_count_then_clipped() -> None:
    # Regression: a per-step CFG list carries one value per *configured* step. img2img/inpaint clips the
    # sampling schedule (denoising_start/denoising_end), which shrinks the number of active steps. The list
    # must be prepared against the full step count (``total_sigmas``) and *then* sliced to the active
    # window — preparing it against the already-reduced count would reject the user's full-length list.
    steps = 8
    invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[float(i) for i in range(steps)])

    full_cfg_scale = invocation._prepare_cfg_scale(steps)  # built against the FULL step count -> ok
    assert len(full_cfg_scale) == steps

    # Slicing to the active window [start_idx:end_idx] (as _run_diffusion does) keeps the right per-step values.
    start_idx, end_idx = 2, 6
    assert full_cfg_scale[start_idx:end_idx] == [2.0, 3.0, 4.0, 5.0]

    # Preparing against the reduced (clipped) count instead would have raised — the bug this guards against.
    with pytest.raises(ValueError):
        invocation._prepare_cfg_scale(end_idx - start_idx)


def test_get_noise_is_deterministic_and_correctly_shaped() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    device = torch.device("cpu")

    noise_a = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=42)
    noise_b = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=42)
    noise_other = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=43)

    # Shape is (1, latent_channels, H // 8, W // 8).
    assert noise_a.shape == (1, KREA2_LATENT_CHANNELS, 8, 16)
    # Same seed -> identical noise (reproducibility); different seed -> different noise.
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_other)


class _Scheduler:
    def __init__(self, **_kwargs) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, *, sigmas, mu, device) -> None:
        del mu
        self.sigmas = torch.tensor([*sigmas, 0.0], device=device)
        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps


class _Transformer:
    def __init__(self) -> None:
        self.conditioning_values: list[float] = []

    def __call__(self, *, hidden_states, encoder_hidden_states, **_kwargs):
        self.conditioning_values.append(float(encoder_hidden_states.mean()))
        return (torch.zeros_like(hidden_states),)


class _TransformerInfo:
    def __init__(self, transformer: _Transformer) -> None:
        self.transformer = transformer

    @contextmanager
    def model_on_device(self, **_kwargs):
        yield ({}, self.transformer)


def _model_identifier() -> ModelIdentifierField:
    return ModelIdentifierField(
        key="krea-model",
        hash="model-hash",
        name="Krea Model",
        base=BaseModelType.Krea2,
        type=ModelType.Main,
    )


def _runtime_invocation(*, cfg_scale: float | list[float], with_mask: bool = False) -> Krea2DenoiseInvocation:
    return Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=_model_identifier(), loras=[]),
        positive_conditioning=Krea2ConditioningField(conditioning_name="positive"),
        negative_conditioning=Krea2ConditioningField(conditioning_name="negative"),
        cfg_scale=cfg_scale,
        width=16,
        height=16,
        steps=2,
        seed=1,
        shift=1.15,
        denoising_start=0.0,
        denoising_end=1.0,
        latents=LatentsField(latents_name="init") if with_mask else None,
        denoise_mask=DenoiseMaskField(mask_name="mask") if with_mask else None,
    )


def _runtime_context(tmp_path, transformer: _Transformer):
    conditionings = {
        "positive": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 2, 12, 8))]),
        "negative": ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.zeros(1, 2, 12, 8))]
        ),
    }
    tensors = {
        "init": torch.zeros(1, KREA2_LATENT_CHANNELS, 2, 2),
        "mask": torch.zeros(1, 1, 16, 16),
    }
    config = SimpleNamespace(format=ModelFormat.Checkpoint, variant=Krea2VariantType.Turbo)
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: config,
            get_absolute_path=lambda _config: tmp_path,
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda name: tensors[name]),
        util=SimpleNamespace(sd_step_callback=lambda *_args: None),
    )


def _patch_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler", _Scheduler
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_denoise.TorchDevice.choose_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_denoise.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_denoise.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )


def test_run_diffusion_applies_mixed_cfg_only_at_enabled_steps(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()

    latents = _runtime_invocation(cfg_scale=[2.0, 1.0])._run_diffusion(_runtime_context(tmp_path, transformer))

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    assert transformer.conditioning_values == [1.0, 0.0, 1.0]


def test_run_diffusion_reaches_masked_denoising_merge(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    merge_sigmas: list[float] = []

    class _InpaintExtension:
        def __init__(self, **_kwargs) -> None:
            pass

        def merge_intermediate_latents_with_init_latents(self, latents, sigma):
            merge_sigmas.append(sigma)
            return latents

    monkeypatch.setattr("invokeai.app.invocations.krea2_denoise.RectifiedFlowInpaintExtension", _InpaintExtension)

    latents = _runtime_invocation(cfg_scale=1.0, with_mask=True)._run_diffusion(_runtime_context(tmp_path, transformer))

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    assert len(merge_sigmas) == 2
    assert transformer.conditioning_values == [1.0, 1.0]


def test_run_diffusion_uses_per_prompt_position_ids_when_lengths_differ(monkeypatch, tmp_path) -> None:
    # Regression: the positive and negative prompts can tokenize to different lengths. The rotary position
    # ids (text tokens + image grid) must match *each pass's own* text length. Reusing the positive prompt's
    # position ids for the uncond pass leaves the rotary embedding a different length than the negative
    # query sequence, which crashes in the real transformer's apply_rotary_emb.
    _patch_runtime(monkeypatch)

    image_seq_len = 1  # width=height=16 -> 2x2 latent -> a single 2x2 patch

    class _PositionIdChecker:
        def __call__(self, *, hidden_states, encoder_hidden_states, position_ids, **_kwargs):
            text_len = encoder_hidden_states.shape[1]
            pos_len = position_ids.shape[0]
            # The invariant the real Krea2Transformer2DModel enforces: rotary length == text + image tokens.
            assert pos_len == text_len + image_seq_len, (
                f"position_ids length {pos_len} must equal text length {text_len} + image tokens {image_seq_len}"
            )
            return (torch.zeros_like(hidden_states),)

    transformer = _PositionIdChecker()

    # Positive prompt is longer than the negative prompt (3 vs. 2 text tokens).
    conditionings = {
        "positive": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 3, 12, 8))]),
        "negative": ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.zeros(1, 2, 12, 8))]
        ),
    }
    config = SimpleNamespace(format=ModelFormat.Checkpoint, variant=Krea2VariantType.Turbo)
    context = SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: config,
            get_absolute_path=lambda _config: tmp_path,
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda _name: None),
        util=SimpleNamespace(sd_step_callback=lambda *_args: None),
    )

    # cfg_scale > 1 so both the cond (positive) and uncond (negative) passes run each step. Without the fix,
    # the uncond pass receives the positive prompt's position ids and the checker above fails.
    latents = _runtime_invocation(cfg_scale=2.0)._run_diffusion(context)
    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
