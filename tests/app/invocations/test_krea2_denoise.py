import dataclasses
import json
import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    Krea2ConditioningField,
    Krea2StyleReferenceField,
    LatentsField,
    TensorField,
)
from invokeai.app.invocations.krea2.krea2_denoise import (
    KREA2_LATENT_CHANNELS,
    Krea2DenoiseInvocation,
    requires_sidecar_patching,
)
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.backend.krea2 import denoise_working_memory as _ESTIMATOR_MODULE
from invokeai.backend.krea2.sampling_utils import build_sigmas, pack_latents, unpack_latents
from invokeai.backend.krea2.style_reference import Krea2StyleReferenceMode, capture_style_reference
from invokeai.backend.model_manager.taxonomy import BaseModelType, Krea2VariantType, ModelFormat, ModelType
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
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
    invocation = Krea2DenoiseInvocation.model_construct(
        cfg_scale=[float(i) for i in range(steps)], denoising_start=0.25, denoising_end=0.75
    )

    # Steps 2..5 of 8 run, each with its own configured value.
    assert invocation._active_cfg_scale(steps) == [2.0, 3.0, 4.0, 5.0]

    # Preparing against the reduced (clipped) count instead would have raised — the bug this guards against.
    with pytest.raises(ValueError):
        invocation._prepare_cfg_scale(4)


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


def test_load_text_conditioning_concatenates_only_valid_tokens() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    first_embeds = torch.stack(
        [
            torch.full((12, 8), 1.0),
            torch.full((12, 8), 99.0),
            torch.full((12, 8), 3.0),
        ],
        dim=0,
    ).unsqueeze(0)
    conditionings = {
        "first": ConditioningFieldData(
            conditionings=[
                Krea2ConditioningInfo(
                    prompt_embeds=first_embeds,
                    prompt_embeds_mask=torch.tensor([[True, False, True]]),
                )
            ]
        ),
        "second": ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.full((1, 3, 12, 8), 2.0))]
        ),
    }
    regional_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    context = SimpleNamespace(
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda name: regional_mask if name == "region-mask" else None),
    )

    extension = invocation._load_text_conditioning(
        context=context,
        conditioning_field=[
            Krea2ConditioningField(conditioning_name="first", mask=TensorField(tensor_name="region-mask")),
            Krea2ConditioningField(conditioning_name="second"),
        ],
        grid_height=1,
        grid_width=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prompt_embeds = extension.regional_text_conditioning.prompt_embeds

    assert prompt_embeds.shape == (1, 5, 12, 8)
    assert torch.equal(prompt_embeds[:, 0], torch.full((1, 12, 8), 1.0))
    assert torch.equal(prompt_embeds[:, 1], torch.full((1, 12, 8), 3.0))
    assert torch.equal(prompt_embeds[:, 2:], torch.full((1, 3, 12, 8), 2.0))
    assert torch.equal(
        extension.regional_text_conditioning.image_masks[0],
        torch.tensor([[[1.0, 0.0]]]),
    )
    assert extension.regional_text_conditioning.image_masks[1] is None


def test_load_text_conditioning_compacts_a_single_masked_conditioning() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    embeds = torch.arange(4 * 12 * 8, dtype=torch.float32).reshape(1, 4, 12, 8)
    conditionings = {
        "prompt": ConditioningFieldData(
            conditionings=[
                Krea2ConditioningInfo(
                    prompt_embeds=embeds,
                    prompt_embeds_mask=torch.tensor([[True, False, True, False]]),
                )
            ]
        )
    }
    context = SimpleNamespace(
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda _name: None),
    )

    extension = invocation._load_text_conditioning(
        context=context,
        conditioning_field=Krea2ConditioningField(conditioning_name="prompt"),
        grid_height=1,
        grid_width=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prompt_embeds = extension.regional_text_conditioning.prompt_embeds

    assert torch.equal(prompt_embeds, embeds[:, [0, 2]])
    assert extension.get_attention_mask() is None


def test_load_text_conditioning_preserves_none_when_all_masks_are_none() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    conditionings = {
        "first": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 2, 12, 8))]),
        "second": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 3, 12, 8))]),
    }
    context = SimpleNamespace(
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda _name: None),
    )

    extension = invocation._load_text_conditioning(
        context=context,
        conditioning_field=[
            Krea2ConditioningField(conditioning_name="first"),
            Krea2ConditioningField(conditioning_name="second"),
        ],
        grid_height=1,
        grid_width=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    prompt_embeds = extension.regional_text_conditioning.prompt_embeds
    assert prompt_embeds.shape == (1, 5, 12, 8)
    assert extension.get_attention_mask() is None


def test_load_text_conditioning_rejects_an_empty_collection() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    context = SimpleNamespace(conditioning=SimpleNamespace(load=lambda _name: None))

    with pytest.raises(ValueError, match="At least one Krea-2 conditioning is required"):
        invocation._load_text_conditioning(
            context=context,
            conditioning_field=[],
            grid_height=1,
            grid_width=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


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
        self.regional_attention_masks: list[torch.Tensor | None] = []
        self.combined_sequence_lengths: list[int] = []
        self.style_modes: list[object | None] = []
        self.position_id_lengths: list[int] = []
        self.attn_processors = {
            "text_fusion.layerwise_blocks.0.attn.processor": object(),
            "transformer_blocks.0.attn.processor": object(),
            "transformer_blocks.1.attn.processor": object(),
        }
        self.installed_processors = None
        # The working-memory estimate reads the block layers' streamed copies and transients here, and the
        # attention geometry from the config.
        self.transformer_blocks = torch.nn.ModuleList()
        self.config = KREA2_CONFIG

    def set_attn_processor(self, processor) -> None:
        # The real Krea2Transformer2DModel exposes this; denoise swaps in a memory-efficient attention processor.
        self.installed_processors = processor

    def modules(self):
        # Denoise walks the module tree to decide whether LoRA has to be applied as a sidecar: an
        # int8_tensorwise build's Linears hold their weights as buffers, which a direct patch
        # cannot write into. This double stands in for an unquantized model.
        return iter(())

    def __call__(self, *, hidden_states, encoder_hidden_states, **_kwargs):
        self.conditioning_values.append(float(encoder_hidden_states.mean()))
        # The real transformer concatenates [text, image] before attention, so this is the sequence length a
        # regional mask has to match.
        self.combined_sequence_lengths.append(encoder_hidden_states.shape[1] + hidden_states.shape[1])
        processor = self.installed_processors["transformer_blocks.0.attn.processor"]
        attention_mask = processor.regional_prompting_state.attention_mask
        self.regional_attention_masks.append(attention_mask.clone() if attention_mask is not None else None)
        style_state = processor.style_reference_state
        self.style_modes.append(style_state.mode if style_state is not None else None)
        if style_state is not None and style_state.mode is Krea2StyleReferenceMode.CAPTURE:
            # Stand in for the styled block: the real processor stashes the reference's image-token K/V
            # here, and the injection pass refuses to run without it.
            sequence_length = encoder_hidden_states.shape[1] + hidden_states.shape[1]
            captured = torch.zeros(1, 2, sequence_length, 8)
            capture_style_reference(style_state, 0, captured, captured, captured)
        if "position_ids" in _kwargs and _kwargs["position_ids"] is not None:
            self.position_id_lengths.append(int(_kwargs["position_ids"].shape[0]))
        return (torch.zeros_like(hidden_states),)


KREA2_CONFIG = SimpleNamespace(num_attention_heads=48, num_key_value_heads=12, attention_head_dim=128)
"""The attention geometry of the Krea-2 transformer's config."""


class _TransformerInfo:
    def __init__(self, transformer: _Transformer) -> None:
        self.transformer = transformer

    @property
    def model(self):
        """`LoadedModel.model` -- the model without locking it.

        Several denoise nodes read it to size their working-memory reservation, which has to happen
        before `model_on_device` is entered.
        """
        return self.transformer

    @contextmanager
    def model_on_device(self, **kwargs):
        self.working_memory = kwargs["working_memory"]
        yield ({}, self.transformer)


def _model_identifier() -> ModelIdentifierField:
    return ModelIdentifierField(
        key="krea-model",
        hash="model-hash",
        name="Krea Model",
        base=BaseModelType.Krea2,
        type=ModelType.Main,
    )


def _runtime_invocation(
    *,
    cfg_scale: float | list[float],
    with_mask: bool = False,
    negative_conditioning: Krea2ConditioningField | list[Krea2ConditioningField] | None = None,
) -> Krea2DenoiseInvocation:
    return Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=_model_identifier(), loras=[]),
        positive_conditioning=Krea2ConditioningField(conditioning_name="positive"),
        negative_conditioning=negative_conditioning
        if negative_conditioning is not None
        else Krea2ConditioningField(conditioning_name="negative"),
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


def _runtime_context(tmp_path, transformer: _Transformer, *, negative_text_seq_len: int = 2):
    conditionings = {
        "positive": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 2, 12, 8))]),
        "negative": ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.zeros(1, negative_text_seq_len, 12, 8))]
        ),
        "style-prompt": ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.full((1, 5, 12, 8), 0.5))]
        ),
    }
    tensors = {
        "init": torch.zeros(1, KREA2_LATENT_CHANNELS, 2, 2),
        "mask": torch.zeros(1, 1, 16, 16),
        "positive-region": torch.ones(1, 16, 16),
        "negative-region": torch.zeros(1, 16, 16),
        # width=height=16 -> a 2x2 latent; the Qwen-Image VAE emits a frame dim.
        "style-ref": torch.zeros(1, KREA2_LATENT_CHANNELS, 1, 2, 2),
        "style-ref-wrong-size": torch.zeros(1, KREA2_LATENT_CHANNELS, 1, 4, 4),
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
        "invokeai.app.invocations.krea2.krea2_denoise.TorchDevice.choose_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2.krea2_denoise.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2.krea2_denoise.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )


@pytest.mark.parametrize(
    ("start_step", "has_initial_latents", "with_mask"),
    [(0, False, False), (2, True, False), (2, True, True)],
)
def test_run_diffusion_keeps_fp32_sampler_state_with_bf16_model_input(
    monkeypatch, tmp_path, start_step: int, has_initial_latents: bool, with_mask: bool
) -> None:
    """The Euler state stays fp32; only the transformer input is cast.

    Round-tripping the latents through bf16 on every step discards low-order bits, which shows up as
    grain. This pins the exact arithmetic: bf16 in to the model, fp32 accumulation out.
    """
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2.krea2_denoise.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.bfloat16,
    )

    class _Bf16Transformer(_Transformer):
        def __init__(self) -> None:
            super().__init__()
            self.model_inputs: list[torch.Tensor] = []

        def __call__(self, *, hidden_states, **kwargs):
            self.model_inputs.append(hidden_states.detach().clone())
            super().__call__(hidden_states=hidden_states, **kwargs)
            prediction = hidden_states * 0.25 + torch.tensor(
                0.1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            return (prediction,)

    transformer = _Bf16Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0, with_mask=with_mask)
    invocation.steps = 8
    invocation.denoising_start = start_step / invocation.steps
    context = _runtime_context(tmp_path, transformer)
    initial_latents = torch.linspace(-0.4, 0.5, KREA2_LATENT_CHANNELS * 4, dtype=torch.float32).reshape(
        1, KREA2_LATENT_CHANNELS, 2, 2
    )
    if has_initial_latents:
        invocation.latents = LatentsField(latents_name="init")
        tensors = {"init": initial_latents, "mask": torch.zeros(1, 1, 16, 16)}
        context.tensors.load = lambda name: tensors[name]
    intermediate_states = []
    context.util.sd_step_callback = lambda state, _base: intermediate_states.append(state)

    inpaint_constructor_dtypes: list[torch.dtype] = []
    inpaint_merge_input_dtypes: list[torch.dtype] = []
    inpaint_merge_output_dtypes: list[torch.dtype] = []
    if with_mask:

        class _TrackingInpaintExtension:
            def __init__(self, *, init_latents, inpaint_mask, noise) -> None:
                inpaint_constructor_dtypes.extend([init_latents.dtype, inpaint_mask.dtype, noise.dtype])
                self._extension = RectifiedFlowInpaintExtension(init_latents, inpaint_mask, noise)

            def merge_intermediate_latents_with_init_latents(self, latents, sigma):
                inpaint_merge_input_dtypes.append(latents.dtype)
                merged = self._extension.merge_intermediate_latents_with_init_latents(latents, sigma)
                inpaint_merge_output_dtypes.append(merged.dtype)
                return merged

        monkeypatch.setattr(
            "invokeai.app.invocations.krea2.krea2_denoise.RectifiedFlowInpaintExtension", _TrackingInpaintExtension
        )

    latents = invocation._run_diffusion(context)

    noise = invocation._get_noise(16, 16, torch.float32, torch.device("cpu"), seed=invocation.seed)
    sigmas = torch.tensor([*build_sigmas(invocation.steps), 0.0], dtype=torch.float32)[start_step:]
    active_steps = len(sigmas) - 1
    if has_initial_latents:
        initial_sigma = sigmas[0].item()
        expected_state = initial_sigma * noise + (1.0 - initial_sigma) * initial_latents
    else:
        expected_state = noise

    assert torch.equal(intermediate_states[0].latents, expected_state)
    expected_packed_state = pack_latents(expected_state, 1, KREA2_LATENT_CHANNELS, 2, 2)
    for step_idx in range(active_steps):
        expected_model_input = expected_packed_state.to(torch.bfloat16)
        assert torch.equal(transformer.model_inputs[step_idx], expected_model_input)

        model_prediction = expected_model_input * 0.25 + torch.tensor(0.1, dtype=torch.bfloat16)
        dt = sigmas[step_idx + 1] - sigmas[step_idx]
        expected_packed_state = expected_packed_state + dt * model_prediction.to(torch.float32)
        expected_state = unpack_latents(expected_packed_state, 2, 2)
        assert torch.allclose(intermediate_states[step_idx + 1].latents, expected_state, atol=1e-7, rtol=0)

    assert latents.dtype == torch.float32
    assert torch.allclose(latents.squeeze(2), expected_state, atol=1e-7, rtol=0)
    assert len(transformer.model_inputs) == active_steps
    assert len(intermediate_states) == active_steps + 1
    if with_mask:
        assert inpaint_constructor_dtypes == [torch.float32, torch.float32, torch.float32]
        assert inpaint_merge_input_dtypes == [torch.float32] * active_steps
        assert inpaint_merge_output_dtypes == [torch.float32] * active_steps


def test_run_diffusion_applies_mixed_cfg_only_at_enabled_steps(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()

    latents = _runtime_invocation(cfg_scale=[2.0, 1.0])._run_diffusion(_runtime_context(tmp_path, transformer))

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    assert transformer.conditioning_values == [1.0, 0.0, 1.0]


def test_run_diffusion_treats_an_empty_negative_collection_as_absent(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()

    latents = _runtime_invocation(
        cfg_scale=2.0,
        negative_conditioning=[],
    )._run_diffusion(_runtime_context(tmp_path, transformer))

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    assert transformer.conditioning_values == [1.0, 1.0]


def test_run_diffusion_switches_regional_masks_between_equal_length_cfg_conditionings(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=2.0)
    invocation.positive_conditioning = Krea2ConditioningField(
        conditioning_name="positive", mask=TensorField(tensor_name="positive-region")
    )
    invocation.negative_conditioning = Krea2ConditioningField(
        conditioning_name="negative", mask=TensorField(tensor_name="negative-region")
    )

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    positive_mask, negative_mask, positive_mask_second_step, negative_mask_second_step = (
        transformer.regional_attention_masks
    )
    assert positive_mask is not None
    assert negative_mask is not None
    assert not torch.equal(positive_mask, negative_mask)
    assert torch.equal(positive_mask, positive_mask_second_step)
    assert torch.equal(negative_mask, negative_mask_second_step)
    regional_processor = transformer.installed_processors["transformer_blocks.0.attn.processor"]
    assert regional_processor.regional_prompting_state.attention_mask is None


def test_run_diffusion_sizes_regional_masks_per_pass_when_cfg_lengths_differ(monkeypatch, tmp_path) -> None:
    # The positive and negative prompts tokenize independently, so each pass needs a mask sized for its own
    # text sequence. A mask carried over from the other pass would trip the processor's shape guard.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=2.0)
    invocation.positive_conditioning = Krea2ConditioningField(
        conditioning_name="positive", mask=TensorField(tensor_name="positive-region")
    )
    invocation.negative_conditioning = Krea2ConditioningField(
        conditioning_name="negative", mask=TensorField(tensor_name="negative-region")
    )

    invocation._run_diffusion(_runtime_context(tmp_path, transformer, negative_text_seq_len=5))

    # width=height=16 -> 2x2 latent -> a single 2x2 patch, so image_seq_len is 1.
    assert transformer.combined_sequence_lengths == [3, 6, 3, 6]
    assert [tuple(mask.shape) for mask in transformer.regional_attention_masks] == [(3, 3), (6, 6), (3, 3), (6, 6)]
    for mask, sequence_length in zip(
        transformer.regional_attention_masks, transformer.combined_sequence_lengths, strict=True
    ):
        assert mask.shape == (sequence_length, sequence_length)


def test_run_diffusion_releases_regional_mask_when_transformer_raises(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)

    class _FailingTransformer(_Transformer):
        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            raise RuntimeError("transformer failure")

    transformer = _FailingTransformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.positive_conditioning = Krea2ConditioningField(
        conditioning_name="positive", mask=TensorField(tensor_name="positive-region")
    )

    with pytest.raises(RuntimeError, match="transformer failure"):
        invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    regional_processor = transformer.installed_processors["transformer_blocks.0.attn.processor"]
    assert regional_processor.regional_prompting_state.attention_mask is None


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

    monkeypatch.setattr("invokeai.app.invocations.krea2.krea2_denoise.RectifiedFlowInpaintExtension", _InpaintExtension)

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
        attn_processors = {"transformer_blocks.0.attn.processor": object()}
        transformer_blocks = torch.nn.ModuleList()
        config = KREA2_CONFIG

        def set_attn_processor(self, processor) -> None:
            pass

        def modules(self):
            return iter(())

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


def _mu_branch_invocation(shift: float | None = None) -> Krea2DenoiseInvocation:
    # shift=None so the resolution-aware mu branch runs (not the fixed-mu override); width/height
    # 16 -> a 2x2 latent -> 1x1 grid -> image_seq_len == 1.
    return Krea2DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=_model_identifier(), loras=[]),
        positive_conditioning=Krea2ConditioningField(conditioning_name="positive"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=16,
        height=16,
        steps=2,
        seed=1,
        shift=shift,
        denoising_start=0.0,
        denoising_end=1.0,
        latents=None,
        denoise_mask=None,
    )


def _mu_branch_context(tmp_path, transformer: _Transformer, variant=Krea2VariantType.Base):
    # variant=Base -> _is_distilled() is False -> the resolution-aware mu branch is taken.
    # Pass variant=Turbo to exercise the distilled fixed-mu branch instead.
    conditionings = {
        "positive": ConditioningFieldData(conditionings=[Krea2ConditioningInfo(prompt_embeds=torch.ones(1, 2, 12, 8))]),
    }
    config = SimpleNamespace(format=ModelFormat.Checkpoint, variant=variant)
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: config,
            get_absolute_path=lambda _config: tmp_path,
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda _name: None),
        util=SimpleNamespace(sd_step_callback=lambda *_args: None),
    )


def _install_mu_capturing_scheduler(monkeypatch, captured: list[float], stop: bool = False) -> None:
    class _MuCapturingScheduler(FlowMatchEulerDiscreteScheduler):
        """The real scheduler, so a checkpoint's scheduler_config.json loads as it would in production."""

        def set_timesteps(self, *, sigmas, mu, device) -> None:
            captured.append(mu)
            # A frame of real size need not be denoised to know the mu it was handed.
            if stop:
                raise _StopAtTimesteps
            self.sigmas = torch.tensor([*sigmas, 0.0], device=device)
            self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps

    monkeypatch.setattr(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
        _MuCapturingScheduler,
    )


class _StopAtTimesteps(Exception):
    """Ends a run once the scheduler has been handed its mu."""


@pytest.mark.parametrize(
    ("width", "height", "mu"),
    [
        (512, 512, 0.58125),  # 1024 tokens: on the line
        (1024, 1024, 0.90625),  # 4096 tokens
        (2048, 768, 1.1229166666666667),  # 6144 tokens: area decides, not a side
        (1280, 1280, 1.15),  # 6400 tokens: the line's end
        (2048, 2048, 1.15),  # 16384 tokens: would be 2.206 extrapolated
    ],
)
def test_raw_mu_follows_the_stock_fit_up_to_its_end_and_holds_there(monkeypatch, tmp_path, width, height, mu) -> None:
    # The model folder has no scheduler/, as for a GGUF or single-file transformer, so the stock scheduler is used.
    _patch_runtime(monkeypatch)
    captured: list[float] = []
    _install_mu_capturing_scheduler(monkeypatch, captured, stop=True)
    invocation = _mu_branch_invocation()
    invocation.width, invocation.height = width, height

    with pytest.raises(_StopAtTimesteps):
        invocation._run_diffusion(_mu_branch_context(tmp_path, _Transformer()))

    assert captured == [pytest.approx(mu)]


@pytest.mark.parametrize(
    ("size", "mu"),
    [
        (16, 0.168),  # 1 token: 0.3 + 1.2 * (1 - 100) / 900, below this fit's lower end
        (2048, 1.5),  # 16384 tokens: held at this config's max_shift
    ],
)
def test_raw_mu_follows_and_holds_at_the_end_of_a_checkpoints_own_scheduler_fit(
    monkeypatch, tmp_path, size: int, mu: float
) -> None:
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "use_dynamic_shifting": True,
                "base_image_seq_len": 100,
                "max_image_seq_len": 1000,
                "base_shift": 0.3,
                "max_shift": 1.5,
                "time_shift_type": "exponential",
            }
        )
    )
    _patch_runtime(monkeypatch)
    captured: list[float] = []
    _install_mu_capturing_scheduler(monkeypatch, captured, stop=True)
    invocation = _mu_branch_invocation()
    invocation.width = invocation.height = size

    with pytest.raises(_StopAtTimesteps):
        invocation._run_diffusion(_mu_branch_context(tmp_path, _Transformer()))

    assert captured == [pytest.approx(mu)]


@pytest.mark.parametrize("shift", [0.0, -1.0])
def test_run_diffusion_treats_a_non_positive_shift_as_unset_on_turbo(monkeypatch, tmp_path, shift: float) -> None:
    """Regression: an untouched shift field arrived as 0.0 and silently replaced the distilled mu.

    The node editor builds every float field with ``schemaObject.default ?? 0``, so an optional shift the
    user never touched reaches the node as 0.0 rather than None. Taking that literally set mu=0, which
    flattens the Turbo sigma schedule and produced visibly grainy images.
    """
    from invokeai.backend.krea2.sampling_utils import KREA2_DISTILLED_MU

    _patch_runtime(monkeypatch)
    captured: list[float] = []
    _install_mu_capturing_scheduler(monkeypatch, captured)

    _mu_branch_invocation(shift=shift)._run_diffusion(
        _mu_branch_context(tmp_path, _Transformer(), variant=Krea2VariantType.Turbo)
    )

    assert captured == [pytest.approx(KREA2_DISTILLED_MU)]


def test_run_diffusion_treats_a_zero_shift_as_unset_on_raw(monkeypatch, tmp_path) -> None:
    # The same bypass must fall through to the resolution-aware branch for an undistilled checkpoint.
    _patch_runtime(monkeypatch)
    captured: list[float] = []
    _install_mu_capturing_scheduler(monkeypatch, captured)

    _mu_branch_invocation(shift=0.0)._run_diffusion(_mu_branch_context(tmp_path, _Transformer()))

    assert captured == [pytest.approx(0.5 + 0.65 * (1 - 256) / 6144)]  # 1 token on the stock fit


def test_run_diffusion_still_honors_an_explicit_positive_shift(monkeypatch, tmp_path) -> None:
    # The override itself must keep working - only non-positive values are reinterpreted as "unset".
    _patch_runtime(monkeypatch)
    captured: list[float] = []
    _install_mu_capturing_scheduler(monkeypatch, captured)

    _mu_branch_invocation(shift=1.8)._run_diffusion(
        _mu_branch_context(tmp_path, _Transformer(), variant=Krea2VariantType.Turbo)
    )

    assert captured == [pytest.approx(1.8)]


def _working_memory(monkeypatch, tmp_path, invocation, transformer=None, context=None, models=None):
    """The estimate the node hands the transformer's lock. `models` are what loading another model's key gives."""
    transformer = transformer if transformer is not None else _Transformer()
    info = _TransformerInfo(transformer)
    context = context if context is not None else _runtime_context(tmp_path, transformer)
    context.models.load = lambda identifier: (models or {}).get(identifier.key, info)
    _patch_runtime(monkeypatch)
    invocation._run_diffusion(context)
    return info.working_memory


@pytest.mark.parametrize(("negative_text_seq_len", "text_tokens"), [(1, 2), (2, 2), (5, 5)])
def test_the_lock_prices_the_longer_cfg_caption(monkeypatch, tmp_path, negative_text_seq_len, text_tokens) -> None:
    """The conditional and unconditional passes run one after the other: the longer of the two captions counts, the
    2-token positive one or the negative one. Without CFG the negative caption is never loaded."""
    context = _runtime_context(tmp_path, _Transformer(), negative_text_seq_len=negative_text_seq_len)
    with_cfg = _working_memory(monkeypatch, tmp_path, _runtime_invocation(cfg_scale=4.0), context=context)
    context = _runtime_context(tmp_path, _Transformer(), negative_text_seq_len=negative_text_seq_len)
    without_cfg = _working_memory(monkeypatch, tmp_path, _runtime_invocation(cfg_scale=1.0), context=context)

    assert with_cfg.estimator == "denoise.krea2"
    assert (with_cfg.features["text_tokens"], with_cfg.features["cfg"]) == (text_tokens, True)
    assert (without_cfg.features["text_tokens"], without_cfg.features["cfg"]) == (2, False)


def test_the_lock_prices_the_loras_at_their_file_size(monkeypatch, tmp_path) -> None:
    """A patched LoRA adds about its own size: 0.03 GiB measured +0.02 GiB, 0.21 GiB +0.22 GiB on an RTX 4090."""
    sizes = {"lora-a": 30 * 2**20, "lora-b": 210 * 2**20}
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.transformer = TransformerField(
        transformer=_model_identifier(),
        loras=[
            LoRAField(
                lora=ModelIdentifierField(key=key, hash=key, name=key, base=BaseModelType.Krea2, type=ModelType.LoRA),
                weight=1.0,
            )
            for key in sizes
        ],
    )
    patches = {
        key: SimpleNamespace(model=ModelPatchRaw(layers={}), config=SimpleNamespace(file_size=size))
        for key, size in sizes.items()
    }

    estimate = _working_memory(monkeypatch, tmp_path, invocation, models=patches)

    assert (estimate.features["lora_count"], estimate.features["lora_bytes"]) == (2, sum(sizes.values()))
    assert {term.name: term.bytes for term in estimate.terms}["lora_patches"] == sum(sizes.values())


class _LoRAInfo:
    """`LoadedModel` of a LoRA: its patch, its config and the RAM pin the patcher enters."""

    def __init__(self) -> None:
        self.model = ModelPatchRaw(layers={})
        self.config = SimpleNamespace(file_size=2**20)
        self.pinned = 0

    @contextmanager
    def model_in_ram(self):
        self.pinned += 1
        yield self.model


def test_each_lora_is_loaded_once_for_the_estimate_and_the_patcher(monkeypatch, tmp_path) -> None:
    """Every load signals progress and emits load events, so a LoRA loaded for the estimate and again for the patcher
    shows twice. Both read the one load, and the patcher pins exactly the patches the lock was sized for."""
    weights = {"lora-a": 0.5, "lora-b": 1.25}
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.transformer = TransformerField(
        transformer=_model_identifier(),
        loras=[
            LoRAField(
                lora=ModelIdentifierField(key=key, hash=key, name=key, base=BaseModelType.Krea2, type=ModelType.LoRA),
                weight=weight,
            )
            for key, weight in weights.items()
        ],
    )
    lora_infos = {key: _LoRAInfo() for key in weights}
    transformer_info = _TransformerInfo(_Transformer())
    loads: list[str] = []

    def load(identifier):
        loads.append(identifier.key)
        return lora_infos.get(identifier.key, transformer_info)

    applied: list[tuple[object, float]] = []

    @contextmanager
    def patcher(*, patches, **_kwargs):
        for patch, weight, in_ram in patches:
            with in_ram as pinned:
                assert pinned is patch
            applied.append((patch, weight))
        yield

    context = _runtime_context(tmp_path, transformer_info.transformer)
    context.models.load = load
    # Records answer with a file size too, so the count, not the size's source, decides the outcome.
    context.models.get_config(None).file_size = 2**20
    _patch_runtime(monkeypatch)
    monkeypatch.setattr("invokeai.app.invocations.krea2.krea2_denoise.LayerPatcher.apply_smart_model_patches", patcher)

    invocation._run_diffusion(context)

    assert sorted(key for key in loads if key in weights) == sorted(weights)
    assert applied == [(lora_infos[key].model, weight) for key, weight in weights.items()]
    assert [info.pinned for info in lora_infos.values()] == [1, 1]
    assert transformer_info.working_memory.features["lora_count"] == 2


def test_a_loaded_lora_that_is_no_patch_is_refused_before_the_lock(monkeypatch, tmp_path) -> None:
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.transformer = TransformerField(
        transformer=_model_identifier(),
        loras=[
            LoRAField(
                lora=ModelIdentifierField(key="x", hash="x", name="x", base=BaseModelType.Krea2, type=ModelType.LoRA),
                weight=1.0,
            )
        ],
    )

    with pytest.raises(TypeError, match="Expected ModelPatchRaw for LoRA 'x'"):
        _working_memory(monkeypatch, tmp_path, invocation, models={"x": SimpleNamespace(model=object())})


def _style_reference_field(
    *, latents_name: str = "style-ref", width: int = 16, height: int = 16, **overrides
) -> Krea2StyleReferenceField:
    # blocks="0-1" because the stub transformer only exposes two main blocks.
    return Krea2StyleReferenceField(
        reference_latents_name=latents_name,
        width=width,
        height=height,
        blocks="0-1",
        **overrides,
    )


def test_run_diffusion_runs_a_reference_pass_before_each_target_pass(monkeypatch, tmp_path) -> None:
    # Two passes per step: the reference is captured, then spliced into the target. The reference pass must
    # never carry a regional mask -- it has its own, different sequence length.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field()

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    assert transformer.style_modes == [
        Krea2StyleReferenceMode.CAPTURE,
        Krea2StyleReferenceMode.INJECT,
        Krea2StyleReferenceMode.CAPTURE,
        Krea2StyleReferenceMode.INJECT,
    ]


def test_run_diffusion_injects_style_into_both_cfg_passes(monkeypatch, tmp_path) -> None:
    # Styling only the conditional pass would make CFG amplify (styled_cond - plain_uncond). One reference
    # pass is shared by both, so this is three forwards per step and not four.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.style_reference = _style_reference_field()

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    assert (
        transformer.style_modes
        == [
            Krea2StyleReferenceMode.CAPTURE,
            Krea2StyleReferenceMode.INJECT,
            Krea2StyleReferenceMode.INJECT,
        ]
        * 2
    )


def test_run_diffusion_reuses_the_positive_prompt_for_the_reference_pass_by_default(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field()

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    # The positive conditioning is all ones; the reference pass sees the same embeddings.
    assert transformer.conditioning_values == [1.0, 1.0, 1.0, 1.0]


def test_run_diffusion_uses_reference_specific_position_ids(monkeypatch, tmp_path) -> None:
    # The style prompt can tokenize to a different length than the positive prompt. The reference pass needs
    # its own rotary position ids or the real transformer's apply_rotary_emb sees a mismatched length.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field()
    invocation.style_reference_conditioning = Krea2ConditioningField(conditioning_name="style-prompt")

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    image_seq_len = 1  # width=height=16 -> 2x2 latent -> a single 2x2 patch
    # Reference prompt is 5 tokens, positive is 2.
    assert transformer.position_id_lengths == [5 + image_seq_len, 2 + image_seq_len] * 2


def test_run_diffusion_clears_the_style_cache_when_the_transformer_raises(monkeypatch, tmp_path) -> None:
    # The processors stay installed on the *cached* transformer, so a retained cache leaks up to ~1.7 GiB
    # of VRAM until the app restarts.
    _patch_runtime(monkeypatch)

    class _FailingTransformer(_Transformer):
        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            raise RuntimeError("transformer failure")

    transformer = _FailingTransformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field()

    with pytest.raises(RuntimeError, match="transformer failure"):
        invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    style_state = transformer.installed_processors["transformer_blocks.0.attn.processor"].style_reference_state
    assert style_state is not None
    assert style_state._cache == {}
    assert style_state.mode is Krea2StyleReferenceMode.OFF


def test_run_diffusion_rejects_a_style_reference_encoded_at_a_different_size(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field(width=32, height=32)

    with pytest.raises(ValueError, match="encoded at 32x32 but denoise is set to 16x16"):
        invocation._run_diffusion(_runtime_context(tmp_path, _Transformer()))


def test_run_diffusion_rejects_style_reference_latents_of_the_wrong_shape(monkeypatch, tmp_path) -> None:
    # The recorded dims agree but the tensor itself does not -- catch it before it reaches attention.
    _patch_runtime(monkeypatch)
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field(latents_name="style-ref-wrong-size")

    with pytest.raises(ValueError, match=r"latents are \(4, 4\) but denoise expects \(2, 2\)"):
        invocation._run_diffusion(_runtime_context(tmp_path, _Transformer()))


class _SubclassWithOwnProcessors(Krea2DenoiseInvocation):
    """Stands in for the prompt-weighting node pack, which overrides the attention-processor seam."""

    installed: ClassVar[bool] = False

    def _install_attention_processors(self, transformer, exit_stack):
        type(self).installed = True
        return super()._install_attention_processors(transformer, exit_stack)


def test_a_subclass_overriding_the_attention_seam_still_runs_without_style_reference(monkeypatch, tmp_path) -> None:
    # Regression: style reference must not widen _install_attention_processors' signature. Custom node
    # packs override that seam with the documented two-argument form, and a third positional argument
    # broke every generation they ran - style reference connected or not.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _SubclassWithOwnProcessors.model_construct(**_runtime_invocation(cfg_scale=1.0).__dict__)
    _SubclassWithOwnProcessors.installed = False

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    assert _SubclassWithOwnProcessors.installed is True


def test_style_reference_refuses_a_subclass_that_installs_its_own_processors(monkeypatch, tmp_path) -> None:
    # Re-installing would silently discard the subclass's processors, so say so instead.
    _patch_runtime(monkeypatch)
    invocation = _SubclassWithOwnProcessors.model_construct(**_runtime_invocation(cfg_scale=1.0).__dict__)
    invocation.style_reference = _style_reference_field()

    with pytest.raises(ValueError, match="installs its own Krea-2 attention processors"):
        invocation._run_diffusion(_runtime_context(tmp_path, _Transformer()))


def test_run_diffusion_without_a_style_reference_runs_no_extra_passes(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()

    _runtime_invocation(cfg_scale=1.0)._run_diffusion(_runtime_context(tmp_path, transformer))

    assert transformer.style_modes == [None, None]
    assert transformer.installed_processors["transformer_blocks.0.attn.processor"].style_reference_state is None


def test_run_diffusion_ignores_a_style_reference_at_strength_zero(monkeypatch, tmp_path) -> None:
    # 0 is documented as disabling the reference. Running the machinery anyway would cost a capture pass per
    # step (~2x runtime) and a retained K/V cache, all for an attention mix of 0.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field(style_strength=0.0)

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    assert transformer.style_modes == [None, None]
    assert transformer.installed_processors["transformer_blocks.0.attn.processor"].style_reference_state is None


def test_run_diffusion_still_runs_a_style_reference_at_a_small_strength(monkeypatch, tmp_path) -> None:
    # Only an exact 0 is the bypass; anything above it still styles.
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field(style_strength=0.01)

    invocation._run_diffusion(_runtime_context(tmp_path, transformer))

    assert Krea2StyleReferenceMode.CAPTURE in transformer.style_modes


def test_the_lock_prices_the_style_references_captured_kv(monkeypatch, tmp_path) -> None:
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.style_reference = _style_reference_field()

    estimate = _working_memory(monkeypatch, tmp_path, invocation)

    # Two styled blocks, one key and one value each, 12 K/V heads of 128 over the one target token, in fp32.
    assert estimate.features["style_kv_bytes"] == 2 * 2 * 12 * 1 * 128 * 4
    # The reference pass reuses the 2-token positive caption.
    assert estimate.features["style_text_tokens"] == 2


def test_regional_attention_memory_includes_masks_build_scratch_and_dtype_sized_attention_bias() -> None:
    positive = SimpleNamespace(attention_mask_numel=120, attention_mask_build_scratch_numel=40)
    negative = SimpleNamespace(attention_mask_numel=100, attention_mask_build_scratch_numel=40)

    assert Krea2DenoiseInvocation._regional_attention_mask_bytes(positive, negative, torch.bfloat16) == 500
    assert Krea2DenoiseInvocation._regional_attention_mask_bytes(positive, negative, torch.float32) == 740
    assert Krea2DenoiseInvocation._regional_attention_mask_bytes(positive, None, torch.bfloat16) == 400


class TestRequiresSidecarPatching:
    """LoRA cannot be written into an int8 buffer, and the config format does not say when one
    is present: an `int8_tensorwise` Krea-2 is a plain `checkpoint` as far as the config knows."""

    class _Tree:
        def __init__(self, *modules) -> None:
            self._modules = modules

        def modules(self):
            return iter(self._modules)

    def test_an_unquantized_checkpoint_is_patched_directly(self) -> None:
        assert not requires_sidecar_patching(self._Tree(torch.nn.Linear(2, 2)), ModelFormat.Checkpoint)

    def test_a_gguf_model_is_still_recognised_by_its_format(self) -> None:
        assert requires_sidecar_patching(self._Tree(), ModelFormat.GGUFQuantized)

    def test_an_int8_convrot_checkpoint_is_recognised_by_its_modules(self) -> None:
        int8_linear = Int8ConvrotLinear(
            weight=torch.zeros(4, 4, dtype=torch.int8), weight_scale=torch.ones(4, 1), convrot=False
        )
        tree = self._Tree(torch.nn.Linear(2, 2), int8_linear)
        assert requires_sidecar_patching(tree, ModelFormat.Checkpoint)


class TestTheInt8DequantTransientReachesTheReservation:
    """An int8_tensorwise build materializes the dequantized, derotated weight per forward call.

    The activation estimate cannot see it -- the weights are buffers, and the transient is allocated
    inside the layer -- so without this the cache reserves for the activations alone and a model that
    loaded comfortably OOMs in its first step. A helper existing is not the same as it being called,
    which is the regression worth catching here rather than in the helper's own unit test.
    """

    def test_an_int8_model_reserves_the_transient_of_its_block_layer(self, monkeypatch, tmp_path) -> None:
        quantized = _Transformer()
        layer = Int8ConvrotLinear(
            weight=torch.zeros(1024, 512, dtype=torch.int8),
            weight_scale=torch.ones(1024, 1),
            convrot=False,
        )
        quantized.transformer_blocks = torch.nn.ModuleList([layer])
        quantized.modules = lambda: quantized.transformer_blocks.modules()

        estimate = _working_memory(monkeypatch, tmp_path, _runtime_invocation(cfg_scale=1.0), transformer=quantized)

        # `_patch_runtime` pins the compute dtype to float32, which is what the node passes on.
        transient = peak_dequant_transient_bytes(layer, torch.float32)
        terms = {term.name: term.bytes for term in estimate.terms}
        assert transient > 0
        assert terms["dequant_transient"] == transient
        # The node cannot know the cache will load the transformer whole: the layer's int8 weight and float32 scale
        # are reserved for streaming.
        assert terms["streamed_layer"] == 1024 * 512 + 1024 * 4


class TestTheAttentionScoreMatrixReachesTheReservation:
    """The activation estimate assumes a fused attention kernel. Where the build has none -- ROCm on Windows, MPS --
    the main blocks also build a score matrix (12.9 GiB per call for 1024px unchunked), and the reservation must
    price it: for the heads of the loaded model and the sequence the transformer really attends over."""

    def _reservation_for(
        self, monkeypatch, tmp_path, score_bytes: int, invocation=None, **context_kwargs
    ) -> tuple[int, list[dict], _Transformer]:
        transformer = _Transformer()
        info = _TransformerInfo(transformer)
        context = _runtime_context(tmp_path, transformer, **context_kwargs)
        context.models.load = lambda _identifier: info
        _patch_runtime(monkeypatch)
        asked: list[dict] = []

        def score_matrix_bytes(**kwargs) -> int:
            asked.append(kwargs)
            return score_bytes

        monkeypatch.setattr(_ESTIMATOR_MODULE, "sdpa_score_matrix_bytes", score_matrix_bytes)
        (invocation or _runtime_invocation(cfg_scale=1.0))._run_diffusion(context)
        return info.working_memory.reservation_bytes, asked, transformer

    def test_the_score_matrix_is_reserved_for_the_attended_sequence(self, monkeypatch, tmp_path) -> None:
        fused_bytes, _, _ = self._reservation_for(monkeypatch, tmp_path, score_bytes=0)
        materializing_bytes, asked, transformer = self._reservation_for(monkeypatch, tmp_path, score_bytes=12345)

        assert materializing_bytes == fused_bytes + 12345
        assert len(asked) == 1
        assert (asked[0]["num_heads"], asked[0]["head_dim"]) == (48, 128)
        assert asked[0]["seq_len"] == transformer.combined_sequence_lengths[0]
        assert asked[0]["has_attn_mask"] is False

    def test_the_longer_of_the_cfg_passes_and_the_regional_mask_are_priced(self, monkeypatch, tmp_path) -> None:
        invocation = _runtime_invocation(cfg_scale=2.0)
        invocation.positive_conditioning = Krea2ConditioningField(
            conditioning_name="positive", mask=TensorField(tensor_name="positive-region")
        )
        invocation.negative_conditioning = Krea2ConditioningField(
            conditioning_name="negative", mask=TensorField(tensor_name="negative-region")
        )

        _, asked, transformer = self._reservation_for(
            monkeypatch, tmp_path, score_bytes=1, invocation=invocation, negative_text_seq_len=5
        )

        assert len(set(transformer.combined_sequence_lengths)) == 2, "the passes must attend over different lengths"
        assert asked[0]["seq_len"] == max(transformer.combined_sequence_lengths)
        assert asked[0]["has_attn_mask"] is True


# --- Edit references (reference_latents) -----------------------------------------------------------------

# width=height=16 is a 2x2 latent, i.e. one target token. The references are larger on purpose: their own
# grids, not the target's, must size their tokens and position ids.
_REFERENCE_TENSORS = {
    "ref-a": torch.ones(1, KREA2_LATENT_CHANNELS, 1, 4, 2),  # 5D, as the Qwen-Image VAE emits: 2x1 grid
    "ref-b": torch.ones(1, KREA2_LATENT_CHANNELS, 2, 2),  # 4D: 1x1 grid
    "ref-wrong-channels": torch.ones(1, 4, 2, 2),
    "ref-odd": torch.ones(1, KREA2_LATENT_CHANNELS, 3, 2),
    "ref-two-frames": torch.ones(1, KREA2_LATENT_CHANNELS, 2, 2, 2),
}
_REFERENCE_TOKENS = 2 + 1


def _reference_context(
    tmp_path, transformer, positive_references: int = 0, negative_references: int | None = None, **kwargs
):
    """`_runtime_context` plus the reference tensors, with prompts encoded as having seen `*_references` images."""
    context = _runtime_context(tmp_path, transformer, **kwargs)
    base_tensors = context.tensors.load
    context.tensors.load = lambda name: _REFERENCE_TENSORS[name] if name in _REFERENCE_TENSORS else base_tensors(name)
    counts = {
        "positive": positive_references,
        "negative": positive_references if negative_references is None else negative_references,
    }
    base_conditioning = context.conditioning.load

    def load_conditioning(name: str) -> ConditioningFieldData:
        data = base_conditioning(name)
        if name not in counts:
            return data
        info = dataclasses.replace(data.conditionings[0], num_reference_images=counts[name])
        return ConditioningFieldData(conditionings=[info])

    context.conditioning.load = load_conditioning
    context.util.signal_progress = lambda *_args, **_kwargs: None
    return context


def _references(*names: str) -> list[LatentsField]:
    return [LatentsField(latents_name=name) for name in names]


def _tiny_krea2_transformer():
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

    # Text width and layer count match the runtime context's (1, seq, 12, 8) prompt embeddings.
    torch.manual_seed(0)
    return Krea2Transformer2DModel(
        in_channels=64,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=12,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()


def _refusing_context(tmp_path, positive_references: int = 0, negative_references: int | None = None):
    context = _reference_context(tmp_path, _Transformer(), positive_references, negative_references)

    def load(_identifier):
        raise AssertionError("the transformer must not be loaded for a refused request")

    context.models.load = load
    return context


@pytest.mark.parametrize(
    ("configure", "message"),
    [
        (
            lambda inv: setattr(inv, "reference_latents", _references("ref-b", "ref-b", "ref-b", "ref-b")),
            "at most 3 reference latents, got 4",
        ),
        (
            lambda inv: setattr(inv, "style_reference", _style_reference_field()),
            "cannot be combined with a style reference",
        ),
        (
            lambda inv: setattr(
                inv,
                "positive_conditioning",
                Krea2ConditioningField(conditioning_name="positive", mask=TensorField(tensor_name="positive-region")),
            ),
            "cannot be combined with regional",
        ),
        (
            lambda inv: setattr(
                inv,
                "negative_conditioning",
                [Krea2ConditioningField(conditioning_name="negative", mask=TensorField(tensor_name="negative-region"))],
            ),
            "cannot be combined with regional",
        ),
        (lambda inv: setattr(inv, "reference_latents", _references("ref-wrong-channels")), r"expected \(1, 16"),
        (lambda inv: setattr(inv, "reference_latents", _references("ref-two-frames")), r"expected \(1, 16"),
        (lambda inv: setattr(inv, "reference_latents", _references("ref-a", "ref-odd")), "latents 2 are 3x2"),
    ],
)
def test_run_diffusion_refuses_unusable_references_before_loading_the_transformer(
    monkeypatch, tmp_path, configure, message
) -> None:
    _patch_runtime(monkeypatch)
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = _references("ref-a")
    configure(invocation)
    # The prompts agree with the references, so only the refusal under test can fire.
    encoded = len(invocation._reference_latent_fields())

    with pytest.raises(ValueError, match=message):
        invocation._run_diffusion(_refusing_context(tmp_path, positive_references=encoded))


@pytest.mark.parametrize(
    ("reference_latents", "positive_references", "negative_references", "message"),
    [
        # Latents without the prompt having seen the images.
        (
            ["ref-b"],
            0,
            None,
            "positive Krea-2 prompt was encoded with 0 reference image.*has 1 reference latents. Connect the same reference images, in the same order",
        ),
        # A prompt encoded with references, but no latents for the transformer.
        ([], 2, None, "positive Krea-2 prompt was encoded with 2 reference image.*has 0 reference latents"),
        # The positive prompt agrees; the negative one does not.
        (["ref-a", "ref-b"], 2, 1, "negative Krea-2 prompt was encoded with 1 reference image"),
    ],
)
def test_run_diffusion_refuses_prompts_encoded_with_a_different_reference_count(
    monkeypatch, tmp_path, reference_latents, positive_references, negative_references, message
) -> None:
    _patch_runtime(monkeypatch)
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = _references(*reference_latents)

    with pytest.raises(ValueError, match=message):
        invocation._run_diffusion(_refusing_context(tmp_path, positive_references, negative_references))


def test_a_zero_strength_style_reference_is_no_conflict_with_references() -> None:
    # Strength 0 is documented as "nothing connected".
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.reference_latents = _references("ref-b")
    invocation.style_reference = _style_reference_field(style_strength=0.0)

    invocation._validate_inputs()


@torch.no_grad()
def test_run_diffusion_feeds_the_references_to_both_cfg_passes(monkeypatch, tmp_path) -> None:
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module

    _patch_runtime(monkeypatch)
    transformer = _tiny_krea2_transformer()
    calls: list[dict] = []
    real_forward = denoise_module.forward_with_references

    def recording_forward(model, hidden_states, encoder_hidden_states, timestep, position_ids, references):
        calls.append({"text_len": encoder_hidden_states.shape[1], "ids": position_ids, "refs": references})
        return real_forward(model, hidden_states, encoder_hidden_states, timestep, position_ids, references)

    monkeypatch.setattr(denoise_module, "forward_with_references", recording_forward)
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = _references("ref-a", "ref-b")

    latents = invocation._run_diffusion(
        _reference_context(tmp_path, transformer, positive_references=2, negative_text_seq_len=3)
    )

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    # Two steps, conditional then unconditional, each with its own text length and the same references.
    assert [call["text_len"] for call in calls] == [2, 3, 2, 3]
    for call in calls:
        assert call["refs"].shape == (1, _REFERENCE_TOKENS, KREA2_LATENT_CHANNELS * 4)
        assert torch.equal(
            call["ids"][call["text_len"] :],
            torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=torch.float32),
        )


def test_run_diffusion_without_references_keeps_the_stock_transformer_call(monkeypatch, tmp_path) -> None:
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module

    _patch_runtime(monkeypatch)

    def unexpected(*_args, **_kwargs):
        pytest.fail("the reference path ran without references")

    monkeypatch.setattr(denoise_module, "forward_with_references", unexpected)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = []

    invocation._run_diffusion(_reference_context(tmp_path, transformer))

    assert transformer.position_id_lengths == [2 + 1, 2 + 1] * 2


def test_run_diffusion_reserves_memory_for_the_reference_tokens(monkeypatch, tmp_path) -> None:
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module

    asked_seq_lens: list[int] = []

    def working_memory(reference_latents):
        transformer = _Transformer()
        context = _reference_context(tmp_path, transformer, positive_references=len(reference_latents or []))
        invocation = _runtime_invocation(cfg_scale=1.0)
        invocation.reference_latents = reference_latents
        return _working_memory(monkeypatch, tmp_path, invocation, transformer=transformer, context=context)

    def score_matrix_bytes(**kwargs) -> int:
        asked_seq_lens.append(kwargs["seq_len"])
        return 0

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        denoise_module, "forward_with_references", lambda _model, hidden_states, *_args: torch.zeros_like(hidden_states)
    )
    monkeypatch.setattr(_ESTIMATOR_MODULE, "sdpa_score_matrix_bytes", score_matrix_bytes)

    without_references = working_memory(None)
    with_references = working_memory(_references("ref-a", "ref-b"))

    # Every reference token is an image token to every block: on the line, beside 2 text tokens and 1 target token.
    assert (without_references.features["reference_tokens"], without_references.features["tokens"]) == (0, 3)
    assert (with_references.features["reference_tokens"], with_references.features["tokens"]) == (
        _REFERENCE_TOKENS,
        3 + _REFERENCE_TOKENS,
    )
    assert (with_references.features["cached_reference_tokens"], with_references.features["reference_kv_bytes"]) == (
        0,
        0,
    )
    # The attention the score matrix is priced for spans text + target + references.
    assert asked_seq_lens == [2 + 1, 2 + 1 + _REFERENCE_TOKENS]


@torch.no_grad()
def test_run_diffusion_inpaints_only_the_target_next_to_references(monkeypatch, tmp_path) -> None:
    # img2img + inpaint act on the unpacked target latents. With references in the sequence the transformer
    # must still return target tokens only, and the kept region must land exactly on the init latents.
    _patch_runtime(monkeypatch)
    init = torch.randn(1, KREA2_LATENT_CHANNELS, 2, 2, generator=torch.Generator().manual_seed(3))
    keep_left_half = torch.zeros(1, 1, 16, 16)
    keep_left_half[..., :8] = 1.0  # the denoise mask marks what to keep; the node inverts it
    context = _reference_context(tmp_path, _tiny_krea2_transformer(), positive_references=2)
    tensors = {"init": init, "mask": keep_left_half}
    reference_tensors = context.tensors.load
    context.tensors.load = lambda name: tensors[name] if name in tensors else reference_tensors(name)
    invocation = _runtime_invocation(cfg_scale=1.0, with_mask=True)
    invocation.reference_latents = _references("ref-a", "ref-b")

    latents = invocation._run_diffusion(context)

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    torch.testing.assert_close(latents[0, :, 0, :, 0], init[0, :, :, 0])
    assert not torch.allclose(latents[0, :, 0, :, 1], init[0, :, :, 1])


@pytest.mark.parametrize(("cfg_scale", "refused"), [(1.0, False), ([1.0, 1.0], False), ([1.0, 4.0], True)])
def test_the_negative_prompt_reference_count_matters_only_when_cfg_runs(cfg_scale, refused) -> None:
    # A workflow can leave a negative prompt connected at CFG 1, where it never runs.
    invocation = _runtime_invocation(cfg_scale=cfg_scale)
    invocation.reference_latents = _references("ref-a", "ref-b")
    context = _reference_context(None, _Transformer(), positive_references=2, negative_references=0)

    if refused:
        with pytest.raises(ValueError, match="negative Krea-2 prompt was encoded with 0 reference image"):
            invocation._validate_reference_counts(context)
    else:
        invocation._validate_reference_counts(context)


@pytest.mark.parametrize(
    ("cfg_scale", "denoising_start", "denoising_end", "refused"),
    [
        # CFG only in steps the window skips: the negative prompt never runs.
        ([4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.5, 1.0, False),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0], 0.0, 0.5, False),
        # A CFG step inside the window: it runs and must match.
        ([4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0, 1.0, True),
        ([1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0], 0.5, 1.0, True),
    ],
)
def test_the_negative_prompt_reference_count_follows_the_clipped_cfg_schedule(
    cfg_scale, denoising_start, denoising_end, refused
) -> None:
    invocation = _runtime_invocation(cfg_scale=cfg_scale)
    invocation.steps = 8
    invocation.denoising_start = denoising_start
    invocation.denoising_end = denoising_end
    invocation.reference_latents = _references("ref-b")
    context = _reference_context(None, _Transformer(), positive_references=1, negative_references=0)

    if refused:
        with pytest.raises(ValueError, match="negative Krea-2 prompt was encoded with 0 reference image"):
            invocation._validate_reference_counts(context)
    else:
        invocation._validate_reference_counts(context)


def test_run_diffusion_never_loads_a_negative_prompt_whose_cfg_steps_are_skipped(monkeypatch, tmp_path) -> None:
    # CFG only in steps 0-1, which denoising_start=0.5 skips: validation and the run must agree that the negative
    # prompt does not run, so its mismatched reference count is no obstacle and it is never read.
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        denoise_module, "forward_with_references", lambda _model, hidden_states, *_args: torch.zeros_like(hidden_states)
    )
    invocation = _runtime_invocation(cfg_scale=[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    invocation.steps = 8
    invocation.denoising_start = 0.5
    invocation.latents = LatentsField(latents_name="init")
    invocation.reference_latents = _references("ref-b")
    context = _reference_context(tmp_path, _Transformer(), positive_references=1, negative_references=0)
    loaded: list[str] = []
    load_conditioning = context.conditioning.load
    context.conditioning.load = lambda name: loaded.append(name) or load_conditioning(name)

    latents = invocation._run_diffusion(context)

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    assert "negative" not in loaded


@pytest.mark.parametrize(
    ("denoising_start", "denoising_end", "window"),
    [
        # 0.45 * 8 = 3.6 and 0.95 * 8 = 7.6 round up; truncating would give (3, 7).
        (0.45, 0.95, (4, 8)),
        # 0.3 * 8 = 2.4 and 0.8 * 8 = 6.4 round down.
        (0.3, 0.8, (2, 6)),
    ],
)
def test_the_step_window_rounds_to_the_nearest_step(denoising_start, denoising_end, window) -> None:
    invocation = Krea2DenoiseInvocation.model_construct(denoising_start=denoising_start, denoising_end=denoising_end)

    assert invocation._step_window(8) == window


@pytest.mark.parametrize("prompt", ["positive", "negative"])
def test_run_diffusion_refuses_a_collection_of_prompts_with_references(monkeypatch, tmp_path, prompt: str) -> None:
    # Several prompts are concatenated, so each would bring the references' vision tokens into the text again.
    _patch_runtime(monkeypatch)
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = _references("ref-b")
    setattr(invocation, f"{prompt}_conditioning", [Krea2ConditioningField(conditioning_name=prompt)] * 2)

    with pytest.raises(
        ValueError,
        match=f"The {prompt} Krea-2 prompt is a collection of 2 prompts.*connect a single 'Prompt - Krea-2' node",
    ):
        invocation._run_diffusion(_refusing_context(tmp_path, positive_references=1))


@pytest.mark.parametrize(
    ("references", "positive_prompts", "negative_prompts", "cfg_scale"),
    [
        (1, 1, 1, 4.0),  # a collection of one is a single prompt
        (1, 1, 2, 1.0),  # the negative collection never runs at CFG 1
        (0, 2, 2, 4.0),  # without references, several prompts are ordinary
    ],
    ids=["single_prompts", "unused_negative_collection", "no_references"],
)
def test_prompt_collections_are_refused_only_where_references_meet_them(
    references, positive_prompts, negative_prompts, cfg_scale
) -> None:
    invocation = _runtime_invocation(cfg_scale=cfg_scale)
    invocation.reference_latents = _references(*["ref-b"] * references)
    invocation.positive_conditioning = [Krea2ConditioningField(conditioning_name="positive")] * positive_prompts
    invocation.negative_conditioning = [Krea2ConditioningField(conditioning_name="negative")] * negative_prompts

    invocation._validate_reference_counts(_reference_context(None, _Transformer(), positive_references=references))


# --- Edit references in K/V-cache mode (reference_kv_cache) ----------------------------------------------


def _main_block_kv_state(transformer):
    return transformer.attn_processors["transformer_blocks.0.attn.processor"].reference_kv_cache


def _record_reference_kv(monkeypatch) -> list[tuple[int, torch.Tensor]]:
    from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVCache

    stored: list[tuple[int, torch.Tensor]] = []
    real_store = Krea2ReferenceKVCache.store

    def store(self, block_index, key, value):
        stored.append((block_index, value.clone()))
        real_store(self, block_index, key, value)

    monkeypatch.setattr(Krea2ReferenceKVCache, "store", store)
    return stored


@torch.no_grad()
def test_the_reference_kv_cache_is_built_once_per_run_and_released(monkeypatch, tmp_path) -> None:
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module
    from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVCache

    _patch_runtime(monkeypatch)
    stored = _record_reference_kv(monkeypatch)
    extended: list[int] = []
    real_extend = Krea2ReferenceKVCache.extend

    def extend(self, block_index, key, value, attention_mask):
        extended.append(block_index)
        return real_extend(self, block_index, key, value, attention_mask)

    monkeypatch.setattr(Krea2ReferenceKVCache, "extend", extend)

    def unexpected(*_args, **_kwargs):
        pytest.fail("K/V-cache mode must not put the references into the per-step sequence")

    monkeypatch.setattr(denoise_module, "forward_with_references", unexpected)
    transformer = _tiny_krea2_transformer()
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_latents = _references("ref-a", "ref-b")
    invocation.reference_kv_cache = True

    latents = invocation._run_diffusion(
        _reference_context(tmp_path, transformer, positive_references=2, negative_text_seq_len=3)
    )

    assert latents.shape == (1, KREA2_LATENT_CHANNELS, 1, 2, 2)
    # Two steps with CFG are four transformer passes, but the references ran through the blocks once.
    assert [block_index for block_index, _ in stored] == [0, 1]
    assert stored[0][1].shape[2] == _REFERENCE_TOKENS
    # ...and every block of every pass attended over the cached reference keys.
    assert extended == [0, 1] * (2 * 2)
    # The processors stay on the cached transformer; the cache must not.
    state = _main_block_kv_state(transformer)
    assert len(state) == 0
    assert state.mode.name == "OFF"


@torch.no_grad()
def test_the_reference_kv_cache_is_released_when_a_step_raises(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _tiny_krea2_transformer()
    context = _reference_context(tmp_path, transformer, positive_references=1)

    def step_callback(state, _base) -> None:
        if state.step >= 1:
            raise RuntimeError("step failure")

    context.util.sd_step_callback = step_callback
    invocation = _runtime_invocation(cfg_scale=1.0)
    invocation.reference_latents = _references("ref-b")
    invocation.reference_kv_cache = True

    with pytest.raises(RuntimeError, match="step failure"):
        invocation._run_diffusion(context)

    assert len(_main_block_kv_state(transformer)) == 0


@torch.no_grad()
def test_the_reference_kv_cache_is_built_from_the_lora_patched_weights(monkeypatch, tmp_path) -> None:
    from contextlib import contextmanager

    _patch_runtime(monkeypatch)
    stored = _record_reference_kv(monkeypatch)
    transformer = _tiny_krea2_transformer()

    @contextmanager
    def doubling_patch(*, model, **_kwargs):
        # Stands in for a LoRA on block 0's value projection. V has no norm or bias, so the cached V doubles.
        weight = model.transformer_blocks[0].attn.to_v.weight
        weight.mul_(2.0)
        try:
            yield
        finally:
            weight.div_(2.0)

    def cached_block0_value(patcher) -> torch.Tensor:
        stored.clear()
        monkeypatch.setattr(
            "invokeai.app.invocations.krea2.krea2_denoise.LayerPatcher.apply_smart_model_patches", patcher
        )
        invocation = _runtime_invocation(cfg_scale=1.0)
        invocation.reference_latents = _references("ref-b")
        invocation.reference_kv_cache = True
        invocation._run_diffusion(_reference_context(tmp_path, transformer, positive_references=1))
        return next(value for block_index, value in stored if block_index == 0)

    plain = cached_block0_value(lambda **_kwargs: nullcontext())
    patched = cached_block0_value(doubling_patch)

    torch.testing.assert_close(patched, 2.0 * plain)


def test_the_reference_kv_cache_flag_without_references_changes_nothing(monkeypatch, tmp_path) -> None:
    _patch_runtime(monkeypatch)
    transformer = _Transformer()
    invocation = _runtime_invocation(cfg_scale=4.0)
    invocation.reference_kv_cache = True

    invocation._run_diffusion(_reference_context(tmp_path, transformer))

    assert transformer.installed_processors["transformer_blocks.0.attn.processor"].reference_kv_cache is None
    assert transformer.position_id_lengths == [2 + 1, 2 + 1] * 2


@pytest.mark.parametrize("reference_kv_cache", [False, True], ids=["joint", "kv_cache"])
def test_references_refuse_a_subclass_with_its_own_processors(reference_kv_cache: bool) -> None:
    # Its processors are sized for [text | target]: joint references would reach them as extra image tokens, and the
    # K/V cache lives in the stock processors it replaces. Without references it runs (see the seam regression test).
    invocation = _SubclassWithOwnProcessors.model_construct(
        positive_conditioning=Krea2ConditioningField(conditioning_name="positive"),
        reference_latents=_references("ref-b"),
        reference_kv_cache=reference_kv_cache,
    )

    with pytest.raises(ValueError, match="installs its own Krea-2 attention processors, which cannot take reference"):
        invocation._validate_inputs()


def test_the_reference_kv_cache_reserves_its_cache_instead_of_per_step_tokens(monkeypatch, tmp_path) -> None:
    import invokeai.app.invocations.krea2.krea2_denoise as denoise_module

    def working_memory(reference_latents, kv_cache: bool):
        transformer = _Transformer()
        transformer.transformer_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(28))
        context = _reference_context(tmp_path, transformer, positive_references=len(reference_latents or []))
        invocation = _runtime_invocation(cfg_scale=1.0)
        invocation.reference_latents = reference_latents
        invocation.reference_kv_cache = kv_cache
        return _working_memory(monkeypatch, tmp_path, invocation, transformer=transformer, context=context)

    asked_seq_lens: list[int] = []

    def score_matrix_bytes(**kwargs) -> int:
        asked_seq_lens.append(kwargs["seq_len"])
        return 0

    _patch_runtime(monkeypatch)
    # The stub transformer has no blocks to cache; the precompute itself is covered with a real model above.
    monkeypatch.setattr(denoise_module, "precompute_reference_kv", lambda *_args: None)
    monkeypatch.setattr(_ESTIMATOR_MODULE, "sdpa_score_matrix_bytes", score_matrix_bytes)

    without_references = working_memory(None, kv_cache=True)
    cached = working_memory(_references("ref-a", "ref-b"), kv_cache=True)

    # The references leave the step's sequence of 2 text tokens and 1 target token, which is no shorter than theirs.
    assert (cached.features["reference_tokens"], cached.features["cached_reference_tokens"]) == (0, _REFERENCE_TOKENS)
    assert cached.features["tokens"] == without_references.features["tokens"] == 3
    # 28 blocks of 12 K/V heads retained, one block's references expanded to 48 heads; fp32 under the patched runtime.
    retained = 28 * 2 * 12 * 128 * _REFERENCE_TOKENS * 4
    widest_append = 2 * 48 * 128 * _REFERENCE_TOKENS * 4
    assert cached.features["reference_kv_bytes"] == retained + widest_append
    assert without_references.features["reference_kv_bytes"] == 0
    # Scores: S = 2 text + 1 target queries over S + 3 reference keys is the larger area (18 > 3 * 3), priced as
    # the smallest square that holds it.
    step, references = 2 + 1, _REFERENCE_TOKENS
    assert asked_seq_lens == [step, math.ceil(math.sqrt(max(step * (step + references), references**2)))]
