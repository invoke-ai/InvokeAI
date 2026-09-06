import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D

from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.backend.hidiffusion.hidiffusion import (
    _get_raunet_step_range,
    _get_resolution_aware_switching_threshold_ratio,
    _resize_controlnet_residual,
    make_diffusers_cross_attn_down_block,
    make_diffusers_downsampler_block,
    switching_threshold_ratio_dict,
    text_to_img_controlnet_switching_threshold_ratio_dict,
)
from invokeai.backend.hidiffusion.hidiffusion import (
    remove_hidiffusion as real_remove_hidiffusion,
)
from invokeai.backend.stable_diffusion.extensions.hidiffusion import HiDiffusionExt
from invokeai.backend.stable_diffusion.hidiffusion_utils import hidiffusion_patch


class DummySubmodule:
    pass


class PatchedSubmodule(DummySubmodule):
    _parent = DummySubmodule


class DummyUNet:
    def __init__(self):
        self.num_upsamplers = 3
        self.layer = DummySubmodule()

    def named_modules(self):
        return [("", self), ("layer", self.layer)]


class ModelMixin(torch.nn.Module):
    """Minimal diffusers-like UNet accepted by the vendored HiDiffusion type check."""

    def __init__(self):
        super().__init__()
        self.num_upsamplers = 3


class WindowMeanAttention(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return hidden_states.mean(dim=1, keepdim=True).expand_as(hidden_states)


class WindowAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_ada_layer_norm = False
        self.use_ada_layer_norm_zero = False
        self.use_layer_norm = True
        self.use_ada_layer_norm_continuous = False
        self.use_ada_layer_norm_single = False
        self.pos_embed = None
        self.norm1 = torch.nn.Identity()
        self.attn1 = WindowMeanAttention()
        self.only_cross_attention = False
        self.attn2 = None
        self.norm3 = torch.nn.Identity()
        self.ff = torch.nn.Identity()
        self._chunk_size = None


class WindowAttentionModelMixin(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer = WindowAttentionBlock()


class CachedHiDiffusionModelMixin(ModelMixin):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()


def test_hidiffusion_patch_supports_bare_model_mixin_without_public_name_or_path():
    model = ModelMixin()

    assert not hasattr(model, "name_or_path")
    assert not hasattr(model, "_name_or_path")

    with hidiffusion_patch(model, name_or_path="runwayml/stable-diffusion-v1-5"):
        assert model.info["pipeline"] is model
        assert model.num_upsamplers == 15

    assert model.num_upsamplers == 3
    assert not hasattr(model, "_name_or_path")
    assert not hasattr(model, "info")


def test_hidiffusion_window_attention_uses_seeded_generator_instead_of_global_rng():
    module_keys = {
        "down_module_key": [],
        "down_module_key_extra": [],
        "up_module_key": [],
        "up_module_key_extra": [],
        "windown_attn_module_key": ["transformer"],
    }
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1)

    def run_with_global_seed(global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        model = WindowAttentionModelMixin()
        generator = torch.Generator(device="cpu").manual_seed(1234)

        with (
            patch("invokeai.backend.hidiffusion.hidiffusion.sd15_hidiffusion_key", return_value=module_keys),
            hidiffusion_patch(
                model,
                name_or_path="runwayml/stable-diffusion-v1-5",
                apply_raunet=False,
                apply_window_attn=True,
                generator=generator,
            ),
        ):
            model.info["size"] = (8, 8)
            return model.transformer(hidden_states).clone()

    first = run_with_global_seed(0)
    second = run_with_global_seed(1)

    torch.testing.assert_close(first, second)


def test_hidiffusion_window_attention_reuses_shift_within_logical_step():
    module_keys = {
        "down_module_key": [],
        "down_module_key_extra": [],
        "up_module_key": [],
        "up_module_key_extra": [],
        "windown_attn_module_key": ["transformer"],
    }
    model = WindowAttentionModelMixin()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1)

    with (
        patch("invokeai.backend.hidiffusion.hidiffusion.sd15_hidiffusion_key", return_value=module_keys),
        hidiffusion_patch(
            model,
            name_or_path="runwayml/stable-diffusion-v1-5",
            apply_raunet=False,
            apply_window_attn=True,
            generator=generator,
        ),
    ):
        model.info["size"] = (8, 8)
        model.info["step_index"] = 0
        first = model.transformer(hidden_states).clone()
        generator_state_after_first_forward = generator.get_state().clone()
        second = model.transformer(hidden_states).clone()

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(generator.get_state(), generator_state_after_first_forward)

        model.info["step_index"] = 1
        model.transformer(hidden_states)
        assert not torch.equal(generator.get_state(), generator_state_after_first_forward)


@pytest.mark.parametrize("is_text_to_image", [False, True])
def test_hidiffusion_patch_uses_controlnet_aware_forward_for_bare_unet(is_text_to_image: bool):
    model = ModelMixin()
    original_class = model.__class__

    with hidiffusion_patch(
        model,
        name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        apply_raunet=True,
        apply_window_attn=False,
        has_controlnet=True,
        is_controlnet_text_to_image=is_text_to_image,
    ):
        assert model.__class__ is not original_class
        assert model.__class__._parent is original_class
        assert model.info["text_to_img_controlnet"] is is_text_to_image

    assert model.__class__ is original_class


def test_hidiffusion_patch_does_not_replace_unet_forward_for_window_attention_only():
    model = ModelMixin()
    original_class = model.__class__

    with hidiffusion_patch(
        model,
        name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        apply_raunet=False,
        apply_window_attn=True,
        has_controlnet=True,
    ):
        assert model.__class__ is original_class


@pytest.mark.parametrize(("source_size", "target_size"), [(32, 16), (46, 23), (48, 24), (64, 32)])
def test_hidiffusion_resizes_controlnet_residuals_to_current_feature_map(source_size: int, target_size: int):
    residual = torch.ones(1, 2, source_size, source_size)
    feature_map = torch.zeros(1, 2, target_size, target_size)

    resized_residual = _resize_controlnet_residual(residual, feature_map)
    combined = feature_map + resized_residual

    assert combined.shape == feature_map.shape
    torch.testing.assert_close(combined, torch.ones_like(feature_map))


def test_hidiffusion_patch_resets_cached_runtime_state_when_reenabled():
    module_keys = {
        "down_module_key": [],
        "down_module_key_extra": ["block"],
        "up_module_key": [],
        "up_module_key_extra": [],
        "windown_attn_module_key": [],
    }
    model = CachedHiDiffusionModelMixin()

    with patch("invokeai.backend.hidiffusion.hidiffusion.sd15_hidiffusion_key", return_value=module_keys):
        with hidiffusion_patch(model, name_or_path="runwayml/stable-diffusion-v1-5"):
            model.block.timestep = 7
            model.block.T1_ratio = 0.9
            model.block.T1_start = 2
            model.block.T1_end = 8
            model.block.T1 = 9
            model.block.max_timestep = 99

        assert "timestep" not in model.block.__dict__

        with hidiffusion_patch(model, name_or_path="runwayml/stable-diffusion-v1-5"):
            assert model.block.timestep == 0
            assert model.block.T1_ratio == 0
            assert model.block.T1_start == 0
            assert model.block.T1_end == 0
            assert model.block.T1 == 0
            assert model.block.max_timestep == 50


def test_hidiffusion_teardown_restores_downsampler_geometry_after_forward_error():
    module_keys = {
        "down_module_key": ["block"],
        "down_module_key_extra": [],
        "up_module_key": [],
        "up_module_key_extra": [],
        "windown_attn_module_key": [],
    }
    model = ModelMixin()
    model._num_timesteps = 10
    model.block = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
    original_stride = model.block.stride
    original_padding = model.block.padding
    original_dilation = model.block.dilation

    with patch("invokeai.backend.hidiffusion.hidiffusion.sd15_hidiffusion_key", return_value=module_keys):
        with hidiffusion_patch(
            model,
            name_or_path="runwayml/stable-diffusion-v1-5",
            apply_window_attn=False,
        ):
            model.info["size"] = (64, 64)
            with (
                patch(
                    "invokeai.backend.hidiffusion.hidiffusion.F.conv2d",
                    side_effect=RuntimeError("injected convolution failure"),
                ),
                pytest.raises(RuntimeError, match="injected convolution failure"),
            ):
                model.block(torch.zeros(1, 1, 16, 16))

            # Temporary geometry is passed directly to conv2d and never written
            # to the cached module, even before teardown runs.
            assert model.block.stride == original_stride
            assert model.block.padding == original_padding
            assert model.block.dilation == original_dilation

    assert model.block.stride == original_stride
    assert model.block.padding == original_padding
    assert model.block.dilation == original_dilation


def test_hidiffusion_patch_restores_state_when_apply_hidiffusion_raises():
    original_switching = copy.deepcopy(switching_threshold_ratio_dict)
    original_controlnet = copy.deepcopy(text_to_img_controlnet_switching_threshold_ratio_dict)

    model = SimpleNamespace(
        unet=DummyUNet(),
        _name_or_path="original-model-name",
        config=SimpleNamespace(_name_or_path="original-config-name"),
    )
    hook = MagicMock()

    def fake_apply_hidiffusion(patched_model, **kwargs):
        assert patched_model._name_or_path == "patched-model-name"
        assert patched_model.config._name_or_path == "patched-model-name"

        assert kwargs["t1_ratio"] == 0.25
        assert kwargs["t2_ratio"] == 0.1
        assert switching_threshold_ratio_dict == original_switching
        assert text_to_img_controlnet_switching_threshold_ratio_dict == original_controlnet

        patched_model.unet.num_upsamplers = 99
        patched_model.unet.layer.info = {"hooks": [hook]}
        patched_model.unet.layer.__class__ = PatchedSubmodule
        raise RuntimeError("hidiffusion boom")

    try:
        with (
            patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion", side_effect=fake_apply_hidiffusion),
            patch(
                "invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion",
                wraps=real_remove_hidiffusion,
            ) as mock_remove_hidiffusion,
        ):
            with pytest.raises(RuntimeError, match="hidiffusion boom"):
                with hidiffusion_patch(
                    model,
                    name_or_path="patched-model-name",
                    t1_ratio=0.25,
                    t2_ratio=0.1,
                ):
                    pass

        assert mock_remove_hidiffusion.call_count == 1
        assert switching_threshold_ratio_dict == original_switching
        assert text_to_img_controlnet_switching_threshold_ratio_dict == original_controlnet
        assert model.unet.num_upsamplers == 3
        assert model.unet.layer.__class__ is DummySubmodule
        assert model.unet.layer.info["hooks"] == []
        hook.remove.assert_called_once()
        assert model._name_or_path == "original-model-name"
        assert model.config._name_or_path == "original-config-name"
    finally:
        switching_threshold_ratio_dict.clear()
        switching_threshold_ratio_dict.update(original_switching)
        text_to_img_controlnet_switching_threshold_ratio_dict.clear()
        text_to_img_controlnet_switching_threshold_ratio_dict.update(original_controlnet)


def test_hidiffusion_patch_restores_state_before_propagating_remove_error():
    original_switching = copy.deepcopy(switching_threshold_ratio_dict)
    original_controlnet = copy.deepcopy(text_to_img_controlnet_switching_threshold_ratio_dict)

    model = SimpleNamespace(
        unet=DummyUNet(),
        _name_or_path="original-model-name",
        config=SimpleNamespace(_name_or_path="original-config-name"),
    )

    def fake_apply_hidiffusion(patched_model, **_kwargs):
        patched_model.unet.num_upsamplers = 99

    try:
        with (
            patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion", side_effect=fake_apply_hidiffusion),
            patch(
                "invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion",
                side_effect=RuntimeError("remove boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="remove boom"):
                with hidiffusion_patch(
                    model,
                    name_or_path="patched-model-name",
                    t1_ratio=0.25,
                    t2_ratio=0.1,
                ):
                    pass

        assert switching_threshold_ratio_dict == original_switching
        assert text_to_img_controlnet_switching_threshold_ratio_dict == original_controlnet
        assert model.unet.num_upsamplers == 3
        assert model._name_or_path == "original-model-name"
        assert model.config._name_or_path == "original-config-name"
    finally:
        switching_threshold_ratio_dict.clear()
        switching_threshold_ratio_dict.update(original_switching)
        text_to_img_controlnet_switching_threshold_ratio_dict.clear()
        text_to_img_controlnet_switching_threshold_ratio_dict.update(original_controlnet)


def test_hidiffusion_patch_removes_spoofed_name_from_config_internal_dict():
    class InternalDictConfig:
        def __init__(self):
            self._internal_dict = {}

        def __getattr__(self, name):
            try:
                return self._internal_dict[name]
            except KeyError as error:
                raise AttributeError(name) from error

    config = InternalDictConfig()
    model = SimpleNamespace(unet=DummyUNet(), config=config)

    with (
        patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion"),
        patch("invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion"),
    ):
        with hidiffusion_patch(model, name_or_path="patched-model-name"):
            assert config._internal_dict["_name_or_path"] == "patched-model-name"

    assert "_name_or_path" not in config._internal_dict


def test_hidiffusion_ratio_overrides_are_isolated_between_overlapping_patches():
    original_switching = copy.deepcopy(switching_threshold_ratio_dict)
    original_controlnet = copy.deepcopy(text_to_img_controlnet_switching_threshold_ratio_dict)
    first_model = SimpleNamespace(unet=DummyUNet())
    second_model = SimpleNamespace(unet=DummyUNet())
    applied_overrides: list[tuple[object, float | None, float | None]] = []

    def fake_apply_hidiffusion(model, **kwargs):
        applied_overrides.append((model, kwargs["t1_ratio"], kwargs["t2_ratio"]))

    with (
        patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion", side_effect=fake_apply_hidiffusion),
        patch("invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion"),
    ):
        first_patch = hidiffusion_patch(first_model, name_or_path="first", t1_ratio=0.2, t2_ratio=0.1)
        second_patch = hidiffusion_patch(second_model, name_or_path="second", t1_ratio=0.8, t2_ratio=0.9)
        first_patch.__enter__()
        second_patch.__enter__()
        first_patch.__exit__(None, None, None)
        second_patch.__exit__(None, None, None)

    assert applied_overrides == [(first_model, 0.2, 0.1), (second_model, 0.8, 0.9)]
    assert switching_threshold_ratio_dict == original_switching
    assert text_to_img_controlnet_switching_threshold_ratio_dict == original_controlnet


def test_hidiffusion_patch_forwards_generation_context():
    model = SimpleNamespace(unet=DummyUNet())

    with (
        patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion") as mock_apply_hidiffusion,
        patch("invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion"),
    ):
        with hidiffusion_patch(
            model,
            name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            is_inpainting_task=True,
        ):
            pass

    kwargs = mock_apply_hidiffusion.call_args.kwargs
    assert kwargs["is_inpainting_task"] is True


@pytest.mark.parametrize(
    ("size", "threshold", "override", "expected_ratio"),
    [
        ((256, 256), "T1_ratio", None, 0.4),
        ((384, 384), "T1_ratio", None, 0.4),
        ((512, 512), "T1_ratio", None, 0.7),
        ((384, 384), "T2_ratio", None, 0.0),
        ((512, 256), "T1_ratio", None, 0.4),
        ((384, 384), "T1_ratio", 0.25, 0.25),
    ],
)
def test_hidiffusion_ratios_use_upstream_discrete_presets(
    size: tuple[int, int], threshold: str, override: float | None, expected_ratio: float
):
    module = SimpleNamespace(
        model="sdxl",
        switching_threshold_ratio=threshold,
        info={
            "switching_threshold_overrides": {"T1_ratio": override, "T2_ratio": override},
            "text_to_img_controlnet": False,
        },
    )

    ratio = _get_resolution_aware_switching_threshold_ratio(module, *size)

    assert ratio == pytest.approx(expected_ratio)


def test_hidiffusion_controlnet_uses_its_normal_resolution_preset():
    module = SimpleNamespace(
        model="sdxl",
        switching_threshold_ratio="T1_ratio",
        info={
            "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": None},
            "text_to_img_controlnet": True,
        },
    )

    assert _get_resolution_aware_switching_threshold_ratio(module, 256, 256) == pytest.approx(0.5)
    assert _get_resolution_aware_switching_threshold_ratio(module, 512, 512) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("size", "threshold", "is_inpainting", "t2_override", "expected"),
    [
        ((256, 256), "T2_ratio", False, None, (0.0, 0, 8)),
        ((256, 256), "T1_ratio", False, None, (0.4, 8, 20)),
        ((256, 256), "T2_ratio", True, None, (0.0, 0, 0)),
        ((256, 256), "T1_ratio", True, None, (0.4, 0, 20)),
        ((512, 512), "T2_ratio", False, None, (0.3, 0, 15)),
        ((512, 512), "T1_ratio", False, None, (0.7, 0, 35)),
        ((256, 256), "T2_ratio", False, 0.1, (0.1, 0, 5)),
        ((256, 256), "T1_ratio", False, 0.1, (0.4, 5, 20)),
    ],
)
def test_hidiffusion_raunet_schedule_matches_upstream_stages(
    size: tuple[int, int],
    threshold: str,
    is_inpainting: bool,
    t2_override: float | None,
    expected: tuple[float, int, int],
):
    module = SimpleNamespace(
        model="sdxl",
        max_timestep=50,
        switching_threshold_ratio=threshold,
        info={
            "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": t2_override},
            "text_to_img_controlnet": False,
            "is_inpainting_task": is_inpainting,
            "is_playground": False,
        },
    )

    assert _get_raunet_step_range(module, *size) == expected


@pytest.mark.parametrize("t1_override", [None, 0.4])
def test_hidiffusion_rejects_t2_above_the_resolved_t1(t1_override: float | None):
    module = SimpleNamespace(
        model="sdxl",
        switching_threshold_ratio="T2_ratio",
        info={
            "switching_threshold_overrides": {"T1_ratio": t1_override, "T2_ratio": 0.5},
            "text_to_img_controlnet": False,
        },
    )

    with pytest.raises(ValueError, match="T2 ratio must be less than or equal to the T1 ratio"):
        _get_resolution_aware_switching_threshold_ratio(module, 256, 256)


def test_denoise_invocation_rejects_explicit_t2_above_t1():
    invocation = DenoiseLatentsInvocation.model_construct(hidiffusion_t1_ratio=0.4, hidiffusion_t2_ratio=0.5)

    with pytest.raises(ValueError, match="T2 ratio must be less than or equal to the T1 ratio"):
        invocation.validate_hidiffusion_ratio_order()


def test_logical_step_prevents_sequential_guidance_from_advancing_t2_twice():
    patched_conv = make_diffusers_downsampler_block(torch.nn.Conv2d)
    module = patched_conv(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
    module.info = {
        "size": (256, 256),
        "pipeline": SimpleNamespace(_num_timesteps=10),
        "text_to_img_controlnet": False,
        "is_inpainting_task": False,
        "is_playground": False,
        "step_index": 0,
        "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": 0.4},
    }
    module.model = "sdxl"
    module.switching_threshold_ratio = "T2_ratio"
    hidden_states = torch.ones(1, 1, 8, 8)

    negative = module(hidden_states)
    positive = module(hidden_states)

    assert negative.shape[-2:] == (2, 2)
    assert positive.shape[-2:] == (2, 2)
    assert module.timestep == 0

    module.info["step_index"] = 4
    after_t2 = module(hidden_states)
    assert after_t2.shape[-2:] == (4, 4)


def test_hidiffusion_extension_sets_logical_step_on_patched_unet():
    unet = SimpleNamespace(info={"step_index": None})
    ctx = SimpleNamespace(unet=unet, step_index=3)
    extension = HiDiffusionExt(name_or_path="runwayml/stable-diffusion-v1-5")

    extension.set_step_index(ctx)

    assert unet.info["step_index"] == 3


def test_t2i_adapter_residual_is_resized_for_active_raunet():
    patched_block = make_diffusers_cross_attn_down_block(CrossAttnDownBlock2D)
    module = patched_block(
        in_channels=4,
        out_channels=4,
        temb_channels=4,
        num_layers=2,
        resnet_groups=1,
        num_attention_heads=1,
        cross_attention_dim=4,
        add_downsample=False,
    )
    module.info = {
        "size": (64, 64),
        "pipeline": SimpleNamespace(_num_timesteps=10),
        "text_to_img_controlnet": False,
        "is_inpainting_task": False,
        "is_playground": False,
        "step_index": 0,
        "switching_threshold_overrides": {"T1_ratio": 1.0, "T2_ratio": 1.0},
    }
    module.model = "sd15"
    module.switching_threshold_ratio = "T2_ratio"

    hidden_states, output_states = module(
        hidden_states=torch.randn(1, 4, 8, 8),
        temb=torch.randn(1, 4),
        encoder_hidden_states=torch.randn(1, 2, 4),
        additional_residuals=torch.randn(1, 4, 8, 8),
    )

    assert hidden_states.shape[-2:] == (4, 4)
    assert output_states[-1].shape[-2:] == (4, 4)


def test_sdxl_primary_raunet_is_active_after_aggressive_stage_until_t1():
    patched_block = make_diffusers_cross_attn_down_block(CrossAttnDownBlock2D)
    module = patched_block(
        in_channels=4,
        out_channels=4,
        temb_channels=4,
        num_layers=2,
        resnet_groups=1,
        num_attention_heads=1,
        cross_attention_dim=4,
        add_downsample=False,
    )
    module.info = {
        "size": (256, 256),
        "pipeline": SimpleNamespace(_num_timesteps=50),
        "text_to_img_controlnet": False,
        "is_inpainting_task": False,
        "is_playground": False,
        "step_index": 0,
        "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": None},
    }
    module.model = "sdxl"
    module.switching_threshold_ratio = "T1_ratio"
    inputs = {
        "hidden_states": torch.randn(1, 4, 8, 8),
        "temb": torch.randn(1, 4),
        "encoder_hidden_states": torch.randn(1, 2, 4),
    }

    hidden_states, _ = module(**inputs)
    assert hidden_states.shape[-2:] == (8, 8)

    module.info["step_index"] = 8
    hidden_states, _ = module(**inputs)
    assert hidden_states.shape[-2:] == (4, 4)

    module.info["step_index"] = 20
    hidden_states, _ = module(**inputs)
    assert hidden_states.shape[-2:] == (8, 8)


def test_sdxl_additional_raunet_is_active_before_aggressive_boundary():
    patched_conv = make_diffusers_downsampler_block(torch.nn.Conv2d)
    module = patched_conv(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
    module.info = {
        "size": (256, 256),
        "pipeline": SimpleNamespace(_num_timesteps=50),
        "text_to_img_controlnet": False,
        "is_inpainting_task": False,
        "is_playground": False,
        "step_index": 0,
        "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": None},
    }
    module.model = "sdxl"
    module.switching_threshold_ratio = "T2_ratio"
    hidden_states = torch.ones(1, 1, 8, 8)

    assert module(hidden_states).shape[-2:] == (2, 2)

    module.info["step_index"] = 7
    assert module(hidden_states).shape[-2:] == (2, 2)

    module.info["step_index"] = 8
    assert module(hidden_states).shape[-2:] == (4, 4)


def test_sdxl_t2_override_controls_downsampler_at_2048_resolution():
    patched_conv = make_diffusers_downsampler_block(torch.nn.Conv2d)
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

    def run(t2_ratio: float) -> torch.Tensor:
        module = patched_conv(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        torch.nn.init.constant_(module.weight, 1.0)
        module.info = {
            "size": (256, 256),
            "pipeline": SimpleNamespace(_num_timesteps=30),
            "text_to_img_controlnet": False,
            "is_inpainting_task": False,
            "is_playground": False,
            "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": t2_ratio},
        }
        module.model = "sdxl"
        module.switching_threshold_ratio = "T2_ratio"
        return module(hidden_states)

    assert not torch.equal(run(0.0), run(0.4))


def test_sdxl_automatic_ratios_preserve_extreme_resolution_preset():
    patched_conv = make_diffusers_downsampler_block(torch.nn.Conv2d)
    module = patched_conv(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
    module.info = {
        "size": (512, 512),
        "pipeline": SimpleNamespace(_num_timesteps=30),
        "text_to_img_controlnet": False,
        "is_inpainting_task": False,
        "is_playground": False,
        "switching_threshold_overrides": {"T1_ratio": None, "T2_ratio": None},
    }
    module.model = "sdxl"
    module.switching_threshold_ratio = "T2_ratio"

    module(torch.ones(1, 1, 8, 8))

    assert module.T1_ratio == 0.3
    assert module.T1 == 9
