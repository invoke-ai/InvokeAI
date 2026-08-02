import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.hidiffusion.hidiffusion import (
    remove_hidiffusion as real_remove_hidiffusion,
)
from invokeai.backend.hidiffusion.hidiffusion import (
    switching_threshold_ratio_dict,
    text_to_img_controlnet_switching_threshold_ratio_dict,
)
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
            model.block.aggressive_raunet = True
            model.block.T1_ratio = 0.9
            model.block.T1 = 9
            model.block.T1_start = 2
            model.block.T1_end = 8
            model.block.max_timestep = 99

        assert "timestep" not in model.block.__dict__

        with hidiffusion_patch(model, name_or_path="runwayml/stable-diffusion-v1-5"):
            assert model.block.timestep == 0
            assert model.block.aggressive_raunet is False
            assert model.block.T1_ratio == 0
            assert model.block.T1 == 0
            assert model.block.T1_start == 0
            assert model.block.T1_end == 0
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

    def fake_apply_hidiffusion(patched_model, **_kwargs):
        assert patched_model._name_or_path == "patched-model-name"
        assert patched_model.config._name_or_path == "patched-model-name"

        first_switching_entry = next(iter(switching_threshold_ratio_dict.values()))
        first_controlnet_entry = next(iter(text_to_img_controlnet_switching_threshold_ratio_dict.values()))
        assert first_switching_entry["T1_ratio"] == 0.25
        assert first_switching_entry["T2_ratio"] == 0.1
        assert first_controlnet_entry["T1_ratio"] == 0.25
        assert first_controlnet_entry["T2_ratio"] == 0.1

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
