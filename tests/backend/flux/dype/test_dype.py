"""Tests for DyPE (Dynamic Position Extrapolation) module."""

import torch

from invokeai.backend.flux.dype.base import (
    DyPEConfig,
    compute_vision_yarn_freqs,
    get_timestep_kappa,
)
from invokeai.backend.flux.dype.embed import DyPEEmbedND
from invokeai.backend.flux.dype.presets import (
    DYPE_PRESET_4K,
    DYPE_PRESET_AREA,
    DYPE_PRESET_AUTO,
    DYPE_PRESET_MANUAL,
    DYPE_PRESET_OFF,
    DYPE_PRESETS,
    get_dype_config_for_area,
    get_dype_config_for_resolution,
    get_dype_config_from_preset,
)
from invokeai.backend.flux.dype.rope import rope_dype
from invokeai.backend.flux.extensions.dype_extension import DyPEExtension


class TestDyPEConfig:
    """Tests for DyPEConfig dataclass."""

    def test_default_values(self):
        config = DyPEConfig()
        assert config.enable_dype is True
        assert config.base_resolution == 1024
        assert config.dype_scale == 2.0
        assert config.dype_exponent == 2.0
        assert config.dype_start_sigma == 1.0

    def test_custom_values(self):
        config = DyPEConfig(
            enable_dype=False,
            base_resolution=512,
            dype_scale=4.0,
            dype_exponent=3.0,
            dype_start_sigma=0.5,
        )
        assert config.enable_dype is False
        assert config.base_resolution == 512
        assert config.dype_scale == 4.0


class TestDyPEExtension:
    """Tests for DyPE extension helpers."""

    def test_resolve_step_sigma_prefers_scheduler_sigmas_tensor(self):
        sigma = DyPEExtension.resolve_step_sigma(
            fallback_sigma=0.42,
            step_index=1,
            scheduler_sigmas=torch.tensor([1.0, 0.75, 0.5]),
        )
        assert sigma == 0.75

    def test_resolve_step_sigma_falls_back_without_scheduler_sigmas(self):
        sigma = DyPEExtension.resolve_step_sigma(
            fallback_sigma=0.42,
            step_index=1,
            scheduler_sigmas=None,
        )
        assert sigma == 0.42


class TestKappa:
    """Tests for the DyPE timestep scheduler."""

    def test_get_timestep_kappa_clamps_to_zero_without_scale(self):
        assert (
            get_timestep_kappa(
                current_sigma=0.5,
                dype_scale=0.0,
                dype_exponent=2.0,
                dype_start_sigma=1.0,
            )
            == 0.0
        )

    def test_get_timestep_kappa_is_stronger_early(self):
        early_kappa = get_timestep_kappa(
            current_sigma=1.0,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        late_kappa = get_timestep_kappa(
            current_sigma=0.1,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )

        assert early_kappa == 2.0
        assert late_kappa < early_kappa

    def test_get_timestep_kappa_clamps_above_start_sigma(self):
        kappa = get_timestep_kappa(
            current_sigma=2.0,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        assert kappa == 2.0


class TestRopeDype:
    """Tests for DyPE-enhanced RoPE function."""

    def test_rope_dype_shape(self):
        """Test that rope_dype returns correct shape."""
        pos = torch.zeros(1, 64)
        dim = 64
        theta = 10000

        config = DyPEConfig()
        result = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=0.5,
            target_height=2048,
            target_width=2048,
            dype_config=config,
        )

        # Shape should be (batch, seq_len, dim/2, 2, 2)
        assert result.shape == (1, 64, dim // 2, 2, 2)

    def test_rope_dype_no_scaling(self):
        """When target is same as base, output should match base rope."""
        pos = torch.arange(16).unsqueeze(0).float()
        dim = 32
        theta = 10000

        config = DyPEConfig(base_resolution=1024)

        # No scaling needed
        result_no_scale = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=0.5,
            target_height=1024,
            target_width=1024,
            dype_config=config,
        )

        # With scaling
        result_with_scale = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=0.5,
            target_height=2048,
            target_width=2048,
            dype_config=config,
        )

        # Results should be different when scaling is applied
        assert not torch.allclose(result_no_scale, result_with_scale)

    def test_rope_dype_late_stage_moves_toward_base_rope(self):
        """Late-stage DyPE should be closer to base RoPE than early-stage DyPE."""
        pos = torch.arange(16).unsqueeze(0).float()
        dim = 32
        theta = 10000

        config = DyPEConfig(base_resolution=1024)

        base_result = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=1.0,
            target_height=1024,
            target_width=1024,
            dype_config=config,
        )
        early_result = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=1.0,
            target_height=2048,
            target_width=2048,
            dype_config=config,
        )
        late_result = rope_dype(
            pos=pos,
            dim=dim,
            theta=theta,
            current_sigma=0.05,
            target_height=2048,
            target_width=2048,
            dype_config=config,
        )

        early_delta = torch.mean(torch.abs(early_result - base_result))
        late_delta = torch.mean(torch.abs(late_result - base_result))

        assert late_delta < early_delta


class TestDyPEEmbedND:
    """Tests for DyPEEmbedND module."""

    def test_init(self):
        """Test DyPEEmbedND initialization."""
        config = DyPEConfig()
        embedder = DyPEEmbedND(
            dim=128,
            theta=10000,
            axes_dim=[16, 56, 56],
            dype_config=config,
        )

        assert embedder.dim == 128
        assert embedder.theta == 10000
        assert embedder.axes_dim == [16, 56, 56]

    def test_set_step_state(self):
        """Test step state update."""
        config = DyPEConfig()
        embedder = DyPEEmbedND(
            dim=128,
            theta=10000,
            axes_dim=[16, 56, 56],
            dype_config=config,
        )

        embedder.set_step_state(sigma=0.5, height=2048, width=2048)

        assert embedder._current_sigma == 0.5
        assert embedder._target_height == 2048
        assert embedder._target_width == 2048

    def test_forward_shape(self):
        """Test forward pass output shape."""
        config = DyPEConfig()
        embedder = DyPEEmbedND(
            dim=128,
            theta=10000,
            axes_dim=[16, 56, 56],
            dype_config=config,
        )

        # Create input ids tensor (batch=1, seq_len=64, n_axes=3)
        ids = torch.zeros(1, 64, 3)

        result = embedder(ids)

        # Output should have shape (batch, 1, seq_len, dim)
        # Actually the shape is (batch, 1, seq_len, dim/2, 2, 2) based on rope output
        assert result.dim() == 6
        assert result.shape[0] == 1  # batch
        assert result.shape[1] == 1  # unsqueeze
        assert result.shape[2] == 64  # seq_len


class TestDyPEPresets:
    """Tests for DyPE preset configurations."""

    def test_preset_4k_exists(self):
        """Test that 4K preset is defined."""
        assert DYPE_PRESET_4K in DYPE_PRESETS

    def test_get_dype_config_for_resolution_below_threshold(self):
        """When resolution is below threshold, should return None."""
        config = get_dype_config_for_resolution(
            width=1024,
            height=1024,
            activation_threshold=1536,
        )
        assert config is None

        config = get_dype_config_for_resolution(
            width=1536,
            height=1024,
            activation_threshold=1536,
        )
        assert config is None

    def test_get_dype_config_for_resolution_above_threshold(self):
        """When resolution is above threshold, should return config."""
        config = get_dype_config_for_resolution(
            width=2048,
            height=2048,
            activation_threshold=1536,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_for_resolution_dynamic_scale(self):
        """Higher resolution should result in higher dype_scale."""
        config_2k = get_dype_config_for_resolution(
            width=2048,
            height=2048,
            base_resolution=1024,
            activation_threshold=1536,
        )
        config_4k = get_dype_config_for_resolution(
            width=4096,
            height=4096,
            base_resolution=1024,
            activation_threshold=1536,
        )

        assert config_2k is not None
        assert config_4k is not None
        assert config_4k.dype_scale > config_2k.dype_scale

    def test_get_dype_config_for_area_below_threshold(self):
        """When area is below threshold area, should return None."""
        config = get_dype_config_for_area(
            width=1024,
            height=1024,
        )
        assert config is None

    def test_get_dype_config_for_area_above_threshold(self):
        """When area is above threshold area, should return config."""
        config = get_dype_config_for_area(
            width=2048,
            height=1536,
            base_resolution=1024,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_for_area_penalizes_extreme_aspect_ratios(self):
        balanced_extreme = get_dype_config_for_area(
            width=2304,
            height=1152,
            base_resolution=1024,
        )
        extreme = get_dype_config_for_area(
            width=2304,
            height=960,
            base_resolution=1024,
        )
        balanced_same_area = get_dype_config_for_area(
            width=2048,
            height=1080,
            base_resolution=1024,
        )

        assert balanced_extreme is not None
        assert extreme is not None
        assert balanced_same_area is not None
        assert extreme.dype_scale < balanced_extreme.dype_scale
        assert extreme.dype_scale < balanced_same_area.dype_scale

    def test_get_dype_config_for_area_is_closer_to_auto_strength(self):
        area = get_dype_config_for_area(
            width=1728,
            height=1152,
            base_resolution=1024,
        )
        auto = get_dype_config_for_resolution(
            width=1728,
            height=1152,
            base_resolution=1024,
            activation_threshold=1536,
        )

        assert area is not None
        assert auto is not None
        assert area.dype_scale > auto.dype_scale * 0.9
        assert area.dype_scale < auto.dype_scale * 1.1

    def test_get_dype_config_for_area_uses_higher_exponent_than_old_curve(self):
        config = get_dype_config_for_area(
            width=1536,
            height=1024,
            base_resolution=1024,
        )

        assert config is not None
        assert 1.25 <= config.dype_exponent <= 2.0

    def test_get_dype_config_from_preset_area(self):
        """Preset AREA should use area-based config."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_AREA,
            width=2048,
            height=1536,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_from_preset_off(self):
        """Preset OFF should return None."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_OFF,
            width=2048,
            height=2048,
        )
        assert config is None

    def test_get_dype_config_from_preset_auto(self):
        """Preset AUTO should use resolution-based config."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_AUTO,
            width=2048,
            height=2048,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_from_preset_4k(self):
        """Preset 4K should use 4K settings."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_4K,
            width=3840,
            height=2160,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_from_preset_manual_custom_overrides(self):
        """Custom scale/exponent should override defaults only with 'manual' preset."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_MANUAL,
            width=2048,
            height=2048,
            custom_scale=5.0,
            custom_exponent=10.0,
        )
        assert config is not None
        assert config.dype_scale == 5.0
        assert config.dype_exponent == 10.0

    def test_get_dype_config_from_preset_4k_ignores_custom(self):
        """4K preset should ignore custom scale/exponent values."""
        config = get_dype_config_from_preset(
            preset=DYPE_PRESET_4K,
            width=3840,
            height=2160,
            custom_scale=5.0,
            custom_exponent=10.0,
        )
        assert config is not None
        # Custom values should be ignored - preset values used instead
        assert config.dype_scale == 2.0  # 4K preset default
        assert config.dype_exponent == 2.0  # 4K preset default


class TestFrequencyComputation:
    """Tests for frequency computation functions."""

    def test_compute_vision_yarn_freqs_shape(self):
        """Test vision_yarn frequency computation shape."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos, sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=2.0,
            scale_w=2.0,
            current_sigma=0.5,
            dype_config=config,
        )

        assert cos.shape == sin.shape
        assert cos.shape[0] == 1  # batch
        assert cos.shape[1] == 16  # seq_len

    def test_compute_vision_yarn_freqs_reverts_to_base_rope_at_zero_sigma(self):
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        dy_cos, dy_sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=2.0,
            scale_w=2.0,
            current_sigma=0.0,
            dype_config=config,
        )
        base_cos, base_sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=1.0,
            scale_w=1.0,
            current_sigma=0.0,
            dype_config=config,
        )

        assert torch.allclose(dy_cos, base_cos)
        assert torch.allclose(dy_sin, base_sin)
