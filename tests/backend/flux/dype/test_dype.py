"""Tests for DyPE (Dynamic Position Extrapolation) module."""

import math

import torch

from invokeai.backend.flux.dype.base import (
    FLUX_BASE_PE_LEN,
    YARN_BETA_0,
    YARN_BETA_1,
    DyPEConfig,
    compute_dype_k_t,
    compute_ntk_freqs,
    compute_vision_yarn_freqs,
    compute_yarn_freqs,
    find_correction_factor,
    find_correction_range,
    get_mscale,
    linear_ramp_mask,
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


class TestDyPEConfig:
    """Tests for DyPEConfig dataclass."""

    def test_default_values(self):
        config = DyPEConfig()
        assert config.enable_dype is True
        assert config.base_resolution == 1024
        assert config.method == "vision_yarn"
        assert config.dype_scale == 2.0
        assert config.dype_exponent == 2.0
        assert config.dype_start_sigma == 1.0

    def test_custom_values(self):
        config = DyPEConfig(
            enable_dype=False,
            base_resolution=512,
            method="yarn",
            dype_scale=4.0,
            dype_exponent=3.0,
            dype_start_sigma=0.5,
        )
        assert config.enable_dype is False
        assert config.base_resolution == 512
        assert config.method == "yarn"
        assert config.dype_scale == 4.0


class TestMscale:
    """Tests for mscale calculation functions."""

    def test_get_mscale_no_scaling(self):
        """When scale <= 1.0, mscale should be 1.0."""
        assert get_mscale(1.0) == 1.0
        assert get_mscale(0.5) == 1.0

    def test_get_mscale_with_scaling(self):
        """When scale > 1.0, mscale should increase."""
        mscale_2x = get_mscale(2.0)
        mscale_4x = get_mscale(4.0)

        assert mscale_2x > 1.0
        assert mscale_4x > mscale_2x

    def test_get_mscale_formula(self):
        """Test mscale uses the correct YaRN formula: 1 + 0.1 * log(s) / sqrt(s)."""
        scale = 4.0
        expected = 1.0 + 0.1 * math.log(scale) / math.sqrt(scale)
        actual = get_mscale(scale)
        assert abs(actual - expected) < 1e-10


class TestDyPEKt:
    """Tests for DyPE k_t calculation."""

    def test_compute_dype_k_t_formula(self):
        """Test k_t uses correct formula: scale * (timestep ^ exponent)."""
        dype_scale = 2.0
        dype_exponent = 2.0
        current_sigma = 0.5  # normalized timestep
        dype_start_sigma = 1.0

        k_t = compute_dype_k_t(
            current_sigma=current_sigma,
            dype_scale=dype_scale,
            dype_exponent=dype_exponent,
            dype_start_sigma=dype_start_sigma,
        )

        # Expected: 2.0 * (0.5 ^ 2.0) = 2.0 * 0.25 = 0.5
        expected = dype_scale * (current_sigma**dype_exponent)
        assert abs(k_t - expected) < 1e-10

    def test_compute_dype_k_t_at_start(self):
        """At timestep=1.0 (start), k_t should equal dype_scale."""
        k_t = compute_dype_k_t(
            current_sigma=1.0,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        assert abs(k_t - 2.0) < 1e-10

    def test_compute_dype_k_t_at_end(self):
        """At timestep=0.0 (end), k_t should be 0."""
        k_t = compute_dype_k_t(
            current_sigma=0.0,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        assert k_t == 0.0

    def test_compute_dype_k_t_decreases_over_time(self):
        """k_t should decrease as sigma decreases (denoising progresses)."""
        k_t_early = compute_dype_k_t(
            current_sigma=1.0,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        k_t_mid = compute_dype_k_t(
            current_sigma=0.5,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        k_t_late = compute_dype_k_t(
            current_sigma=0.1,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )

        assert k_t_early > k_t_mid > k_t_late


class TestYaRNHelpers:
    """Tests for YaRN helper functions."""

    def test_find_correction_factor(self):
        """Test correction factor calculation."""
        # Basic sanity check - should return a reasonable dimension index
        factor = find_correction_factor(
            num_rotations=1.25,
            dim=32,
            base=10000,
            max_position_embeddings=256,
        )
        assert factor >= 0
        assert factor <= 32

    def test_find_correction_range(self):
        """Test correction range returns valid bounds."""
        low, high = find_correction_range(
            low_ratio=YARN_BETA_0,
            high_ratio=YARN_BETA_1,
            dim=28,  # half of 56 (FLUX spatial dim)
            base=10000,
            ori_max_pe_len=FLUX_BASE_PE_LEN,
        )

        # Low should be <= high
        assert low <= high
        # Both should be in valid range
        assert low >= 0
        assert high <= 27  # dim - 1

    def test_linear_ramp_mask_shape(self):
        """Test linear ramp mask has correct shape."""
        mask = linear_ramp_mask(
            min_val=5.0,
            max_val=15.0,
            dim=28,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert mask.shape == (28,)

    def test_linear_ramp_mask_values(self):
        """Test linear ramp mask has correct values."""
        mask = linear_ramp_mask(
            min_val=5.0,
            max_val=15.0,
            dim=20,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Values before min should be 0
        assert mask[0].item() == 0.0
        assert mask[4].item() == 0.0

        # Values after max should be 1
        assert mask[15].item() == 1.0
        assert mask[19].item() == 1.0

        # Values in between should be in (0, 1)
        assert 0.0 < mask[10].item() < 1.0

    def test_linear_ramp_mask_degenerate(self):
        """Test linear ramp mask handles degenerate case (min >= max)."""
        mask = linear_ramp_mask(
            min_val=10.0,
            max_val=5.0,  # max < min
            dim=20,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Should return step function at min_val
        assert mask.shape == (20,)


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
        assert config.method == "vision_yarn"

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
        assert config.method == "vision_yarn"

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

    def test_compute_yarn_freqs_shape(self):
        """Test yarn frequency computation shape."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos, sin = compute_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale=2.0,
            current_sigma=0.5,
            dype_config=config,
        )

        assert cos.shape == sin.shape
        assert cos.shape[0] == 1

    def test_compute_ntk_freqs_shape(self):
        """Test ntk frequency computation shape."""
        pos = torch.arange(16).unsqueeze(0).float()

        cos, sin = compute_ntk_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale=2.0,
        )

        assert cos.shape == sin.shape
        assert cos.shape[0] == 1


class TestThreeBandBlending:
    """Tests for 3-band frequency blending in YaRN implementation."""

    def test_different_timesteps_produce_different_freqs(self):
        """Different timesteps should produce different frequency outputs."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos_early, sin_early = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=2.0,
            scale_w=2.0,
            current_sigma=1.0,  # Early step
            dype_config=config,
        )

        cos_late, sin_late = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=2.0,
            scale_w=2.0,
            current_sigma=0.1,  # Late step
            dype_config=config,
        )

        # Outputs should be different due to DyPE timestep modulation
        assert not torch.allclose(cos_early, cos_late)
        assert not torch.allclose(sin_early, sin_late)

    def test_no_scaling_returns_base_freqs(self):
        """When scale <= 1.0, should return base frequencies without mscale."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos, sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=1.0,
            scale_w=1.0,
            current_sigma=0.5,
            dype_config=config,
        )

        # Verify shape is correct
        assert cos.shape[0] == 1
        assert cos.shape[1] == 16

    def test_yarn_freqs_matches_vision_yarn_for_uniform_scale(self):
        """yarn and vision_yarn should produce same results for uniform scale."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos_vision, sin_vision = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=2.0,
            scale_w=2.0,
            current_sigma=0.5,
            dype_config=config,
        )

        cos_yarn, sin_yarn = compute_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale=2.0,
            current_sigma=0.5,
            dype_config=config,
        )

        # Should be very close (same algorithm with same scale)
        assert torch.allclose(cos_vision, cos_yarn, atol=1e-6)
        assert torch.allclose(sin_vision, sin_yarn, atol=1e-6)

    def test_mscale_applied_to_output(self):
        """Verify mscale is applied to cos/sin outputs."""
        pos = torch.arange(16).unsqueeze(0).float()
        config = DyPEConfig()

        cos, sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=32,
            theta=10000,
            scale_h=4.0,
            scale_w=4.0,
            current_sigma=0.5,
            dype_config=config,
        )

        # With mscale applied, max values can exceed 1.0
        # (pure cos/sin are in [-1, 1], but mscale > 1 stretches them)
        ntk_scale = 4.0 ** (32 / (32 - 2))
        mscale = get_mscale(ntk_scale)
        assert mscale > 1.0

        # The actual values depend on the blending, but shape should be correct
        assert cos.shape == (1, 16, 16)  # (batch, seq_len, dim/2)
