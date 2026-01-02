"""Tests for DyPE (Dynamic Position Extrapolation) module."""

import pytest
import torch

from invokeai.backend.flux.dype.base import (
    DyPEConfig,
    compute_ntk_freqs,
    compute_vision_yarn_freqs,
    compute_yarn_freqs,
    get_mscale,
    get_timestep_mscale,
)
from invokeai.backend.flux.dype.embed import DyPEEmbedND
from invokeai.backend.flux.dype.presets import (
    DYPE_PRESETS,
    DyPEPreset,
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

    def test_get_timestep_mscale_no_scaling(self):
        """When scale <= 1.0, timestep_mscale should be 1.0."""
        result = get_timestep_mscale(
            scale=1.0,
            current_sigma=0.5,
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        assert result == 1.0

    def test_get_timestep_mscale_high_sigma(self):
        """Early steps (high sigma) should have stronger scaling."""
        early_mscale = get_timestep_mscale(
            scale=2.0,
            current_sigma=1.0,  # Early step
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )
        late_mscale = get_timestep_mscale(
            scale=2.0,
            current_sigma=0.1,  # Late step
            dype_scale=2.0,
            dype_exponent=2.0,
            dype_start_sigma=1.0,
        )

        # Early steps should have larger mscale than late steps
        assert early_mscale >= late_mscale


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
        assert DyPEPreset.PRESET_4K in DYPE_PRESETS

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

    def test_get_dype_config_from_preset_off(self):
        """Preset OFF should return None."""
        config = get_dype_config_from_preset(
            preset=DyPEPreset.OFF,
            width=2048,
            height=2048,
        )
        assert config is None

    def test_get_dype_config_from_preset_auto(self):
        """Preset AUTO should use resolution-based config."""
        config = get_dype_config_from_preset(
            preset=DyPEPreset.AUTO,
            width=2048,
            height=2048,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_from_preset_4k(self):
        """Preset 4K should use 4K settings."""
        config = get_dype_config_from_preset(
            preset=DyPEPreset.PRESET_4K,
            width=3840,
            height=2160,
        )
        assert config is not None
        assert config.enable_dype is True

    def test_get_dype_config_from_preset_custom_overrides(self):
        """Custom scale/exponent should override preset values."""
        config = get_dype_config_from_preset(
            preset=DyPEPreset.PRESET_4K,
            width=3840,
            height=2160,
            custom_scale=5.0,
            custom_exponent=10.0,
        )
        assert config is not None
        assert config.dype_scale == 5.0
        assert config.dype_exponent == 10.0


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
