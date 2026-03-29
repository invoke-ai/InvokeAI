"""Tests for the Qwen Image denoise invocation."""

import math

import numpy as np
import pytest

from invokeai.app.invocations.qwen_image_denoise import QwenImageDenoiseInvocation


class TestPrepareCfgScale:
    """Test _prepare_cfg_scale utility method."""

    def test_scalar_cfg_scale(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=4.0)
        result = inv._prepare_cfg_scale(5)
        assert result == [4.0, 4.0, 4.0, 4.0, 4.0]

    def test_list_cfg_scale(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=[1.0, 2.0, 3.0])
        result = inv._prepare_cfg_scale(3)
        assert result == [1.0, 2.0, 3.0]

    def test_list_cfg_scale_length_mismatch(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale=[1.0, 2.0])
        with pytest.raises(AssertionError):
            inv._prepare_cfg_scale(3)

    def test_invalid_cfg_scale_type(self):
        inv = QwenImageDenoiseInvocation.model_construct(cfg_scale="invalid")
        with pytest.raises(ValueError, match="Invalid CFG scale type"):
            inv._prepare_cfg_scale(3)


class TestComputeSigmas:
    """Test _compute_sigmas schedule computation."""

    def test_default_schedule_basic(self):
        """Default schedule should produce N+1 values ending at 0."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=None)
        assert len(sigmas) == 5  # N+1 including terminal 0
        assert sigmas[-1] == 0.0
        assert sigmas[0] == 1.0  # First sigma is always 1.0

    def test_default_schedule_monotonically_decreasing(self):
        """Sigmas should decrease monotonically."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=10, shift_override=None)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], f"Sigma at {i} ({sigmas[i]}) not > sigma at {i+1} ({sigmas[i+1]})"

    def test_shift_override_changes_schedule(self):
        """A shift override should produce different sigma values than the default."""
        inv = QwenImageDenoiseInvocation.model_construct()
        default_sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=None)
        shifted_sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=3.0)
        # Both should have same length
        assert len(default_sigmas) == len(shifted_sigmas)
        # But different intermediate values (first and last are the same: 1.0 and 0.0)
        assert default_sigmas[0] == shifted_sigmas[0] == 1.0
        assert default_sigmas[-1] == shifted_sigmas[-1] == 0.0
        # At least one intermediate value should differ
        assert any(
            abs(d - s) > 1e-6 for d, s in zip(default_sigmas[1:-1], shifted_sigmas[1:-1])
        ), "Shift override should change intermediate sigma values"

    def test_shift_override_uses_log(self):
        """With shift_override=3.0, mu should be log(3)."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=3.0)
        # For shift=3, mu=log(3), the exponential shift formula gives:
        # sigma(t) = exp(mu) / (exp(mu) + (1/t - 1)) = 3 / (3 + (1/t - 1))
        # At t=0.75: sigma = 3 / (3 + 1/0.75 - 1) = 3 / (3 + 0.333) = 0.9
        assert abs(sigmas[1] - 0.9) < 0.01

    def test_shift_override_no_terminal_stretch(self):
        """With shift override, no terminal stretch should be applied."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=3.0)
        # Without terminal stretch, the last non-zero sigma should be 0.5
        # (from exp(log(3)) / (exp(log(3)) + (1/0.25 - 1)) = 3/(3+3) = 0.5)
        assert abs(sigmas[-2] - 0.5) < 0.01

    def test_default_schedule_has_terminal_stretch(self):
        """Default schedule should apply terminal stretch to 0.02."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas = inv._compute_sigmas(image_seq_len=4096, num_steps=30, shift_override=None)
        # The last non-zero sigma should be close to shift_terminal (0.02)
        assert abs(sigmas[-2] - 0.02) < 0.005

    def test_different_step_counts(self):
        """Different step counts should produce different schedule lengths."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas_4 = inv._compute_sigmas(image_seq_len=4096, num_steps=4, shift_override=None)
        sigmas_30 = inv._compute_sigmas(image_seq_len=4096, num_steps=30, shift_override=None)
        assert len(sigmas_4) == 5
        assert len(sigmas_30) == 31

    def test_image_seq_len_affects_mu(self):
        """Different image sequence lengths should produce different schedules (via dynamic mu)."""
        inv = QwenImageDenoiseInvocation.model_construct()
        sigmas_small = inv._compute_sigmas(image_seq_len=256, num_steps=4, shift_override=None)
        sigmas_large = inv._compute_sigmas(image_seq_len=8192, num_steps=4, shift_override=None)
        # Both same length but different values
        assert len(sigmas_small) == len(sigmas_large)
        assert any(abs(s - l) > 1e-6 for s, l in zip(sigmas_small[1:-1], sigmas_large[1:-1]))


class TestPackUnpackLatents:
    """Test latent packing and unpacking roundtrip."""

    def test_pack_unpack_roundtrip(self):
        """Packing then unpacking should restore the original tensor."""
        import torch

        latents = torch.randn(1, 16, 128, 128)
        packed = QwenImageDenoiseInvocation._pack_latents(latents, 1, 16, 128, 128)
        assert packed.shape == (1, 64 * 64, 64)  # (B, H/2*W/2, C*4)

        unpacked = QwenImageDenoiseInvocation._unpack_latents(packed, 128, 128)
        assert unpacked.shape == (1, 16, 128, 128)
        assert torch.allclose(latents, unpacked)

    def test_pack_shape(self):
        """Pack should produce the correct shape."""
        import torch

        latents = torch.randn(1, 16, 140, 118)
        packed = QwenImageDenoiseInvocation._pack_latents(latents, 1, 16, 140, 118)
        assert packed.shape == (1, 70 * 59, 64)

    def test_unpack_shape(self):
        """Unpack should produce the correct shape."""
        import torch

        packed = torch.randn(1, 70 * 59, 64)
        unpacked = QwenImageDenoiseInvocation._unpack_latents(packed, 140, 118)
        assert unpacked.shape == (1, 16, 140, 118)
