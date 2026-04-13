"""Tests for the Qwen Image denoise invocation."""

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
