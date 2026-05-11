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


class TestAlignRefLatentDims:
    """Test reference latent dim alignment for 2x2 packing."""

    def test_even_dims_unchanged(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(96, 64) == (96, 64)

    def test_odd_dims_trimmed_to_even(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(97, 65) == (96, 64)
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(150, 151) == (150, 150)

    def test_minimum_aligned_dims(self):
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(2, 2) == (2, 2)
        assert QwenImageDenoiseInvocation._align_ref_latent_dims(3, 2) == (2, 2)

    def test_raises_on_zero_dim(self):
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(0, 64)
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(64, 0)

    def test_raises_on_one_dim(self):
        """A 1-pixel latent aligns to 0 and must be rejected."""
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(1, 64)
        with pytest.raises(ValueError, match="spatial dims must be >= 2"):
            QwenImageDenoiseInvocation._align_ref_latent_dims(64, 1)


class TestBuildImgShapes:
    """Test img_shapes construction. Regression test for the ghosting/doubling bug
    where ref and noisy segments shared identical spatial RoPE positions."""

    def test_txt2img_single_segment(self):
        """No reference latent → single segment for the noisy latent only."""
        result = QwenImageDenoiseInvocation._build_img_shapes(64, 64)
        assert result == [[(1, 32, 32)]]

    def test_edit_uses_distinct_ref_dims(self):
        """Edit-mode img_shapes must place ref segment at the ref's OWN dims, not
        the noisy dims. Identical dims caused the ghosting artifact."""
        noisy_h, noisy_w = 64, 64
        ref_h, ref_w = 96, 64
        result = QwenImageDenoiseInvocation._build_img_shapes(noisy_h, noisy_w, ref_h, ref_w)
        assert result == [[(1, 32, 32), (1, 48, 32)]]
        # The bug was that both segments had the same shape:
        assert result[0][0] != result[0][1]

    def test_edit_matches_diffusers_layout(self):
        """Structure must match diffusers QwenImageEditPipeline (single batch,
        nested list of (frame, h//2, w//2) tuples)."""
        result = QwenImageDenoiseInvocation._build_img_shapes(80, 112, 128, 96)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 2
        assert result[0][0] == (1, 40, 56)
        assert result[0][1] == (1, 64, 48)
