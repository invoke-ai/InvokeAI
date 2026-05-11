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


class TestMaybeClampRefLatentSize:
    """Test the diffusers-style VAE_IMAGE_SIZE clamp applied to reference latents
    before packing. This is defense-in-depth for backend callers (direct API,
    older graph JSON) that wire qwen_image_i2l without explicit width/height —
    without the clamp, the transformer receives an out-of-distribution sequence
    length and VRAM usage spikes on large reference images."""

    def test_in_budget_latent_unchanged(self):
        """A 1024² ref image → 128x128 latent → exactly the budget. Pass through."""
        import torch

        ref = torch.randn(1, 16, 128, 128)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 128, 128)
        assert result is ref  # identity, no copy

    def test_small_latent_unchanged(self):
        """A 512² ref → 64x64 latent (4x under budget). Pass through unchanged."""
        import torch

        ref = torch.randn(1, 16, 64, 64)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 64, 64)
        assert result is ref

    def test_native_resolution_landscape_clamped(self):
        """A native 1600x1200 image → 200x150 latents. Should clamp to the same
        dims diffusers produces (1184x896 pixels → 148x112 latents)."""
        import torch

        ref = torch.randn(1, 16, 150, 200)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 112, 148)

    def test_native_resolution_portrait_clamped(self):
        """1200x1600 → 150x200 latents → diffusers target 896x1184 → 112x148."""
        import torch

        ref = torch.randn(1, 16, 200, 150)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 148, 112)

    def test_huge_latent_clamped(self):
        """A 4096x4096 image → 512x512 latents (16x budget). Clamp to 128x128
        latents (= 1024² pixels), well within model's trained distribution."""
        import torch

        ref = torch.randn(1, 16, 512, 512)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        assert result.shape == (1, 16, 128, 128)

    def test_clamp_preserves_aspect_ratio_within_rounding(self):
        """Aspect ratio of the clamped latent should match the input to within
        the 32-pixel snapping granularity used by diffusers."""
        import torch

        # 1920x1080 (16:9, ~2M pixels)
        ref = torch.randn(1, 16, 135, 240)
        result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
        # diffusers: calculate_dimensions(1024², 16/9) → (1376, 768) px → (172, 96) latent
        assert result.shape == (1, 16, 96, 172)

    def test_clamp_output_is_packable(self):
        """The clamped latent must have even spatial dims (required by 2x2 packing)
        before _align_ref_latent_dims is called. Because the clamp snaps to 32px
        in pixel space and vae_scale_factor=8, every clamp output is a multiple
        of 4 in latent space (and therefore even)."""
        import torch

        for h, w in [(150, 200), (200, 150), (135, 240), (512, 512)]:
            ref = torch.randn(1, 16, h, w)
            result = QwenImageDenoiseInvocation._maybe_clamp_ref_latent_size(ref)
            _, _, rh, rw = result.shape
            assert rh % 2 == 0, f"clamp produced odd height {rh} for input ({h},{w})"
            assert rw % 2 == 0, f"clamp produced odd width {rw} for input ({h},{w})"


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
