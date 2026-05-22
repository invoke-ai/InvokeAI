"""Tests for the Wan 2.2 I2V reference-image VAE-latent encoder helper."""

from unittest.mock import MagicMock

import torch
from PIL import Image

from invokeai.backend.wan.extensions.wan_ref_image_extension import (
    encode_reference_image_to_condition,
    preprocess_reference_image,
)


def _make_fake_vae(z_dim: int = 16, spatial_scale: int = 8, temporal_scale: int = 4) -> MagicMock:
    """Stand-in for ``AutoencoderKLWan`` that returns deterministic latents.

    ``encode(pixel)`` returns a fake distribution whose ``sample()`` yields
    a tensor sized exactly as the real Wan VAE would: ``[B, z_dim, T_lat, H/8, W/8]``.
    """
    vae = MagicMock()

    # ``next(iter(vae.parameters())).dtype`` is queried; pin to float32.
    param = torch.zeros(1, dtype=torch.float32)
    vae.parameters = MagicMock(return_value=iter([param]))

    # Config carries per-channel normalisation stats.
    vae.config = MagicMock()
    vae.config.latents_mean = [0.0] * z_dim
    vae.config.latents_std = [1.0] * z_dim

    def fake_encode(pixel: torch.Tensor, return_dict: bool = False):
        b, _, t, h, w = pixel.shape
        t_lat = (t - 1) // temporal_scale + 1
        h_lat = h // spatial_scale
        w_lat = w // spatial_scale
        latents = torch.zeros(b, z_dim, t_lat, h_lat, w_lat, dtype=pixel.dtype)

        dist = MagicMock()
        dist.sample = MagicMock(return_value=latents)
        # The pipeline does ``vae.encode(...)[0]`` for non-dict returns.
        return (dist,) if return_dict is False else MagicMock(latent_dist=dist)

    vae.encode = fake_encode
    return vae


class TestPreprocess:
    def test_resize_to_target_dims(self):
        img = Image.new("RGB", (200, 300), (128, 128, 128))
        out = preprocess_reference_image(img, width=64, height=64)
        # Shape: [batch=1, channels=3, time=1, H, W]
        assert out.shape == (1, 3, 1, 64, 64)

    def test_normalised_to_minus_one_to_one(self):
        # Pure-grey image preprocessed should be exactly 0 (since 128/255*2 - 1 ≈ 0.004).
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        out = preprocess_reference_image(img, width=64, height=64)
        # White → 1.0
        assert torch.allclose(out, torch.ones_like(out), atol=1e-4)

        black = Image.new("RGB", (64, 64), (0, 0, 0))
        out_b = preprocess_reference_image(black, width=64, height=64)
        # Black → -1.0
        assert torch.allclose(out_b, -torch.ones_like(out_b), atol=1e-4)

    def test_rejects_non_multiple_of_8(self):
        img = Image.new("RGB", (100, 100))
        import pytest

        with pytest.raises(ValueError, match="multiples of 8"):
            preprocess_reference_image(img, width=65, height=64)


class TestEncodeReferenceImageToCondition:
    """The condition tensor must be 20-channel (4 mask + 16 image latents)
    and shaped for the denoise step's later concat with 16-ch noise latents."""

    def test_shape_at_64x64(self):
        img = Image.new("RGB", (64, 64))
        vae = _make_fake_vae()
        cond = encode_reference_image_to_condition(
            image=img, vae=vae, width=64, height=64, device=torch.device("cpu"), dtype=torch.float32
        )
        # [1, 20, 1, 8, 8] — 4-ch mask + 16-ch latents at H/8, W/8.
        assert cond.shape == (1, 20, 1, 8, 8)

    def test_shape_at_1024x1024(self):
        img = Image.new("RGB", (1024, 1024))
        vae = _make_fake_vae()
        cond = encode_reference_image_to_condition(
            image=img, vae=vae, width=1024, height=1024, device=torch.device("cpu"), dtype=torch.float32
        )
        # 1024/8 = 128 latent spatial dim.
        assert cond.shape == (1, 20, 1, 128, 128)

    def test_first_four_channels_are_all_ones_mask(self):
        img = Image.new("RGB", (64, 64))
        vae = _make_fake_vae()
        cond = encode_reference_image_to_condition(
            image=img, vae=vae, width=64, height=64, device=torch.device("cpu"), dtype=torch.float32
        )
        mask = cond[:, :4]
        # First-frame mask is all-ones at num_frames=1 (every position is the first frame).
        assert torch.equal(mask, torch.ones_like(mask))

    def test_returns_dtype(self):
        img = Image.new("RGB", (64, 64))
        vae = _make_fake_vae()
        cond = encode_reference_image_to_condition(
            image=img, vae=vae, width=64, height=64, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        assert cond.dtype == torch.bfloat16
