"""Tests for Wan 2.2 sampling utilities."""

import torch

from invokeai.backend.model_manager.taxonomy import WanVariantType
from invokeai.backend.wan.sampling_utils import (
    get_default_latent_channels,
    get_spatial_scale_factor,
    make_noise,
)


class TestVariantConstants:
    def test_a14b_uses_8x_spatial(self) -> None:
        assert get_spatial_scale_factor(WanVariantType.T2V_A14B) == 8

    def test_ti2v_5b_uses_16x_spatial(self) -> None:
        assert get_spatial_scale_factor(WanVariantType.TI2V_5B) == 16

    def test_a14b_default_channels(self) -> None:
        assert get_default_latent_channels(WanVariantType.T2V_A14B) == 16

    def test_ti2v_5b_default_channels(self) -> None:
        assert get_default_latent_channels(WanVariantType.TI2V_5B) == 48


class TestMakeNoise:
    def test_a14b_shape_at_1024(self) -> None:
        noise = make_noise(
            batch_size=1,
            latent_channels=16,
            height=1024,
            width=1024,
            spatial_scale_factor=8,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            seed=42,
        )
        assert noise.shape == (1, 16, 1, 128, 128)
        assert noise.dtype == torch.bfloat16

    def test_ti2v_shape_at_1024(self) -> None:
        noise = make_noise(
            batch_size=1,
            latent_channels=48,
            height=1024,
            width=1024,
            spatial_scale_factor=16,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            seed=42,
        )
        assert noise.shape == (1, 48, 1, 64, 64)

    def test_seed_is_deterministic(self) -> None:
        kwargs = dict(
            batch_size=1,
            latent_channels=16,
            height=256,
            width=256,
            spatial_scale_factor=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=123,
        )
        a = make_noise(**kwargs)
        b = make_noise(**kwargs)
        assert torch.allclose(a, b)

    def test_seed_changes_output(self) -> None:
        a = make_noise(
            batch_size=1, latent_channels=16, height=256, width=256, spatial_scale_factor=8,
            device=torch.device("cpu"), dtype=torch.float32, seed=1,
        )
        b = make_noise(
            batch_size=1, latent_channels=16, height=256, width=256, spatial_scale_factor=8,
            device=torch.device("cpu"), dtype=torch.float32, seed=2,
        )
        assert not torch.allclose(a, b)
