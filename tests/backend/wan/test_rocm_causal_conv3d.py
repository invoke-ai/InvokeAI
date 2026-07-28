"""Tests for the ROCm WanCausalConv3d conv2d decomposition.

The decomposition replaces MIOpen's Im3d2Col conv3d fallback (61% of Wan VAE
decode GPU time on RDNA3; ~48x slower than the decomposed path). These tests pin
that the decomposed forward is numerically equivalent to the stock diffusers
forward on CPU — including the causal feature-cache path — so the ROCm-gated
class patch can never change results, only speed.
"""

import pytest
import torch

from invokeai.backend.wan.rocm_causal_conv3d import (
    _decomposed_conv3d,
    _decomposed_forward,
    _patch_wan_causal_conv3d,
)

diffusers = pytest.importorskip("diffusers")
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d  # noqa: E402


@pytest.mark.parametrize(
    "kernel_size",
    [(3, 3, 3), (1, 1, 1), (3, 1, 1), (1, 3, 3)],
    ids=["3x3x3", "1x1x1", "temporal-only", "spatial-only"],
)
def test_decomposed_conv3d_matches_f_conv3d(kernel_size: tuple[int, int, int]) -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(6, 10, kernel_size=kernel_size)
    x = torch.randn(2, 6, 5, 12, 16)
    ref = torch.nn.functional.conv3d(x, conv.weight, conv.bias)
    got = _decomposed_conv3d(conv, x)
    assert torch.allclose(ref, got, atol=1e-5)


def test_decomposed_forward_matches_stock_wan_causal_conv3d() -> None:
    torch.manual_seed(1)
    conv = WanCausalConv3d(4, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 4, 5, 10, 14)
    ref = WanCausalConv3d.forward(conv, x)
    got = _decomposed_forward(conv, x)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5)


def test_decomposed_forward_matches_stock_with_feature_cache() -> None:
    """The VAE's frame-by-frame decode passes cached trailing frames as cache_x;
    the decomposition must reproduce the stock causal-cache arithmetic exactly."""
    torch.manual_seed(2)
    conv = WanCausalConv3d(4, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 4, 4, 10, 14)
    cache = torch.randn(1, 4, 2, 10, 14)
    ref = WanCausalConv3d.forward(conv, x, cache_x=cache)
    got = _decomposed_forward(conv, x, cache_x=cache)
    assert torch.allclose(ref, got, atol=1e-5)


def test_decomposed_forward_falls_back_to_conv3d_for_strided_convs() -> None:
    """Encoder downsample convs are strided; the temporal taps couple under stride,
    so those must go through F.conv3d untouched."""
    torch.manual_seed(3)
    conv = WanCausalConv3d(4, 8, kernel_size=3, stride=(1, 2, 2), padding=1)
    x = torch.randn(1, 4, 5, 12, 16)
    ref = WanCausalConv3d.forward(conv, x)
    got = _decomposed_forward(conv, x)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5)


def test_class_patch_is_idempotent_and_preserves_behavior() -> None:
    torch.manual_seed(4)
    stock_forward = WanCausalConv3d.forward
    try:
        conv = WanCausalConv3d(4, 8, kernel_size=3, padding=1)
        x = torch.randn(1, 4, 5, 10, 14)
        ref = conv(x)

        _patch_wan_causal_conv3d()
        patched_forward = WanCausalConv3d.forward
        _patch_wan_causal_conv3d()  # second call must be a no-op
        assert WanCausalConv3d.forward is patched_forward
        assert WanCausalConv3d.forward is not stock_forward

        assert torch.allclose(conv(x), ref, atol=1e-5)
    finally:
        WanCausalConv3d.forward = stock_forward
        if hasattr(WanCausalConv3d, "_invokeai_rocm_conv2d_decomposition"):
            delattr(WanCausalConv3d, "_invokeai_rocm_conv2d_decomposition")
