"""Tests for the ROCm MiniMaxH3VideoCausalConv3d conv2d decomposition.

The decomposition replaces MIOpen's Im3d2Col conv3d fallback on RDNA3 (~48x
slower than the decomposed path, measured on the Wan VAE's identical failure
mode). These tests pin that the decomposed forward is numerically equivalent to
the stock vendored forward on CPU — including the reflect spatial padding and
causal temporal padding — so the ROCm-gated class patch can never change
results, only speed.
"""

import torch

from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d
from invokeai.backend.minimax_h3.rocm_causal_conv3d import (
    _decomposed_conv3d,
    _decomposed_forward,
    _patch_minimax_h3_causal_conv3d,
)


def test_decomposed_conv3d_matches_f_conv3d() -> None:
    torch.manual_seed(0)
    for kernel_size in [(3, 3, 3), (1, 1, 1), (3, 1, 1), (1, 3, 3)]:
        conv = torch.nn.Conv3d(6, 10, kernel_size=kernel_size)
        x = torch.randn(2, 6, 5, 12, 16)
        ref = torch.nn.functional.conv3d(x, conv.weight, conv.bias)
        got = _decomposed_conv3d(conv, x)
        assert torch.allclose(ref, got, atol=1e-5), f"mismatch for kernel {kernel_size}"


def test_decomposed_forward_matches_stock_causal_conv3d() -> None:
    """The workhorse encoder conv: 3x3x3, reflect spatial padding, causal temporal padding."""
    torch.manual_seed(1)
    conv = MiniMaxH3VideoCausalConv3d(4, 8, kernel_size=3, spatial_padding=1, temporal_padding=2)
    x = torch.randn(1, 4, 5, 10, 14)
    ref = MiniMaxH3VideoCausalConv3d.forward(conv, x)
    got = _decomposed_forward(conv, x)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5)


def test_decomposed_forward_matches_stock_pointwise_shortcut() -> None:
    """Resnet shortcut convs are 1x1x1 with no padding."""
    torch.manual_seed(2)
    conv = MiniMaxH3VideoCausalConv3d(4, 8, kernel_size=1)
    x = torch.randn(1, 4, 5, 10, 14)
    ref = MiniMaxH3VideoCausalConv3d.forward(conv, x)
    got = _decomposed_forward(conv, x)
    assert torch.allclose(ref, got, atol=1e-5)


def test_decomposed_forward_falls_back_to_conv3d_for_strided_convs() -> None:
    """Encoder downsample convs are strided; the temporal taps couple under stride,
    so those must go through F.conv3d untouched."""
    torch.manual_seed(3)
    conv = MiniMaxH3VideoCausalConv3d(4, 8, kernel_size=3, stride=(2, 2, 2), spatial_padding=1, temporal_padding=2)
    x = torch.randn(1, 4, 6, 12, 16)
    ref = MiniMaxH3VideoCausalConv3d.forward(conv, x)
    got = _decomposed_forward(conv, x)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5)


def test_class_patch_is_idempotent_and_preserves_behavior() -> None:
    torch.manual_seed(4)
    stock_forward = MiniMaxH3VideoCausalConv3d.forward
    try:
        conv = MiniMaxH3VideoCausalConv3d(4, 8, kernel_size=3, spatial_padding=1, temporal_padding=2)
        x = torch.randn(1, 4, 5, 10, 14)
        ref = conv(x)

        _patch_minimax_h3_causal_conv3d()
        patched_forward = MiniMaxH3VideoCausalConv3d.forward
        _patch_minimax_h3_causal_conv3d()  # second call must be a no-op
        assert MiniMaxH3VideoCausalConv3d.forward is patched_forward
        assert MiniMaxH3VideoCausalConv3d.forward is not stock_forward

        assert torch.allclose(conv(x), ref, atol=1e-5)
    finally:
        MiniMaxH3VideoCausalConv3d.forward = stock_forward
        if hasattr(MiniMaxH3VideoCausalConv3d, "_invokeai_rocm_conv2d_decomposition"):
            delattr(MiniMaxH3VideoCausalConv3d, "_invokeai_rocm_conv2d_decomposition")
