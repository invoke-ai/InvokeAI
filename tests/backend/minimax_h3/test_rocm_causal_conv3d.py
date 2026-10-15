"""Tests for the ROCm MiniMaxH3VideoCausalConv3d conv2d decomposition.

The decomposition replaces MIOpen's Im3d2Col conv3d fallback on RDNA3 (~48x
slower than the decomposed path, measured on the Wan VAE's identical failure
mode). These tests pin that the decomposed forward is numerically equivalent to
the stock vendored forward on CPU — including the reflect spatial padding and
causal temporal padding — so the ROCm-gated class patch can never change
results, only speed.
"""

import random

import pytest
import torch

import invokeai.backend.minimax_h3.rocm_causal_conv3d as mod
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d
from invokeai.backend.minimax_h3.rocm_causal_conv3d import (
    _decomposed_conv3d,
    _decomposed_forward,
    _patch_minimax_h3_causal_conv3d,
)


def _restore_stock_forward() -> None:
    """Put the class back to its unpatched state.

    The patch is process-wide and idempotent, and on a ROCm build it is real: any earlier test on the
    same xdist worker that loads the MiniMax H3 video VAE for real leaves the class patched. A test that then records the patched forward as "stock" cannot tell whether patching
    did anything, and fails only on ROCm rigs, only for some worker assignments.
    """
    if getattr(MiniMaxH3VideoCausalConv3d, mod._SENTINEL, False):
        assert mod._STOCK_FORWARD is not None
        MiniMaxH3VideoCausalConv3d.forward = mod._STOCK_FORWARD
        delattr(MiniMaxH3VideoCausalConv3d, mod._SENTINEL)


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
    _restore_stock_forward()
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
        _restore_stock_forward()


def test_patch_applies_on_every_hip_version(monkeypatch) -> None:
    """There is no HIP-version gate: MIOpen in rocm7.2 still takes the
    ~50x Im3d2Col fallback for this encoder's shapes (W7900: 208 s vs 3.6 s per chunk).
    Non-HIP builds are never patched; INVOKEAI_ROCM_CONV3D=native opts out."""
    import invokeai.backend.minimax_h3.rocm_causal_conv3d as mod

    calls: list[bool] = []
    monkeypatch.setattr(mod, "_patch_minimax_h3_causal_conv3d", lambda: calls.append(True))

    monkeypatch.setattr(torch.version, "hip", "7.2.53211")
    mod.patch_minimax_h3_causal_conv3d_for_rocm()
    assert calls == [True], "must decompose on HIP 7.2+"

    monkeypatch.setattr(torch.version, "hip", "7.1.25424")
    mod.patch_minimax_h3_causal_conv3d_for_rocm()
    assert calls == [True, True], "must decompose on HIP < 7.2"

    monkeypatch.setattr(torch.version, "hip", None)
    mod.patch_minimax_h3_causal_conv3d_for_rocm()
    assert calls == [True, True], "CUDA/CPU builds are never patched"

    monkeypatch.setattr(torch.version, "hip", "7.2.53211")
    monkeypatch.setattr(mod, "_MODE", "native")
    mod.patch_minimax_h3_causal_conv3d_for_rocm()
    assert calls == [True, True], "INVOKEAI_ROCM_CONV3D=native leaves the stock forward in place"


needs_rocm = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="needs a ROCm GPU")


@pytest.mark.slow
@needs_rocm
def test_decomposition_matches_native_on_device_under_allocation_churn() -> None:
    """Both device paths stay within bf16 noise of a CPU fp32 reference on a 17-frame encoder chunk
    at a non-square size while junk allocations come and go between calls. `slow`: a dev-machine
    lane, not CI."""
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("needs ~8 GiB free")
    torch.manual_seed(0)
    conv = MiniMaxH3VideoCausalConv3d(64, 64, kernel_size=3, spatial_padding=1, temporal_padding=2)
    x = torch.randn((1, 64, 17, 224, 384))
    with torch.inference_mode():
        reference = MiniMaxH3VideoCausalConv3d.forward(conv, x)
        conv = conv.to("cuda", torch.bfloat16)
        x = x.to("cuda", torch.bfloat16)
        rnd = random.Random(0)
        for _ in range(20):
            junk = [
                torch.empty(rnd.randint(1, 400) * 2**20, device="cuda", dtype=torch.uint8)
                for _ in range(rnd.randint(1, 4))
            ]
            for fn in (MiniMaxH3VideoCausalConv3d.forward, _decomposed_forward):
                err = (fn(conv, x).float().cpu() - reference).abs().max().item()
                assert err < 0.1, f"{fn.__name__}: max abs error {err:.4f} vs CPU fp32"
            del junk
            if rnd.random() < 0.5:
                torch.cuda.empty_cache()
