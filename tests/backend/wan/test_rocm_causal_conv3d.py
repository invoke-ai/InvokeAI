"""Tests for the ROCm WanCausalConv3d conv2d decomposition.

The decomposition replaces MIOpen's Im3d2Col conv3d fallback (61% of Wan VAE
decode GPU time on RDNA3; ~48x slower than the decomposed path). These tests pin
that the decomposed forward is numerically equivalent to the stock diffusers
forward on CPU — including the causal feature-cache path — so the ROCm-gated
class patch can never change results, only speed.
"""

import random
import time

import pytest
import torch

import invokeai.backend.wan.rocm_causal_conv3d as mod
from invokeai.backend.wan.rocm_causal_conv3d import (
    _decomposed_conv3d,
    _decomposed_forward,
    _patch_wan_causal_conv3d,
)

diffusers = pytest.importorskip("diffusers")
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d  # noqa: E402


def _restore_stock_forward() -> None:
    """Put the class back to its unpatched state.

    The patch is process-wide and idempotent, and on a ROCm build it is real: any earlier test on the
    same xdist worker that loads a Wan-family VAE for real (the loader matrix does) leaves the class
    patched. A test that then records the patched forward as "stock" cannot tell whether patching
    did anything, and fails only on ROCm rigs, only for some worker assignments.
    """
    if getattr(WanCausalConv3d, mod._SENTINEL, False):
        assert mod._STOCK_FORWARD is not None
        WanCausalConv3d.forward = mod._STOCK_FORWARD
        delattr(WanCausalConv3d, mod._SENTINEL)


@pytest.fixture(autouse=True)
def _decomposed_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's INVOKEAI_ROCM_CONV3D=verify shell must not reroute the parity tests."""
    monkeypatch.setattr(mod, "_MODE", "decomposed")


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
    _restore_stock_forward()
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
        _restore_stock_forward()


def test_patch_applies_on_every_hip_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ROCm build decomposes, HIP 7.2+ included: native conv3d there is only fast when a
    conv emits a single output frame, and multi-frame encodes/decodes still take the Im3d2Col
    fallback (a HIP >= 7.2 gate shipped once on a single-frame timing and regressed video
    encode/decode). CUDA/CPU builds are never patched; the diagnostic modes still work."""
    calls: list[bool] = []
    monkeypatch.setattr(mod, "_patch_wan_causal_conv3d", lambda: calls.append(True))

    # No version parser exists any more: any HIP build string patches.
    for hip in ("7.1.25424", "7.2.10101", "10.0.999"):
        calls.clear()
        monkeypatch.setattr(torch.version, "hip", hip)
        mod.patch_wan_causal_conv3d_for_rocm()
        assert calls == [True], f"must decompose on HIP {hip}"

    calls.clear()
    monkeypatch.setattr(torch.version, "hip", None)
    mod.patch_wan_causal_conv3d_for_rocm()
    assert calls == [], "CUDA/CPU builds are never patched"

    monkeypatch.setattr(torch.version, "hip", "7.2.10101")
    monkeypatch.setattr(mod, "_MODE", "verify")
    mod.patch_wan_causal_conv3d_for_rocm()
    assert calls == [True], "verify mode patches (diagnostics must be able to run)"

    calls.clear()
    monkeypatch.setattr(mod, "_MODE", "native")
    mod.patch_wan_causal_conv3d_for_rocm()
    assert calls == [], "native mode never patches"


needs_rocm = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="needs a ROCm GPU")


@pytest.mark.slow
@needs_rocm
class TestOnRocmHardware:
    """The reason the decomposition exists, on the real device. `slow`: a dev-machine lane, not CI."""

    # One 4-frame encoder chunk (plus its first frame) at 832x480, base_dim channels: the smallest
    # production shape past MIOpen's Im3d2Col cliff. Sub-production shapes run ~2x either way and
    # would rationalize a gate, which is exactly how the HIP >= 7.2 regression shipped.
    SHAPE = (1, 96, 5, 480, 832)
    NEEDS_FREE_BYTES = 14 * 2**30  # native peaks ~10-12 GiB on the column buffer

    @staticmethod
    def _timed(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def _conv_and_input(self) -> tuple[WanCausalConv3d, torch.Tensor]:
        torch.cuda.empty_cache()  # a previous test's cached column buffer must not force a skip
        if torch.cuda.mem_get_info()[0] < self.NEEDS_FREE_BYTES:
            pytest.skip("native conv3d's Im3d2Col column buffer needs ~14 GiB free")
        torch.manual_seed(0)
        conv = WanCausalConv3d(96, 96, kernel_size=3, padding=1).to("cuda", torch.bfloat16)
        return conv, torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16)

    def test_native_conv3d_still_takes_the_multiframe_fallback(self) -> None:
        """Pins why the patch has no HIP-version gate. When this passes on a new torch, the
        decomposition may be unnecessary."""
        conv, x = self._conv_and_input()
        stock = mod._STOCK_FORWARD or WanCausalConv3d.forward
        with torch.inference_mode():
            for fn in (stock, _decomposed_forward):
                fn(conv, x)  # warm-up: MIOpen find + kernel compile
            native = self._timed(lambda: stock(conv, x))
            decomposed = self._timed(lambda: _decomposed_forward(conv, x))
        # Measured 128x on a W7900 with torch 2.13.0+rocm7.2; 10x leaves room for clock/contention noise.
        assert native / decomposed >= 10, (
            f"native conv3d is now within {native / decomposed:.1f}x of the decomposition for multi-frame "
            "chunks: the Wan decomposition may be unnecessary"
        )

    def test_long_clip_encode_completes(self) -> None:
        """Pins the contiguous copies in `_decomposed_conv3d`. With the natural strided views,
        MIOpen faults asynchronously part-way through a long clip (161 frames at 720x1024 aborted
        or segfaulted the process every time on torch 2.13.0+rocm7.2; 81 frames passed) -- so a
        fault here kills the pytest process rather than failing an assertion. Weights are random:
        the fault does not depend on them."""
        from diffusers.models.autoencoders import AutoencoderKLWan

        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            pytest.skip("the 161-frame encode peaks ~9 GiB")
        stock_forward = WanCausalConv3d.forward
        already_patched = getattr(WanCausalConv3d, mod._SENTINEL, False)
        _patch_wan_causal_conv3d()
        try:
            torch.manual_seed(0)
            vae = AutoencoderKLWan(z_dim=16).to("cuda", torch.bfloat16).eval()
            clip = torch.zeros((1, 3, 161, 1024, 720), device="cuda", dtype=torch.bfloat16)
            clip[:, :, 0] = torch.rand((1, 3, 1024, 720), device="cuda", dtype=torch.bfloat16) * 2 - 1
            with torch.inference_mode():
                latents = vae.encode(clip, return_dict=False)[0].mode()
            torch.cuda.synchronize()
        finally:
            if not already_patched:
                WanCausalConv3d.forward = stock_forward
                delattr(WanCausalConv3d, mod._SENTINEL)
        assert latents.shape == (1, 16, 41, 128, 90)
        assert torch.isfinite(latents).all()

    def test_decomposition_matches_native_under_allocation_churn(self) -> None:
        """The decomposition's outputs must not depend on allocator state (the failure class once
        blamed on it): both device paths stay within bf16 noise of a CPU fp32 reference while
        junk allocations come and go between calls."""
        conv, x = self._conv_and_input()
        x = x[:, :, :, :240, :416]  # non-square, past the cache path, cheap enough for many calls
        cache = torch.randn((1, 96, 2, 240, 416), device="cuda", dtype=torch.bfloat16)
        stock = mod._STOCK_FORWARD or WanCausalConv3d.forward
        with torch.inference_mode():
            reference = WanCausalConv3d.forward(conv.float().cpu(), x.float().cpu(), cache_x=cache.float().cpu())
            conv = conv.to("cuda", torch.bfloat16)
            rnd = random.Random(0)
            for _ in range(20):
                junk = [
                    torch.empty(rnd.randint(1, 400) * 2**20, device="cuda", dtype=torch.uint8)
                    for _ in range(rnd.randint(1, 4))
                ]
                for fn in (stock, _decomposed_forward):
                    out = fn(conv, x, cache_x=cache).float().cpu()
                    err = (out - reference).abs().max().item()
                    assert err < mod._VERIFY_TOL, f"{fn.__name__}: max abs error {err:.4f} vs CPU fp32"
                del junk
                if rnd.random() < 0.5:
                    torch.cuda.empty_cache()
