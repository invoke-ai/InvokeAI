"""ROCm workaround: decompose WanCausalConv3d into per-temporal-tap conv2d calls.

MIOpen (ROCm's cuDNN equivalent) has no implicit-GEMM 3D-convolution kernels for
the shapes used by the Wan VAE on RDNA3 — it falls back to ``Im3d2Col``, which
materializes every 3x3x3 patch into a matrix before a GEMM. MIOpen's *2D*
convolutions are well optimized, and a stride-1 kT x kH x kW conv3d is exactly
the sum of kT conv2d taps over shifted temporal slices, so this module rebinds
``WanCausalConv3d.forward`` to that decomposition.

The fallback is a size cliff, not a slope: a conv that emits a single output
frame (an image encode or decode) degenerates to a 2D conv and is fast natively,
and so are small canvases, but every multi-frame chunk at video size still takes
Im3d2Col on current MIOpen. Measured on a W7900 with torch 2.13.0+rocm7.2
(A14B VAE, bf16, 832x480), native vs decomposed: encode 17 frames 103 s vs 2.4 s,
decode 5 latent frames 162 s vs 3.6 s, peak VRAM 1-3 GiB lower decomposed. So
the decomposition applies on every HIP version; a single-frame or small-shape
timing cannot justify a gate (one shipped that way once and regressed every
I2V encode and video decode). ``tests/backend/wan/test_rocm_causal_conv3d.py``
has a ``slow`` ROCm test that fails when native conv3d stops taking the fallback.

Numerics: identical math up to floating-point summation order — max abs error vs
``F.conv3d`` is ~1e-6 in fp32; a full bf16 VAE decode differs from native by at
most ~3/255 in pixel space (bf16 accumulation noise, visually imperceptible).

The patch is class-level and idempotent, applied only when torch is a ROCm/HIP
build. It covers every ``AutoencoderKLWan`` consumer (Wan decode/encode nodes,
ref-image encoding, Anima's VAE) regardless of which loader constructed it.
"""

import os

import torch
import torch.nn.functional as F

from invokeai.backend.util.logging import InvokeAILogger

_SENTINEL = "_invokeai_rocm_conv2d_decomposition"


def _decomposed_conv3d(module: torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """``F.conv3d(x, module.weight, module.bias)`` for stride-1/dilation-1/groups-1
    convs, computed as kT batched conv2d taps. ``x`` must already be padded."""
    b, c, t, h, w = x.shape
    k_t = module.weight.shape[2]
    t_out = t - k_t + 1
    # Every tensor handed to MIOpen, and the activation handed back, is made standard-contiguous.
    # The natural expressions are legal strided views (a tap input whose channel stride exceeds
    # its batch stride at B=1, a temporal slice of the weight, a transposed result), and MIOpen in
    # torch 2.13.0+rocm7.2 faults on them: an A14B encode of 161 frames at 720x1024 aborts with a
    # GPU memory-access fault (or segfaults the process) every time with the views, never with the
    # copies. The fault is asynchronous -- HIP_LAUNCH_BLOCKING=1 hides it -- and needs a long clip,
    # so short repros pass. The copies cost a small fraction of the conv itself.
    out = None
    for k in range(k_t):
        xs = x[:, :, k : k + t_out].transpose(1, 2).reshape(b * t_out, c, h, w).contiguous()
        o = F.conv2d(xs, module.weight[:, :, k].contiguous(), None)
        del xs  # keep at most one tap input live alongside the accumulator
        out = o if out is None else out.add_(o)
    assert out is not None
    if module.bias is not None:
        out = out + module.bias.view(1, -1, 1, 1)
    oh, ow = out.shape[-2:]
    return out.reshape(b, t_out, -1, oh, ow).transpose(1, 2).contiguous()


def _decomposed_forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
    if _MODE == "verify":
        return _verified_forward(self, x, cache_x)
    # Causal-padding / feature-cache handling copied verbatim from
    # diffusers.models.autoencoders.autoencoder_kl_wan.WanCausalConv3d.forward.
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    if self.stride != (1, 1, 1) or self.dilation != (1, 1, 1) or self.groups != 1:
        # Not worth decomposing (and stride couples the temporal taps) — these
        # only occur on encoder downsample convs, which are a minority of calls.
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
    return _decomposed_conv3d(self, x)


# Diagnostic override for A/B checks against native MIOpen conv3d. Values:
#   decomposed (default) — the normal patched path.
#   native               — leave the stock conv3d forward in place (patch becomes a no-op).
#   verify               — run BOTH the stock forward and the decomposition for every call,
#                          log any call whose outputs diverge beyond bf16 accumulation noise
#                          (with its exact shape), and return the stock result. One decode in
#                          this mode pinpoints exactly which conv call (if any) corrupts.
_MODE = os.environ.get("INVOKEAI_ROCM_CONV3D", "decomposed").strip().lower()
_VERIFY_TOL = 0.1  # far above the measured bf16 accumulation noise (~0.03), far below corruption
_STOCK_FORWARD = None


def _verified_forward(self, x: torch.Tensor, cache_x: torch.Tensor | None) -> torch.Tensor:
    assert _STOCK_FORWARD is not None
    reference = _STOCK_FORWARD(self, x, cache_x)
    # Re-run the decomposed path's body directly (not via _decomposed_forward, which would
    # recurse back into this function while _MODE == "verify").
    padding = list(self._padding)
    x_d = x
    cache_d = cache_x
    if cache_d is not None and self._padding[4] > 0:
        cache_d = cache_d.to(x_d.device)
        x_d = torch.cat([cache_d, x_d], dim=2)
        padding[4] -= cache_d.shape[2]
    x_d = F.pad(x_d, padding)
    if self.stride != (1, 1, 1) or self.dilation != (1, 1, 1) or self.groups != 1:
        decomposed = F.conv3d(x_d, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
    else:
        decomposed = _decomposed_conv3d(self, x_d)
    max_diff = (reference.float() - decomposed.float()).abs().max().item()
    if max_diff > _VERIFY_TOL:
        InvokeAILogger.get_logger(__name__).warning(
            f"ROCm conv3d verify MISMATCH: max diff {max_diff:.4f} | input {tuple(x.shape)} "
            f"cache {tuple(cache_x.shape) if cache_x is not None else None} "
            f"weight {tuple(self.weight.shape)} dtype {x.dtype} "
            f"stride {self.stride} padding {tuple(self._padding)}"
        )
    return reference


def _patch_wan_causal_conv3d() -> None:
    """Rebind WanCausalConv3d.forward to the conv2d decomposition (idempotent)."""
    global _STOCK_FORWARD
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    if getattr(WanCausalConv3d, _SENTINEL, False):
        return
    _STOCK_FORWARD = WanCausalConv3d.forward
    WanCausalConv3d.forward = _decomposed_forward
    setattr(WanCausalConv3d, _SENTINEL, True)


def patch_wan_causal_conv3d_for_rocm() -> None:
    """Apply the conv2d decomposition on ROCm builds; no-op elsewhere.

    Call from any loader that constructs an ``AutoencoderKLWan``. cuDNN has real
    implicit-GEMM conv3d kernels, so CUDA builds keep the stock path. There is
    deliberately no HIP-version gate (see the module docstring). The row tearing once
    blamed on this decomposition on HIP 7.2 has two known causes, both since fixed: the
    ROCm fused-SDPA head-dim defect (``install_rocm_sdpa_guard`` in
    ``invokeai.backend.util.attention``; the Wan VAE mid-block attention is wider than its
    threshold) and MIOpen faulting on the strided views the decomposition used to hand it
    (see ``_decomposed_conv3d``). With both in place the decomposition matches native
    conv3d at non-square sizes under allocator churn. ``INVOKEAI_ROCM_CONV3D=native|verify``
    (``_MODE`` above) remain for A/B checks.
    """
    if torch.version.hip is None:
        return
    if _MODE == "native":
        return
    _patch_wan_causal_conv3d()
