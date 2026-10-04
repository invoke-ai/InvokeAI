"""ROCm workaround: decompose MiniMaxH3VideoCausalConv3d into per-temporal-tap conv2d calls.

MIOpen (ROCm's cuDNN equivalent) has no implicit-GEMM 3D-convolution kernels for
VAE-shaped workloads on RDNA3 — it falls back to ``Im3d2Col``, which materializes
every kT x kH x kW patch into a matrix before a GEMM. On a W7900 this measured
~48x slower than the equivalent conv2d work in the Wan VAE (see
``invokeai/backend/wan/rocm_causal_conv3d.py``, which this module mirrors).

In the MiniMax H3 video VAE only the ENCODER is convolutional
(``MiniMaxH3VideoCausalConv3d`` throughout); the decoder is a ViT and is not
affected. The encoder runs on every keyframe-conditioning encode (i2v first/last
frame), so the penalty is paid per generation, on a single-frame workload.
MIOpen's *2D* convolutions are well optimized, and a stride-1 kT x kH x kW conv3d
is exactly the sum of kT conv2d taps over shifted temporal slices, so this module
rebinds ``MiniMaxH3VideoCausalConv3d.forward`` to that decomposition.

Numerics: identical math up to floating-point summation order — max abs error vs
``F.conv3d`` is ~1e-6 in fp32.

Like the Wan twin, the decomposition stays on for every HIP version: measured on
a W7900 with torch 2.13.0+rocm7.2 (HIP 7.2.53211), one 17-frame 768x448 reference
chunk encodes in 208 s fp32 / 222 s under fp16 autocast on native MIOpen conv3d
(peak 9.2 GiB — the Im3d2Col column buffer), against 3.6 s / 2.7 s decomposed
(peak 6.7 GiB). Native conv3d is only fast when a conv emits a single output
frame, which never happens for these 17-frame chunks.

``INVOKEAI_ROCM_CONV3D=native`` (the Wan module's diagnostic override, shared)
leaves the stock forward in place on any HIP version, for A/B or if a future
MIOpen fixes the fallback.

The patch is class-level and idempotent, applied only when torch is a ROCm/HIP
build. It covers every ``AutoencoderKLMiniMaxH3`` consumer (keyframe
conditioning, reference conditioning, latents-to-image/video encode paths)
regardless of which loader constructed it.
"""

import os

import torch
import torch.nn.functional as F

_SENTINEL = "_invokeai_rocm_conv2d_decomposition"

_STOCK_FORWARD = None
"""The forward the patch replaced, kept so a test can restore the class to its unpatched state."""
_MODE = os.environ.get("INVOKEAI_ROCM_CONV3D", "decomposed").strip().lower()


def _decomposed_conv3d(module: torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """``F.conv3d(x, module.weight, module.bias)`` for stride-1/dilation-1/groups-1
    convs, computed as kT batched conv2d taps. ``x`` must already be padded."""
    b, c, t, h, w = x.shape
    k_t = module.weight.shape[2]
    t_out = t - k_t + 1
    # Every tensor handed to MIOpen, and the activation handed back, is made standard-contiguous:
    # the Wan twin's identical strided views made MIOpen in torch 2.13.0+rocm7.2 fault
    # asynchronously part-way through long clips (see `invokeai.backend.wan.rocm_causal_conv3d`).
    # The copies cost a small fraction of the conv itself.
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


def _decomposed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Padding handling copied verbatim from MiniMaxH3VideoCausalConv3d.forward
    # (spatial reflect pad, then causal constant temporal pad).
    if self.spatial_padding > 0:
        padding = self.spatial_padding
        hidden_states = F.pad(hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode)
    if self.temporal_padding > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
    if self.stride != (1, 1, 1) or self.dilation != (1, 1, 1) or self.groups != 1:
        # Not worth decomposing (and stride couples the temporal taps) — these
        # only occur on encoder downsample convs, which are a minority of calls.
        return F.conv3d(hidden_states, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
    return _decomposed_conv3d(self, hidden_states)


def _patch_minimax_h3_causal_conv3d() -> None:
    """Rebind MiniMaxH3VideoCausalConv3d.forward to the conv2d decomposition (idempotent)."""
    from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d

    global _STOCK_FORWARD

    if getattr(MiniMaxH3VideoCausalConv3d, _SENTINEL, False):
        return
    _STOCK_FORWARD = MiniMaxH3VideoCausalConv3d.forward
    MiniMaxH3VideoCausalConv3d.forward = _decomposed_forward
    setattr(MiniMaxH3VideoCausalConv3d, _SENTINEL, True)


def patch_minimax_h3_causal_conv3d_for_rocm() -> None:
    """Apply the conv2d decomposition on every ROCm build; no-op elsewhere.

    Call from any loader that constructs an ``AutoencoderKLMiniMaxH3``. cuDNN has
    real implicit-GEMM conv3d kernels, so CUDA builds keep the stock path.

    There is deliberately no HIP-version gate (see the module docstring): MIOpen in
    torch 2.13.0+rocm7.2 still takes the ~50x Im3d2Col fallback for this encoder's
    shapes. ``INVOKEAI_ROCM_CONV3D=native`` opts out for diagnosis.
    """
    if torch.version.hip is None:
        return
    if _MODE == "native":
        return
    _patch_minimax_h3_causal_conv3d()
