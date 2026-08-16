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

The patch is class-level and idempotent, applied only when torch is a ROCm/HIP
build. It covers every ``AutoencoderKLMiniMaxH3`` consumer (keyframe
conditioning, latents-to-image/video encode paths) regardless of which loader
constructed it.
"""

import torch
import torch.nn.functional as F

_SENTINEL = "_invokeai_rocm_conv2d_decomposition"


def _decomposed_conv3d(module: torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """``F.conv3d(x, module.weight, module.bias)`` for stride-1/dilation-1/groups-1
    convs, computed as kT batched conv2d taps. ``x`` must already be padded."""
    b, c, t, h, w = x.shape
    k_t = module.weight.shape[2]
    t_out = t - k_t + 1
    out = None
    for k in range(k_t):
        xs = x[:, :, k : k + t_out].transpose(1, 2).reshape(b * t_out, c, h, w)
        o = F.conv2d(xs, module.weight[:, :, k], None)
        out = o if out is None else out + o
    assert out is not None
    if module.bias is not None:
        out = out + module.bias.view(1, -1, 1, 1)
    oh, ow = out.shape[-2:]
    return out.reshape(b, t_out, -1, oh, ow).transpose(1, 2)


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

    if getattr(MiniMaxH3VideoCausalConv3d, _SENTINEL, False):
        return
    MiniMaxH3VideoCausalConv3d.forward = _decomposed_forward
    setattr(MiniMaxH3VideoCausalConv3d, _SENTINEL, True)


def patch_minimax_h3_causal_conv3d_for_rocm() -> None:
    """Apply the conv2d decomposition on ROCm builds; no-op elsewhere.

    Call from any loader that constructs an ``AutoencoderKLMiniMaxH3``. cuDNN has
    real implicit-GEMM conv3d kernels, so CUDA builds keep the stock path.
    """
    if torch.version.hip is None:
        return
    _patch_minimax_h3_causal_conv3d()
