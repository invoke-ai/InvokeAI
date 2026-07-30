"""Dequantization for scaled-fp8 single-file checkpoints (ComfyUI "scaled fp8" and MXFP8).

A quantized layer stores an fp8 ``<name>.weight`` alongside a scale tensor. Dequantizing is
``weight * scale``, but three details make a naive implementation wrong, each in a different way:

**E8M0 (microscaling) scales are exponents, not multipliers.** MXFP8 checkpoints store the scale
as a ``uint8`` biased power-of-two exponent, so the multiplier is ``2 ** (scale - 127)``. Treating
the raw byte as the multiplier does not crash — it silently scales weights by ~10^6 (a real
Krea-2 MXFP8 transformer goes from std 0.03 to std 14000), so the model loads and generates
garbage. Distinguished here by dtype: an integer scale is an exponent, a float scale is a
multiplier. A linear scale is never stored as an integer.

**Block-wise scales.** The scale is often not a scalar. Microscaling uses one scale per block of
32 values along the last dim, so a ``(6144, 6144)`` weight carries a ``(6144, 192)`` scale. Plain
multiplication cannot broadcast that and raises "The size of tensor a (3072) must match the size
of tensor b (384)". The scale has to be expanded with ``repeat_interleave`` first.

**Dequantizing straight to the target dtype.** Going via ``.float()`` materialises a complete
4-byte/param copy of the model before a separate downcast pass, spiking peak host RAM to ~2x the
final bf16 size — 50 GB for a 12.5 GB fp8 Krea-2 transformer, which is unloadable on a 32 GB
machine. Multiplying in the target dtype keeps the dict at the bf16 model size plus one transient
tensor. fp8 has only 3 mantissa bits and bf16 shares float32's exponent range, so the bf16
multiply loses no meaningful precision.

This lives here rather than in a loader because the Krea-2 and Qwen-Image loaders both need it,
and two independent copies had already drifted: the Krea-2 copy lacked every fix above.

Two scale-key naming schemes are in the wild:
  - ``<path>.weight`` + ``<path>.weight_scale``  (FLUX, Z-Image, Krea-2 style)
  - ``<path>.weight`` + ``<path>.scale_weight``  (Qwen2.5-VL fp8_scaled style, which also emits
    ``<path>.scale_input`` for activation scaling that we discard)
"""

from collections.abc import Sequence
from typing import Any

import torch

from invokeai.backend.quantization.dequantize_common import (
    resolve_target_dtype,
    to_plain_tensor,
)

SCALE_SUFFIXES = (".weight_scale", ".scale_weight")

#: Exponent bias for OCP Microscaling E8M0 scale bytes.
E8M0_EXPONENT_BIAS = 127


def to_scale_multiplier(scale: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Resolve a stored scale tensor to the multiplier to apply to the weights.

    An integer scale is an E8M0 biased exponent (microscaling), so it becomes ``2 ** (s - 127)``.
    A float scale is already a linear multiplier and is only cast. ``exp2`` is evaluated in float32
    before the cast so the intermediate cannot overflow a narrow dtype; every result is an exact
    power of two and so is representable in bf16/fp16 without rounding.
    """
    if not scale.dtype.is_floating_point:
        return torch.exp2(scale.float() - E8M0_EXPONENT_BIAS).to(compute_dtype)

    return scale.to(compute_dtype)


def expand_scale_to_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Broadcast a possibly block-wise ``scale`` up to ``weight``'s shape.

    A scalar scale is returned untouched — it broadcasts natively. A block-wise scale is repeated
    along each mismatched dimension by the block size implied by the shape ratio (32 for MXFP8).
    """
    if scale.shape == weight.shape or scale.numel() == 1:
        return scale

    for dim in range(len(weight.shape)):
        if dim < len(scale.shape) and scale.shape[dim] != weight.shape[dim]:
            block_size = weight.shape[dim] // scale.shape[dim]
            if block_size > 1:
                scale = scale.repeat_interleave(block_size, dim=dim)

    return scale


def dequantize_scaled_fp8(
    sd: dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    storage_dtype: torch.dtype | None = None,
    skip_patterns: Sequence[str] = (),
) -> int:
    """Dequantize scaled-fp8 weights in ``sd`` in place; returns the number converted.

    Weights are dequantized directly to ``compute_dtype`` (see the module docstring on why not via
    float32), then stored at ``storage_dtype`` when one is given — see
    :func:`~invokeai.backend.quantization.dequantize_common.resolve_target_dtype`. Scale keys are
    removed, including orphans with no matching weight — nothing can be multiplied by those, and
    leaving them behind would inflate the caller's size accounting.
    """
    scale_keys = [k for k in sd if isinstance(k, str) and k.endswith(SCALE_SUFFIXES)]
    count = 0

    for scale_key in scale_keys:
        for suffix in SCALE_SUFFIXES:
            if scale_key.endswith(suffix):
                base = scale_key[: -len(suffix)]
                weight_key = base + ".weight"
                break

        if weight_key in sd:
            target_dtype = resolve_target_dtype(base, compute_dtype, storage_dtype, skip_patterns)
            weight = torch.as_tensor(to_plain_tensor(sd[weight_key])).to(compute_dtype)
            scale = to_scale_multiplier(torch.as_tensor(to_plain_tensor(sd[scale_key])), compute_dtype)
            sd[weight_key] = (weight * expand_scale_to_weight(weight, scale)).to(target_dtype)
            count += 1

        del sd[scale_key]

    return count
