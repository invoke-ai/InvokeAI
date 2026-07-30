"""NVFP4 dequantization for single-file checkpoints.

NVFP4 is NVIDIA's 4-bit float format, as produced by TensorRT Model Optimizer and re-exported by
ComfyUI. Each quantized layer ships three tensors::

    <path>.weight          uint8          two packed FP4 values per byte, along the last dim
    <path>.weight_scale    float8_e4m3fn  one scale per block of 16 values along the last dim
    <path>.weight_scale_2  float32        one scalar scale for the whole tensor

Dequantizing is ``fp4_value * weight_scale * weight_scale_2``. Three details are easy to get wrong,
and only the first fails loudly:

**The weight is 4-bit packed, so its stored last dim is half the logical one.** A ``(6144, 6144)``
weight is stored as ``(6144, 3072)`` uint8. Passing that to ``load_state_dict`` unmodified produces
a wall of "copying a param with shape torch.Size([6144, 3072]) ... the shape in current model is
torch.Size([6144, 6144])" - one line per quantized layer, naming every tensor and diagnosing
nothing.

**The high nibble is the even element.** Byte ``j`` holds logical elements ``2j`` (high nibble) and
``2j + 1`` (low nibble), and consecutive elements are adjacent - not split into low-nibble and
high-nibble halves. Getting the nibble order backwards swaps every adjacent pair, which does not
crash and leaves the weight histogram identical, so it is only detectable by comparing against the
same weights quantized another way. Both facts were established against Krea-2 Turbo, which ships
an MXFP8 export of bit-identical weights (all 174 unquantized tensors compare equal):

===================  ==============  =================
nibble order         sign agreement  per-block cosine
===================  ==============  =================
high nibble first    100.00%         +0.996
low nibble first      49.95%         -0.004
===================  ==============  =================

The interleaved (rather than split-halves) layout is independently confirmed without any reference:
under it, 100% of scale blocks contain the maximum FP4 code, which is what a block scale of
``amax / 6`` guarantees. Under a split-halves layout only ~85% do.

**FP4 codes are not integers.** The three magnitude bits are E2M1 - two exponent, one mantissa - so
the representable magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6, not 0-7. Treating a code as its own
value distorts the top half of every block.

Note that ``.weight_scale`` is *also* the scale key used by ComfyUI "scaled fp8" (see
:mod:`.scaled_fp8`), so a loader handling both must test for NVFP4 first - the distinguishing key is
``.weight_scale_2``. Feeding an NVFP4 checkpoint to the scaled-fp8 path silently multiplies a packed
4-bit weight by an NVFP4 block scale and yields a half-width tensor.
"""

from collections.abc import Sequence
from typing import Any

import torch

from invokeai.backend.quantization.dequantize_common import (
    resolve_target_dtype,
    to_plain_tensor,
)

#: Values per NVFP4 block scale, along the last dim.
NVFP4_BLOCK_SIZE = 16

#: Distinguishes NVFP4 from ComfyUI "scaled fp8", which has no equivalent per-tensor scale.
GLOBAL_SCALE_SUFFIX = ".weight_scale_2"
BLOCK_SCALE_SUFFIX = ".weight_scale"

#: FP4 E2M1 magnitudes, indexed by the low 3 bits of a code.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: Largest representable FP4 magnitude; a block scale is chosen as ``amax / 6``.
FP4_E2M1_MAX = _E2M1_MAGNITUDES[-1]


def fp4_e2m1_lut(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build the 16-entry FP4 E2M1 code-to-value table (bit 3 is the sign, so 8-15 are negative)."""
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=dtype, device=device)
    return torch.cat((magnitudes, -magnitudes))


def is_nvfp4_state_dict(sd: dict[str, Any]) -> bool:
    """True if ``sd`` carries NVFP4 weights, identified by the per-tensor global-scale keys."""
    return any(isinstance(k, str) and k.endswith(GLOBAL_SCALE_SUFFIX) for k in sd)


def unpack_fp4_e2m1(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Expand packed FP4 pairs into values, doubling the last dim.

    The high nibble of each byte is the even element and the low nibble the odd one - see the module
    docstring for how that was established. Written into a preallocated buffer rather than built with
    ``stack`` so only one half-width intermediate exists at a time; these tensors reach 16384 columns.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"NVFP4 packed weights must be uint8, got {packed.dtype}.")

    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=lut.dtype,
        device=packed.device,
    )
    unpacked[..., 0::2] = lut[(packed >> 4).long()]
    unpacked[..., 1::2] = lut[(packed & 0x0F).long()]
    return unpacked


def expand_block_scale(block_scale: torch.Tensor, logical_last_dim: int, *, name: str) -> torch.Tensor:
    """Broadcast a per-block scale up to one entry per weight element along the last dim."""
    blocks = block_scale.shape[-1]
    if blocks == logical_last_dim:
        return block_scale

    block_size, remainder = divmod(logical_last_dim, blocks)
    if remainder or block_size < 1:
        raise ValueError(
            f"{name}: NVFP4 block scale has {blocks} blocks along the last dim, which does not "
            f"evenly divide the {logical_last_dim} unpacked weight columns (expected one scale per "
            f"{NVFP4_BLOCK_SIZE} values)."
        )
    return block_scale.repeat_interleave(block_size, dim=-1)


def dequantize_nvfp4_tensor(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    compute_dtype: torch.dtype,
    *,
    name: str = "weight",
) -> torch.Tensor:
    """Dequantize one NVFP4 weight to ``compute_dtype``."""
    lut = fp4_e2m1_lut(compute_dtype, packed.device)
    values = unpack_fp4_e2m1(packed, lut)

    # Fold the global scale into the (16x smaller) block scale in float32 before downcasting: the
    # product is what actually multiplies the weights, and bf16 has 8 mantissa bits, so scaling
    # twice in bf16 rounds twice for no benefit.
    scale = (block_scale.float() * global_scale.float()).to(compute_dtype)
    scale = expand_block_scale(scale, values.shape[-1], name=name)

    values *= scale
    return values


def dequantize_nvfp4(
    sd: dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    storage_dtype: torch.dtype | None = None,
    skip_patterns: Sequence[str] = (),
) -> int:
    """Dequantize NVFP4 weights in ``sd`` in place; returns the number converted.

    Scale keys are removed, including orphans with no matching weight - nothing can be multiplied by
    those, and leaving them behind would inflate the caller's size accounting and reach
    ``load_state_dict`` as unexpected keys. See :func:`resolve_target_dtype` for ``storage_dtype``
    and ``skip_patterns``.
    """
    global_scale_keys = [k for k in sd if isinstance(k, str) and k.endswith(GLOBAL_SCALE_SUFFIX)]
    count = 0

    for global_scale_key in global_scale_keys:
        base = global_scale_key[: -len(GLOBAL_SCALE_SUFFIX)]
        weight_key = base + ".weight"
        block_scale_key = base + BLOCK_SCALE_SUFFIX

        if weight_key in sd and block_scale_key in sd:
            target_dtype = resolve_target_dtype(base, compute_dtype, storage_dtype, skip_patterns)
            sd[weight_key] = dequantize_nvfp4_tensor(
                to_plain_tensor(sd[weight_key]),
                to_plain_tensor(sd[block_scale_key]),
                to_plain_tensor(sd[global_scale_key]),
                compute_dtype,
                name=weight_key,
            ).to(target_dtype)
            count += 1

        sd.pop(block_scale_key, None)
        del sd[global_scale_key]

    return count
