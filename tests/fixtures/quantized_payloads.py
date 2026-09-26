"""Checkpoint payloads in the shapes the producers actually write.

Every loader test that wants a quantized layer has so far built one itself: five copies of the
convrot quantizer, five nvfp4 packers, six spellings of the fp8 scale, thirteen marker blobs. They
agree today, which is the problem -- they agree by having been copied, so a producer detail that is
wrong is wrong in all of them at once, and a test asserting against its own copy of the encoding
asserts nothing about the encoding.

These are mirrors of what ComfyUI's exporters emit, and each docstring says which part of the
producer it is mirroring. Nothing here imports a loader: a payload is what arrives on disk, and the
question of what a loader does with it belongs to the caller.
"""

import json
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from invokeai.backend.quantization.block_scale_tiles import check_tile_layout
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, build_regular_hadamard

# e4m3's largest finite magnitude. A per-tensor scaled-fp8 export divides by this, so that the
# largest weight lands on the top code rather than at infinity.
FP8_E4M3_MAX = 448.0

# int8 is quantized against 127, not 128: the exporters keep the range symmetric and leave -128
# unused, so a scale derived from 128 is off by one level on every weight.
INT8_LEVELS = 127.0

# The two nvfp4 codes worth +1.0 and -1.0 in E2M1. Paired with a block scale and a global scale they
# give a weight whose exact value is known without depending on the rest of the E2M1 table.
NVFP4_CODE_PLUS_ONE = 2
NVFP4_CODE_MINUS_ONE = 10

# One MX block is 32 elements wide in every published build; the marker only cross-checks it.
MX_BLOCK_SIZE = 32


def comfy_quant_marker(marker: Mapping[str, Any], *, pad: int = 0) -> torch.Tensor:
    """The per-tensor ``.comfy_quant`` blob: UTF-8 JSON in a ``uint8`` tensor.

    ``pad`` appends NUL bytes, which is what Comfy does when it writes these at a fixed width. The
    clone gives the tensor its own storage instead of a view onto the buffer built here, which is
    what keeps a payload independent of the builder that made it.
    """
    raw = json.dumps(dict(marker)).encode("utf-8") + b"\x00" * pad
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


class Int8ConvrotPayload(NamedTuple):
    """What an ``int8_tensorwise`` + ``convrot`` layer stores, and what it must come back as.

    ``dequantized`` is the *round trip*, not the weight that went in: rounding to 127 levels is
    lossy, so the original is something no correct loader can produce and comparing against it
    measures the quantizer's error rather than the loader's.
    """

    codes: torch.Tensor
    scale: torch.Tensor
    dequantized: torch.Tensor


def quantize_convrot(weight: torch.Tensor, *, group_size: int = CONVROT_GROUP_SIZE) -> Int8ConvrotPayload:
    """Mirror of comfy-quants: rotate along the input dim in groups, then per-output-channel int8.

    The rotation is what makes this scheme different from plain int8, and it is also what a loader
    can silently skip: applying the scale without un-rotating gives a weight of the right shape and
    magnitude that correlates with the original at roughly nothing.
    """
    out_features, in_features = weight.shape
    hadamard = build_regular_hadamard(group_size, dtype=weight.dtype)
    grouped = weight.view(out_features, in_features // group_size, group_size)
    rotated = (grouped @ hadamard.T).view(out_features, in_features)

    scale = rotated.abs().amax(dim=1, keepdim=True) / INT8_LEVELS
    codes = torch.clamp(torch.round(rotated / scale), -128, 127).to(torch.int8)

    restored = (codes.float() * scale).view(out_features, in_features // group_size, group_size) @ hadamard
    return Int8ConvrotPayload(codes, scale.to(torch.float32), restored.view(out_features, in_features))


class ScaledFp8Payload(NamedTuple):
    """What a per-tensor scaled-fp8 layer stores, and the weight it decodes back to.

    ``dequantized`` is the round trip rather than the input, for the same reason as
    :class:`Int8ConvrotPayload`: e4m3 has three mantissa bits, so the original is not something a
    correct loader can return.
    """

    codes: torch.Tensor
    scale: torch.Tensor
    dequantized: torch.Tensor


def quantize_scaled_fp8(weight: torch.Tensor) -> ScaledFp8Payload:
    """Mirror of ComfyUI's per-tensor "scaled fp8": one float32 scalar beside an e4m3 weight.

    The scale is stored as the multiplier the loader applies, so the codes are the weight *divided*
    by it. Getting that direction wrong survives every shape check and every dtype check.
    """
    scale = (weight.abs().max() / FP8_E4M3_MAX).to(torch.float32)
    codes = (weight / scale).to(torch.float8_e4m3fn)
    return ScaledFp8Payload(codes, scale, codes.float() * scale)


def nvfp4_codes(positive: torch.Tensor) -> torch.Tensor:
    """Codes for a layer of exactly +1.0 and -1.0 in E2M1, one per element of ``positive``."""
    return torch.where(positive, NVFP4_CODE_PLUS_ONE, NVFP4_CODE_MINUS_ONE).to(torch.uint8)


def nvfp4_tensors(
    path: str, codes: torch.Tensor, *, block_scale: float | torch.Tensor, global_scale: float
) -> dict[str, torch.Tensor]:
    """One nvfp4 layer in checkpoint layout: packed nibbles, a block-scale grid, a global scalar.

    Two 4-bit codes share a byte, high nibble first, so the stored weight is half as wide as the
    logical one. A number fills the grid uniformly; pass a grid to vary it. Either way the grid has
    to *be* a whole number of cuBLAS tiles, so the smallest layer this can build is 128 rows by 64
    columns -- `NVFP4Linear` refuses anything else rather than padding it.

    No expected weight comes back, because stating one means restating the tile permutation the
    decode applies (`unblock_scale_grid`), and an expectation derived from the implementation is no
    expectation. Use :func:`nvfp4_signed_tensors` for a layer whose decoded weight is known outright.
    """
    rows, in_features = codes.shape
    grid = block_scale if isinstance(block_scale, torch.Tensor) else torch.full((rows, in_features // 16), block_scale)
    return {
        f"{path}.weight": (codes[:, 0::2] << 4) | codes[:, 1::2],
        f"{path}.weight_scale": grid.to(torch.float8_e4m3fn),
        f"{path}.weight_scale_2": torch.tensor(global_scale),
    }


def nvfp4_signed_tensors(path: str, positive: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """An nvfp4 layer whose every weight is exactly +0.5 or -0.5, and that expected weight.

    A block scale of 2 and a global scale of 0.25 put the product at +-0.5, exact in every dtype
    involved, so the assertion can be equality rather than a tolerance and a dropped
    ``weight_scale_2`` moves the result by a factor of four. The block scale is uniform, which is
    what makes the expectation independent of the cuBLAS tile order -- that order is pinned on its
    own in ``tests/backend/quantization/test_nvfp4.py`` rather than restated here.
    """
    codes = nvfp4_codes(positive)
    tensors = nvfp4_tensors(path, codes, block_scale=2.0, global_scale=0.25)
    return tensors, torch.where(positive, 0.5, -0.5)


def stored_layout(grid: torch.Tensor) -> torch.Tensor:
    """Lay a row-major grid out the way checkpoints store it.

    By the measured index formula: element ``[m, k]`` lands at flat position ``position``. Shared
    with the nvfp4 and MXFP8 decode tests so both check against the same independent statement of
    the layout rather than against each other.
    """
    rows, blocks = grid.shape
    flat = torch.full((rows * blocks,), float("nan"), dtype=grid.dtype)
    for m in range(rows):
        for k in range(blocks):
            position = ((((m // 128) * (blocks // 4) + k // 4) * 32 + m % 32) * 4 + (m % 128) // 32) * 4 + k % 4
            flat[position] = grid[m, k]
    return flat.reshape(rows, blocks)


def mxfp8_tensors(
    path: str, exponents: torch.Tensor, *, block_size: int = MX_BLOCK_SIZE
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One MXFP8 layer in checkpoint layout, and the block-wise float scale it decodes to.

    This builder existed already, private to `tests/backend/quantization/test_fp8_scaled.py`, which
    now imports it from here instead. That is the whole of the change: MXFP8 was tested, it just was
    not available to the other suites, so a seam-level cell had to hand-roll the layout -- and the
    layout is the part worth getting from one place, because getting it wrong is silent.

    The grid is E8M0 exponent *bytes*, so 127 means `2**0`; folded as linear multipliers the weights
    come out around 127x too large, at the right shape and the right dtype, with nothing raised and
    nothing logged. And it is stored in cuBLAS tiles (:func:`stored_layout`), because that is how
    both published builds store it; read row-major it pairs blocks with the wrong rows.

    The weight is all ones, so the returned scale *is* the expected decoded weight -- an expectation
    that moves by `2**(byte - 127)` per block if either half of the decode is skipped. It is
    independent of the decode: the tile order comes from the hand-written index formula in
    :func:`stored_layout`, and the bias is the spec's 127 rather than `_E8M0_BIAS`.
    """
    rows, blocks = exponents.shape
    # `stored_layout` is a bijection only over whole tiles. Off-tile, its `flat` keeps NaNs, and
    # `.to(torch.uint8)` turns those into zeros -- every scale silently becomes `2**-127`.
    check_tile_layout(rows, blocks)
    tensors = {
        f"{path}.weight": torch.ones(rows, blocks * block_size).to(torch.float8_e4m3fn),
        f"{path}.weight_scale": stored_layout(exponents.to(torch.float64)).to(torch.uint8),
    }
    return tensors, torch.exp2(exponents.float() - 127)


def mxfp8_marker(*, block_size: int = MX_BLOCK_SIZE) -> dict[str, object]:
    """What a layer has to say about itself before the decode will read its grid.

    A `uint8` tensor beside an fp8 weight is otherwise just an unknown producer's convention. The
    format string is spelled out rather than imported from `fp8_scaled`: it is what the *producer*
    writes, so a rename on our side has to be caught by a test, not mirrored by one.
    """
    return {"format": "mxfp8", "block_size": block_size}
