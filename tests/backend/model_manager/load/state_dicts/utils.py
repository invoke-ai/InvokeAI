"""Shared helpers for model-loader state-dict fixtures.

Mirrors `tests/backend/patches/lora_conversions/lora_state_dicts/utils.py`: a fixture module
exports `state_dict_keys`, key name -> shape (or `(shape, dtype)` for the layouts
`scripts/capture_state_dict_fixture.py` captures from a real checkpoint's header). Tests expand it to a
mock state dict with `keys_to_mock_state_dict()`, or write it back as a header with
`write_header_only_safetensors()`, or as a small complete file with `write_shrunk_safetensors()`.
"""

import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def keys_to_mock_state_dict(keys: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Build a state dict of empty tensors from a {key: shape} mapping."""
    return {k: torch.empty(shape) for k, shape in keys.items()}


def token_extents(shape: list[int]) -> list[int]:
    """Shrink a captured shape to token extents, keeping its rank.

    A fixture whose tests read key names, dtypes, values and rank does not need the real extents,
    and the real ones are ruinous: the FLUX.2 mixed-fp8 layout alone is 2.3 billion elements, more
    memory than a CI runner has once the suite runs across several processes. Use this only where
    no assertion and no code path under test reads a size -- where the layout has to stay exact
    (a fused qkv that gets split into thirds, say), expand a single element to the full shape
    instead.
    """
    return [min(extent, 4) for extent in shape]


_SAFETENSORS_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I8": 1, "U8": 1}


def write_header_only_safetensors(path: Path, keys: dict[str, tuple[list[int], str]]) -> Path:
    """Write a safetensors file that has a fixture's real header and none of its data.

    Identification reads a safetensors file's header and nothing else, so a captured layout can be probed at its
    real shapes -- a T5-XXL is told from a smaller T5 by its width -- without writing gigabytes. Anything that
    reads tensor data from such a file fails, which is the point: identification must not.
    """
    header: dict[str, dict] = {}
    offset = 0
    for key, (shape, dtype) in keys.items():
        nbytes = _SAFETENSORS_DTYPE_BYTES[dtype] * math.prod(shape)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return path


_TORCH_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I8": torch.int8,
    "U8": torch.uint8,
}


def keys_to_meta_state_dict(keys: Mapping[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    """A captured layout as meta tensors of its real shapes and dtypes: what identification and a conversion see of
    a file, at no memory cost."""
    return {key: torch.empty(shape, dtype=_TORCH_DTYPES[dtype], device="meta") for key, (shape, dtype) in keys.items()}


def write_shrunk_safetensors(
    path: Path,
    keys: Mapping[str, tuple[list[int], str]],
    *,
    markers: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a fixture's layout as a complete safetensors file, every tensor zeros at `token_extents`.

    For identification that reads more of a file than its tensor entries. The header's metadata comes through
    `safe_open`, which refuses a file whose data a header-only write left out, and a `.comfy_quant` marker is read
    from its bytes: each marker the fixture decoded (`markers`, keyed by layer) is written back as its JSON.

    One layout keeps a size, because identification checks it: a packed nvfp4 weight and its block-scale grid keep
    one whole scale tile of rows and blocks (`_NVFP4_TILE`), so a released layer stays a valid one and a misaligned
    one stays misaligned. Nothing else that reads a size may use this.
    """
    tensors: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in keys.items():
        layer = key.removesuffix(".comfy_quant")
        if markers is not None and layer != key and layer in markers:
            raw = json.dumps(dict(markers[layer])).encode("utf-8")
            tensors[key] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        else:
            tensors[key] = torch.zeros(_shrunk_extents(key, shape, dtype), dtype=_TORCH_DTYPES[dtype])
    save_file(tensors, str(path), metadata=dict(metadata) if metadata else None)
    return path


# One cuBLAS scale tile of an nvfp4 layer: 128 rows by 4 blocks of 16 elements, two elements packed per byte.
_NVFP4_TILE = {"U8": (128, 32), "F8_E4M3": (128, 4)}


def _shrunk_extents(key: str, shape: list[int], dtype: str) -> list[int]:
    """`token_extents`, except for an nvfp4 layer's packed weight and grid, which keep at most one scale tile."""
    nvfp4_part = (dtype == "U8" and key.endswith(".weight")) or (dtype == "F8_E4M3" and key.endswith(".weight_scale"))
    if len(shape) == 2 and nvfp4_part:
        return [min(extent, limit) for extent, limit in zip(shape, _NVFP4_TILE[dtype], strict=True)]
    return token_extents(shape)
