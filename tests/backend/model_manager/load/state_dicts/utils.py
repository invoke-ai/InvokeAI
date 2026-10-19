"""Shared helpers for model-loader state-dict fixtures.

Mirrors `tests/backend/patches/lora_conversions/lora_state_dicts/utils.py`: a fixture module
exports `state_dict_keys`, key name -> shape (or `(shape, dtype)` for the layouts
`scripts/capture_state_dict_fixture.py` captures from a real checkpoint's header). Tests expand it to a
mock state dict with `keys_to_mock_state_dict()`, or write it back as a header with
`write_header_only_safetensors()`.
"""

import json
import math
import struct
from pathlib import Path

import torch


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


_SAFETENSORS_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1}


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
