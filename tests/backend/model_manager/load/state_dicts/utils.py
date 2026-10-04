"""Shared helpers for model-loader state-dict fixtures.

Mirrors `tests/backend/patches/lora_conversions/lora_state_dicts/utils.py`: a fixture module
exports `state_dict_keys: dict[str, list[int]]` (key name -> shape, captured from a real
checkpoint) and tests expand it to a mock state dict with `keys_to_mock_state_dict()`.
"""

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
