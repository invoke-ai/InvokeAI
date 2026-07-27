"""Shared helpers for model-loader state-dict fixtures.

Mirrors `tests/backend/patches/lora_conversions/lora_state_dicts/utils.py`: a fixture module
exports `state_dict_keys: dict[str, list[int]]` (key name -> shape, captured from a real
checkpoint) and tests expand it to a mock state dict with `keys_to_mock_state_dict()`.
"""

import torch


def keys_to_mock_state_dict(keys: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Build a state dict of empty tensors from a {key: shape} mapping."""
    return {k: torch.empty(shape) for k, shape in keys.items()}
