"""Which state dict entries the model cache treats as one tensor.

Everything the cache does with tied weights -- charging their bytes once, moving them as a unit -- rests on
`storage_key` answering equal only for entries that really are the same memory. Answering equal for two unrelated
tensors would hand one layer's weights to another, silently.
"""

import torch

from invokeai.backend.model_manager.load.model_cache.tensor_aliases import (
    analyze_state_dict,
    move_shared,
    storage_key,
)


def test_two_names_for_one_tensor_are_charged_once_and_grouped():
    weight = torch.ones(8, 4)
    aliases = analyze_state_dict({"embed.weight": weight, "head.weight": weight, "bias": torch.ones(4)})

    assert sum(aliases.bytes_by_key.values()) == weight.nelement() * 4 + 16
    assert aliases.groups["head.weight"] == ("embed.weight", "head.weight")
    assert "bias" not in aliases.groups


def test_distinct_tensors_of_equal_shape_are_kept_apart():
    aliases = analyze_state_dict({"a": torch.ones(8, 4), "b": torch.ones(8, 4)})

    assert aliases.groups == {}
    assert all(size > 0 for size in aliases.bytes_by_key.values())


def test_views_of_one_storage_that_read_different_values_are_kept_apart():
    """A same-shaped view with another dtype starts at the same address and is not the same values."""
    weight = torch.ones(8, 4, dtype=torch.float16)

    assert storage_key(weight) != storage_key(weight.view(torch.int16))


def test_tensors_without_memory_answer_by_identity():
    """`data_ptr()` is 0 for meta and zero-element tensors, so keying on it alone would declare them all one tensor."""
    # Held in variables: identity keys are only distinct while both tensors are alive.
    first_meta, second_meta = torch.empty(4, 4, device="meta"), torch.empty(4, 4, device="meta")
    first_empty, second_empty = torch.empty(0), torch.empty(0)

    assert storage_key(first_meta) != storage_key(second_meta)
    assert storage_key(first_empty) != storage_key(second_empty)


def test_moving_one_of_two_names_moves_both_to_the_same_tensor():
    weight = torch.ones(8, 4)
    moved: dict = {}

    first = move_shared(weight, torch.device("meta"), moved)
    second = move_shared(weight, torch.device("meta"), moved)

    assert first is second
    assert first.is_meta
