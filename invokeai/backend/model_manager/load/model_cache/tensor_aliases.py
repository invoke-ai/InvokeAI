"""Recognizing state dict entries that are one tensor under several names.

A tied weight -- a language model's `lm_head.weight` sharing the embedding matrix is the common case, and every Qwen3
text encoder InvokeAI loads is built that way -- appears in `state_dict()` under each of its names. Summing such a
dict charges the memory once per name, and moving each name to the compute device on its own allocates it once per
name and unties them: a Qwen3 4B encoder wastes 0.7 GiB of VRAM that way, an 8B encoder 1.2 GiB.
"""

from dataclasses import dataclass

import torch

from invokeai.backend.util.calc_tensor_size import calc_tensor_size

# A tensor's identity as memory: two entries with the same key are the same bytes, read the same way.
StorageKey = tuple[int, int, torch.dtype, tuple[int, ...], tuple[int, ...]] | int


def storage_key(tensor: torch.Tensor) -> StorageKey:
    """A key that is equal for two tensors exactly when they are the same memory, read the same way.

    Tensors with no memory behind them answer by identity instead: `data_ptr()` is 0 for meta tensors and for
    zero-element ones, so keying on it would declare every such tensor the same memory as the others. Quantized
    tensor subclasses (GGUF, SDNQ) do not expose an untyped storage at all and fall back the same way. Identity
    treats a tensor as unshared, which only forgoes deduplication; it never reports two distinct tensors as one.

    The dtype and the byte offset are part of the key because two views of one storage can start at the same address
    and still be different values -- handing one of them out for the other would be a silently wrong weight.
    """
    try:
        data_pointer = tensor.untyped_storage().data_ptr()
    except (NotImplementedError, RuntimeError, AttributeError):
        return id(tensor)
    if data_pointer == 0:
        return id(tensor)
    return (
        data_pointer,
        tensor.storage_offset() * tensor.element_size(),
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


@dataclass(frozen=True)
class StateDictAliases:
    """How a state dict's keys map onto memory.

    `bytes_by_key` charges each tensor to the first key that names it and the rest 0, so summing it counts the memory
    once. `groups` names, for every key that shares memory, all the keys of that group: a caller moving tensors
    between devices has to keep a group together, or the bytes charged to one key would move under another.
    """

    bytes_by_key: dict[str, int]
    groups: dict[str, tuple[str, ...]]


def analyze_state_dict(state_dict: dict[str, torch.Tensor]) -> StateDictAliases:
    """Charge every tensor in `state_dict` once and find the keys that share memory."""
    bytes_by_key: dict[str, int] = {}
    keys_by_storage: dict[StorageKey, list[str]] = {}
    for key, tensor in state_dict.items():
        key_storage = storage_key(tensor)
        group = keys_by_storage.setdefault(key_storage, [])
        bytes_by_key[key] = calc_tensor_size(tensor) if not group else 0
        group.append(key)
    groups = {key: tuple(group) for group in keys_by_storage.values() if len(group) > 1 for key in group}
    return StateDictAliases(bytes_by_key=bytes_by_key, groups=groups)


def move_shared(tensor: torch.Tensor, device: torch.device, moved: dict[StorageKey, torch.Tensor]) -> torch.Tensor:
    """Move `tensor` to `device`, reusing the result for every other entry that is the same memory.

    `moved` accumulates across one conversion pass, so the tensors stay tied on the target device instead of becoming
    independent copies.
    """
    key = storage_key(tensor)
    already_moved = moved.get(key)
    if already_moved is None:
        already_moved = tensor.to(device)
        moved[key] = already_moved
    return already_moved
