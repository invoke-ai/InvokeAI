"""Small shared contracts for host-RAM admission."""

from collections.abc import Collection, Mapping
from typing import Any, Protocol

import torch


class HostMemoryReserver(Protocol):
    def __call__(self, cache_bytes: int, *, host_bytes_needed: int) -> Any: ...


def require_ram_headroom(reservation_result: Any, bytes_needed: int) -> None:
    """Raise when a host-aware reservation reports that its request cannot fit."""
    if getattr(reservation_result, "ram_headroom_satisfied", True) is False:
        raise MemoryError(
            f"Insufficient host memory for a {bytes_needed / 2**30:.2f} GiB model-load allocation; "
            "close other applications or load a smaller model."
        )


def _tensor_storage_id(tensor: torch.Tensor) -> int:
    """Return the storage identity used by per-reader mmap ownership."""
    return tensor.untyped_storage().data_ptr()


def _source_is_mapped(tensor: Any, source_tensors_mapped: bool | Collection[int] | None) -> bool:
    if source_tensors_mapped is True:
        return True
    if not isinstance(source_tensors_mapped, Collection) or isinstance(source_tensors_mapped, (str, bytes)):
        return False
    return isinstance(tensor, torch.Tensor) and _tensor_storage_id(tensor) in source_tensors_mapped


def state_dict_cast_bytes(
    sd: Mapping[str, Any],
    dtype: torch.dtype,
    *,
    in_place: bool,
    source_tensors_mapped: bool | Collection[int] | None = False,
) -> tuple[int, int]:
    """Return final state-dict bytes and peak extra bytes for a dtype cast.

    A mapped source is reclaimable file-backed storage, not owned RAM. A collection identifies the mapped
    storages by pointer so a state dict assembled from mixed mmap/pread readers is accounted per storage.
    """
    target_bytes = sum(
        tensor.nelement() * (dtype.itemsize if tensor.is_floating_point() else tensor.element_size())
        for tensor in sd.values()
    )
    if not in_place:
        extra_bytes = sum(
            tensor.nelement() * dtype.itemsize
            for tensor in sd.values()
            if tensor.is_floating_point() and getattr(tensor, "dtype", None) != dtype
        )
        return target_bytes, extra_bytes

    source_counts: dict[int, int] = {}
    source_bytes: dict[int, int] = {}
    source_mapped: dict[int, bool] = {}
    for tensor in sd.values():
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            continue
        storage_id = _tensor_storage_id(tensor)
        source_counts[storage_id] = source_counts.get(storage_id, 0) + 1
        source_bytes[storage_id] = tensor.untyped_storage().nbytes()
        source_mapped[storage_id] = _source_is_mapped(tensor, source_tensors_mapped)

    delta = 0
    peak = 0
    for tensor in sd.values():
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            continue
        if not tensor.is_floating_point() or getattr(tensor, "dtype", None) == dtype:
            continue
        new_bytes = tensor.nelement() * dtype.itemsize
        storage_id = _tensor_storage_id(tensor)
        peak = max(peak, delta + new_bytes)
        delta += new_bytes
        source_counts[storage_id] -= 1
        if source_counts[storage_id] == 0 and not source_mapped[storage_id]:
            delta -= source_bytes[storage_id]
    return target_bytes, peak
