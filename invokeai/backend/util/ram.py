"""Small shared contracts for host-RAM admission."""

from collections.abc import Mapping
from typing import Any

import torch


def require_ram_headroom(reservation_result: Any, bytes_needed: int) -> None:
    """Raise when a host-aware reservation reports that its request cannot fit."""
    if getattr(reservation_result, "ram_headroom_satisfied", True) is False:
        raise MemoryError(
            f"Insufficient host memory for a {bytes_needed / 2**30:.2f} GiB model-load allocation; "
            "close other applications or load a smaller model."
        )


def state_dict_cast_bytes(sd: Mapping[str, Any], dtype: torch.dtype, *, in_place: bool) -> tuple[int, int]:
    """Return final state-dict bytes and peak extra bytes for a dtype cast."""
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

    delta = 0
    peak = 0
    for tensor in sd.values():
        if not tensor.is_floating_point() or getattr(tensor, "dtype", None) == dtype:
            continue
        old_bytes = tensor.nelement() * tensor.element_size()
        new_bytes = tensor.nelement() * dtype.itemsize
        peak = max(peak, delta + new_bytes)
        delta += new_bytes - old_bytes
    return target_bytes, peak
