"""Peak live bytes on one device, counted from the tensors aten ops allocate there.

A working-memory estimate states what a call holds at its peak. ``torch.cuda`` can measure that only on a GPU;
this counts the same allocations on any device, the meta device included, so the meta device can stand in for the
accelerator and the comparison runs in CI. It sees every tensor an op returns, not the scratch inside a kernel
(cuBLAS workspaces), and counts requested bytes, without the caching allocator's rounding.
"""

import weakref
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class PeakDeviceBytes(TorchDispatchMode):
    """Tracks the bytes of storages ops allocate on ``device`` while the mode is active, and their peak.

    A storage counts from the op that returned it until it is freed. Outputs that alias an input (views, in-place
    ops) allocate nothing and are skipped, and so is anything that existed before the mode was entered.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.live = 0
        self.peak = 0
        self._tracked: set[int] = set()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if any(ret.alias_info is not None for ret in func._schema.returns):
            return out
        outputs, _ = tree_flatten(out)
        for tensor in outputs:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == self.device.type:
                self._track(tensor.untyped_storage())
        return out

    def _track(self, storage: torch.UntypedStorage) -> None:
        key = id(storage)
        if key in self._tracked:
            return
        self._tracked.add(key)
        nbytes = storage.nbytes()
        self.live += nbytes
        self.peak = max(self.peak, self.live)
        weakref.finalize(storage, self._release, key, nbytes)

    def _release(self, key: int, nbytes: int) -> None:
        self._tracked.discard(key)
        self.live -= nbytes
