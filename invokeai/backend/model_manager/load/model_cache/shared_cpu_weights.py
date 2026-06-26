import threading
from dataclasses import dataclass, field

import torch

from invokeai.backend.util.calc_tensor_size import calc_tensor_size


@dataclass
class _SharedWeightsEntry:
    """A single canonical CPU state dict shared across per-device caches."""

    state_dict: dict[str, torch.Tensor]
    total_bytes: int
    # Number of per-device cached models currently aliasing this entry. The entry is freed
    # (its RAM released) when this drops to zero.
    refcount: int = 0
    _key_bytes: dict[str, int] = field(default_factory=dict)


class SharedCpuWeightsStore:
    """Process-global store of canonical CPU weight tensors, shared across per-device model caches.

    In multi-GPU mode there is one `ModelCache` per generation device. Without coordination each
    cache keeps its own CPU copy of every model's weights, so a model loaded on N GPUs occupies N
    copies in RAM. The cached-model wrappers cannot simply share a single `torch.nn.Module`, because
    loading to VRAM mutates a module's parameters in place (`load_state_dict(assign=True)` / `.to`),
    and two GPUs running the same model concurrently need their params on two different devices at
    once. The CPU weight tensors, however, are read-only and device-agnostic, so they can be shared.

    This store keeps a single canonical CPU `state_dict` per cache key. The first device to load a
    key registers its freshly-built state dict as canonical; subsequent devices `acquire()` the
    canonical and re-point their own module's CPU parameters at the shared tensors (via
    `load_state_dict(assign=True)`), discarding their private duplicate. The result: model weights
    live once in RAM regardless of how many GPUs hold the model.

    Lifetime is reference-counted. Each per-device cached model that adopts an entry must call
    `release()` exactly once when it is evicted; the canonical tensors are dropped only when the
    last device releases them.

    Thread-safety: `acquire()`/`release()` are guarded by an internal lock. Note that model
    construction (where `acquire()` is normally called) is already serialized process-globally by
    `MODEL_LOAD_LOCK.write_lock()`; the internal lock here additionally protects `release()`, which
    runs under a per-cache lock off the global construction lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _SharedWeightsEntry] = {}

    def acquire(self, key: str, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adopt the canonical CPU state dict for `key`, registering `state_dict` as canonical if
        this is the first acquire.

        Increments the entry's refcount. The caller MUST pair every `acquire()` with exactly one
        `release()`.

        Returns:
            The canonical state dict. If this call registered the entry, the returned object is the
            same `state_dict` that was passed in (the caller keeps using its own tensors). Otherwise
            it is the previously-registered canonical dict, and the caller is responsible for
            re-pointing its module at these tensors and dropping the `state_dict` it passed in.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _SharedWeightsEntry(
                    state_dict=state_dict,
                    total_bytes=sum(calc_tensor_size(v) for v in state_dict.values()),
                )
                self._entries[key] = entry
            entry.refcount += 1
            return entry.state_dict

    def release(self, key: str) -> None:
        """Release one reference to `key`'s canonical state dict, freeing it when the count hits 0.

        A `release()` for a key that is not present is a no-op (e.g. a cached model that never
        acquired shared weights, or a double eviction guard).
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount -= 1
            if entry.refcount <= 0:
                del self._entries[key]

    # -- Introspection / accounting (also used by tests) ----------------------------------------

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def refcount(self, key: str) -> int:
        """Return the current refcount for `key`, or 0 if not present."""
        with self._lock:
            entry = self._entries.get(key)
            return entry.refcount if entry is not None else 0

    def total_bytes_in_use(self) -> int:
        """Return the total size (in bytes) of all canonical state dicts currently held.

        This counts each shared model's weights exactly once, regardless of how many devices alias
        it — i.e. the true RAM footprint of cached weights, not the per-device double-count.
        """
        with self._lock:
            return sum(entry.total_bytes for entry in self._entries.values())

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())


# Process-global default store. Per-device caches share this instance so that the same model loaded
# on multiple GPUs keeps a single CPU copy. Tests may construct isolated `SharedCpuWeightsStore`
# instances instead.
SHARED_CPU_WEIGHTS = SharedCpuWeightsStore()
