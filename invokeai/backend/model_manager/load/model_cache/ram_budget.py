import threading
import weakref
from typing import TYPE_CHECKING, Optional

from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore

if TYPE_CHECKING:
    from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


class RamBudget:
    """The single global authority for how much RAM the model caches are actually using.

    In multi-GPU mode there is one `ModelCache` per device. Each cache independently sums the
    `total_bytes()` of the models it holds, so a model resident on N devices is counted N times —
    even though Phase 1/2 made its CPU weights live only ONCE in RAM (see SharedCpuWeightsStore).
    That per-cache double-count makes the caches believe RAM is fuller than it is, causing premature
    eviction and reload churn, and makes `max_cache_ram_gb` meaningless as a system-wide cap.

    RamBudget fixes the accounting by separating RAM into two non-overlapping parts:

    - Shared weights: model weights that are deduplicated in the SharedCpuWeightsStore. Counted
      exactly once via `store.total_bytes_in_use()`, regardless of how many devices hold them.
    - Non-shared RAM: models that are NOT deduplicated (keep_ram_copy disabled, or non-Module
      models whose single in-RAM copy is per-device). These are tracked here as an explicit running
      total; a model resident on N devices contributes N times, which is correct because it really
      does occupy N copies of RAM.

    `total_in_use()` is the sum of the two and reflects the true RAM footprint. All per-device caches
    share one RamBudget and make their eviction decisions against it.

    Thread-safety / lock ordering: RamBudget guards its own counter with an internal lock and NEVER
    acquires a ModelCache lock (it only reads the store, which has its own lock; the cache registry
    below stores references without touching cache locks). Callers update it while holding their
    cache lock, so the only lock order is cache-lock -> (store-lock | budget-lock), never the
    reverse — so it cannot deadlock against the per-device caches.
    """

    def __init__(self, max_bytes: int, shared_store: Optional[SharedCpuWeightsStore]):
        self._max_bytes = max_bytes
        self._store = shared_store
        self._non_shared_bytes = 0
        self._unowned_non_shared_bytes = 0
        self._lock = threading.RLock()
        # The per-device caches attached to this budget. A cache that has exhausted its own
        # evictable entries uses this to ask its peers to release RAM they are holding (see
        # ModelCache._make_room_internal). Weak references: the budget must not keep a
        # shut-down cache alive.
        self._caches: list[weakref.ReferenceType["ModelCache"]] = []
        self._non_shared_bytes_by_cache: dict[weakref.ReferenceType["ModelCache"], int] = {}

    def _find_cache_ref(self, cache: "ModelCache") -> Optional[weakref.ReferenceType["ModelCache"]]:
        return next((ref for ref in self._caches if ref() is cache), None)

    def _on_cache_collected(self, cache_ref: weakref.ReferenceType["ModelCache"]) -> None:
        with self._lock:
            self._non_shared_bytes = max(0, self._non_shared_bytes - self._non_shared_bytes_by_cache.pop(cache_ref, 0))
            self._caches = [ref for ref in self._caches if ref is not cache_ref]

    def register_cache(self, cache: "ModelCache") -> None:
        """Register a per-device cache attached to this budget."""
        with self._lock:
            if self._find_cache_ref(cache) is None:
                self._caches.append(weakref.ref(cache, self._on_cache_collected))

    def peer_caches(self, exclude: "ModelCache") -> list["ModelCache"]:
        """The other live caches attached to this budget."""
        with self._lock:
            refs = list(self._caches)
        return [cache for ref in refs if (cache := ref()) is not None and cache is not exclude]

    @property
    def max_bytes(self) -> int:
        """The global cap on actual model-cache RAM, in bytes."""
        return self._max_bytes

    def add_non_shared(self, nbytes: int, cache: Optional["ModelCache"] = None) -> None:
        """Record `nbytes` of newly-resident non-deduplicated model RAM."""
        with self._lock:
            self._non_shared_bytes += nbytes
            if cache is None:
                self._unowned_non_shared_bytes += nbytes
                return

            cache_ref = self._find_cache_ref(cache)
            if cache_ref is None:
                cache_ref = weakref.ref(cache, self._on_cache_collected)
                self._caches.append(cache_ref)
            self._non_shared_bytes_by_cache[cache_ref] = self._non_shared_bytes_by_cache.get(cache_ref, 0) + nbytes

    def remove_non_shared(self, nbytes: int, cache: Optional["ModelCache"] = None) -> None:
        """Record the release of `nbytes` of non-deduplicated model RAM."""
        with self._lock:
            if cache is None:
                removed_bytes = min(nbytes, self._unowned_non_shared_bytes)
                self._unowned_non_shared_bytes -= removed_bytes
                self._non_shared_bytes -= removed_bytes
                return

            cache_ref = self._find_cache_ref(cache)
            if cache_ref is None:
                return
            tracked_bytes = self._non_shared_bytes_by_cache.get(cache_ref, 0)
            removed_bytes = min(nbytes, tracked_bytes)
            self._non_shared_bytes = max(0, self._non_shared_bytes - removed_bytes)
            remaining_bytes = tracked_bytes - removed_bytes
            if remaining_bytes > 0:
                self._non_shared_bytes_by_cache[cache_ref] = remaining_bytes
            else:
                self._non_shared_bytes_by_cache.pop(cache_ref, None)

    def total_in_use(self) -> int:
        """The true total RAM used by the model caches: shared weights (counted once) + non-shared."""
        # The store read MUST stay outside self._lock. The store's deferred-release drain runs
        # under the store lock and allocates, so a cyclic GC can fire there and run
        # _on_cache_collected, which takes THIS lock (store → budget on one thread). If any thread
        # held the budget lock while calling into the store (budget → store), the two orders would
        # deadlock against each other.
        shared = self._store.total_bytes_in_use() if self._store is not None else 0
        with self._lock:
            non_shared = self._non_shared_bytes
        return shared + non_shared

    def available(self) -> int:
        """Bytes remaining under the global cap (may be negative if over budget)."""
        return self._max_bytes - self.total_in_use()
