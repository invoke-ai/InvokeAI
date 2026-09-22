from dataclasses import dataclass

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)


@dataclass
class CacheRecord:
    """A class that represents a model in the model cache."""

    # Cache key.
    key: str
    # Model in memory.
    cached_model: CachedModelWithPartialLoad | CachedModelOnlyFullLoad
    _locks: int = 0
    # Set by ModelCache.drop_model() when the entry was locked at invalidation time.
    # ModelCache.unlock() evicts the entry as soon as the last lock releases so a setting
    # change (e.g. fp8_storage toggled during an in-flight generation) takes effect on the
    # next load instead of silently being ignored.
    is_stale: bool = False
    # Post-admission grace: set by ModelCache.put() (unless the admission is a prefetch of a
    # model nothing will come back for) and cleared on the entry's first lock(). A freshly
    # admitted model is about to be used — its loader calls get() as soon as put() returns, then
    # locks it for inference — so the asynchronous eviction paths (shared-budget reconcile,
    # peer-requested eviction) must not treat it as idle: they would evict the record out from
    # under the in-flight load, breaking the loader's get() or detaching a live model from the
    # cache's RAM accounting. The grace deliberately survives get(): get() is synchronized, and
    # its own lock-release hook may run a pending reconcile before the caller can lock the record
    # it was just handed. The flag cannot shield a record forever: the cache's local make_room
    # path ignores it (cold loads are serialized under MODEL_LOAD_LOCK, so make_room can never
    # see another loader's entry inside the put()->lock() window), and the next admission on the
    # same cache clears any flag still standing (see the sweep in ModelCache.put()), so a load
    # that errors out — or a LoadedModel dropped without ever locking — cannot dodge budget
    # reconciles indefinitely.
    awaiting_first_use: bool = False

    def lock(self) -> None:
        """Lock this record."""
        self._locks += 1

    def unlock(self) -> None:
        """Unlock this record."""
        self._locks -= 1
        assert self._locks >= 0

    @property
    def is_locked(self) -> bool:
        """Return true if record is locked."""
        return self._locks > 0
