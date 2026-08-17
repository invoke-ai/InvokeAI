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
    # it was just handed. The flag cannot shield a record forever: the synchronous eviction
    # paths (make_room, drop_model) ignore it, and the next admission on the same cache clears
    # any flag still standing (see the sweep in ModelCache.put()), so a load that errors out
    # between put() and the LoadedModel's construction cannot dodge budget reconciles
    # indefinitely. From the wrapper's construction on, the window is tracked by first_use_holds
    # below, whose release is guaranteed by the wrapper's finalizer rather than by the sweep.
    awaiting_first_use: bool = False
    # Count of live LoadedModel wrappers holding this record that have not yet locked it. Armed by
    # ModelCache.register_first_use_hold() (called from LoadedModelWithoutConfig.__init__) and
    # released exactly once per wrapper — on the wrapper's first lock, or by its weakref finalizer
    # if it is dropped without ever locking. Unlike awaiting_first_use, these holds are NOT swept
    # by the next admission: a warm get()'s wrapper can legitimately sit un-entered across another
    # model's cold load (a node retrieves several models before entering their contexts), and its
    # finalizer guarantees the release the sweep exists to backstop. The only recovery sweep is
    # ModelCache._ensure_deferred_worker() zeroing the counts after the deferred worker — the
    # thread that carries finalizer-initiated releases — is found dead, since holds released into
    # a dead worker are dropped and would otherwise shield the record forever.
    first_use_holds: int = 0

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

    @property
    def in_first_use_window(self) -> bool:
        """True while a load or a live LoadedModel wrapper is between obtaining this record and
        locking it. The asynchronous eviction sweeps (shutdown, budget reconcile, peer-requested
        eviction) treat such a record like a locked one: evicting it would detach a record whose
        holder is about to lock it, splitting the model from the cache's RAM accounting and — for
        shared weights — releasing store ownership while the tensors live on, so a peer's reload
        would mint a duplicate canonical copy. The synchronous paths (make_room, drop_model,
        unlock's stale eviction) honor only the first_use_holds half — see awaiting_first_use for
        why an orphaned grace must stay reachable there."""
        return self.awaiting_first_use or self.first_use_holds > 0
