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
    # Set by ModelCache.put() and cleared on the entry's first get()/lock(). A freshly admitted
    # model is about to be used — its loader calls get() as soon as put() returns — so the
    # asynchronous eviction paths (shared-budget reconcile, peer-requested eviction) must not
    # treat it as idle: with a peer's reconcile request pending, put()'s own lock-release hook
    # would otherwise evict the model before the loader can even retrieve it, breaking the
    # in-flight load. The cache's local make_room path ignores this flag (cold loads are
    # serialized under MODEL_LOAD_LOCK, so it can never see another loader's entry inside this
    # window), which also bounds the flag's lifetime if a load errors out between put() and
    # get().
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
