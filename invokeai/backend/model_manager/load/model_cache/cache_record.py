import weakref
from dataclasses import dataclass
from typing import Optional

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
    # indefinitely. That backstop is why the flag is withheld whenever it could outlive its
    # releasers: an admission made with no deferred worker running, and one made after
    # shutdown() — after which no further put() is guaranteed to run the sweep. For the same
    # reason shutdown() retires every flag still standing rather than letting its sweep
    # stale-retain the record it shields: a load cancelled before its retrieval leaves no wrapper,
    # so nothing but that sweep could ever have released it. From the retrieval on, the window is
    # tracked by first_use_holds below, whose release is guaranteed by a finalizer rather than by
    # the sweep.
    awaiting_first_use: bool = False
    # Count of live holders of this record that have retrieved it but not yet locked it. Armed by
    # ModelCache.register_first_use_hold() — from the cache lookup itself
    # (ModelCache.get_with_first_use_claim, so that the window has no unshielded head), or from
    # LoadedModelWithoutConfig.__init__ for a record obtained through plain get() — and released
    # exactly once per holder: on the holder's first lock, or by the weakref finalizer of the
    # FirstUseClaim (or the wrapper) that owns it, if it is dropped without ever locking. Unlike
    # awaiting_first_use, these holds are NOT swept by the next admission: a warm get()'s holder
    # can legitimately sit un-entered across another model's cold load (a node retrieves several
    # models before entering their contexts), and its finalizer guarantees the release the sweep
    # exists to backstop. The only recovery sweep is
    # ModelCache zeroing the counts once the deferred worker — the thread that carries
    # finalizer-initiated releases — is gone (from inside the dying worker itself, and as a
    # backstop at the next worker start and at shutdown), since a release dispatched toward a
    # dead worker may be dropped and would otherwise shield the record forever.
    first_use_holds: int = 0
    # Bumped whenever stranded holds are zeroed (dead-worker recovery). Every hold release
    # carries the epoch it was armed under and is ignored across a bump: without this, a
    # surviving wrapper's late release — or a release enqueued before the old worker died and
    # drained after the restart — would decrement a FRESH hold armed by a different wrapper
    # under the healthy replacement worker, silently unshielding that wrapper's window.
    first_use_holds_epoch: int = 0
    # Weak reference to the FirstUseClaim handed to the loader that admitted this record
    # (ModelCache.put(claim_admission=True)), or None if the admission was not claimed. Resolving
    # it answers, synchronously and with no release path of any kind, the one question the
    # asynchronous eviction sweeps actually need to ask about a just-admitted record: is the load
    # that put it here still running? While it resolves, that loader's own local still holds these
    # tensors, so evicting the record would release shared-store ownership (and debit the budget)
    # for bytes that are still resident — a peer's reload would then mint a duplicate canonical
    # the budget counts once — and would fail the load with an IndexError from a retrieval that
    # can no longer find its own model.
    #
    # This is why the shield the asynchronous sweeps consult is the reference and not the two
    # flags above. awaiting_first_use is unowned: any holder's abandonment
    # (_release_abandoned_holder) and the next admission's sweep clear it on behalf of whoever
    # ran, so one load can strip another's shield. first_use_holds is owned but its release rides
    # on the deferred worker, so dead-worker recovery must zero it — deliberately trading a live
    # holder's shield for the certainty that no record stays shielded forever. A weak reference
    # has neither problem: nothing has to release it, so nothing can release it early or fail to
    # release it at all. That is also why no recovery ever retires it, not even on a shut-down
    # cache with a dead worker: a claim that is still alive means a load that is still between its
    # put() and its retrieval, and evicting its record does not fall back to anything — the
    # retrieval raises. What such a record needs once its claim has died is an eviction trigger,
    # and that is supplied by the claim's finalizer (an abandonment release the queue keeps even
    # with no worker to drain it) and by the sweep every later admission runs when no worker can
    # be started (see ModelCache.put()).
    #
    # Only the asynchronous sweeps consult it, through in_first_use_window below. The synchronous
    # paths (make_room, drop_model, unlock's stale eviction) gate on first_use_holds alone and
    # have never honoured this window's flags — the trade documented on awaiting_first_use — so
    # they are unaffected either way.
    admission_claim_ref: Optional["weakref.ref"] = None
    # Set, lock-free, by ModelCache.release_first_use_grace when a hold-less abandonment of this
    # record (a wrapper or claim that never armed a first-use hold) finds it stale and queues the
    # eviction that stale-ness now owes; cleared by the deferred worker the moment it dequeues that
    # item — before the handler runs, so that neither a raise in the handler, nor the worker dying
    # inside it, nor a fork can leave the gate closed with nothing queued behind it. Such
    # abandonments carry nothing to decrement, so one queued
    # item per record is as good as any number, and this flag is what bounds the queue while no
    # worker is draining it: without it every warm get() whose wrapper is dropped un-entered
    # would add an item that nothing removes (JPPhoto review, 2026-09-04). Hold-carrying
    # releases are deliberately not coalesced — each one retires exactly one hold — and are
    # bounded instead by the holds themselves, which are armed only while a worker is alive.
    abandonment_release_pending: bool = False

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
    def admission_in_flight(self) -> bool:
        """True while the load that admitted this record is still between its put() and its
        retrieval — see admission_claim_ref."""
        claim_ref = self.admission_claim_ref
        return claim_ref is not None and claim_ref() is not None

    @property
    def in_first_use_window(self) -> bool:
        """True while a load, a live first-use claim, or a live LoadedModel wrapper is between
        obtaining this record and locking it — including the load that admitted it, for as long as
        it has not come back to retrieve it (admission_in_flight). The asynchronous eviction sweeps (shutdown, budget
        reconcile, peer-requested eviction) treat such a record like a locked one: evicting it would detach a record whose
        holder is about to lock it, splitting the model from the cache's RAM accounting and — for
        shared weights — releasing store ownership while the tensors live on, so a peer's reload
        would mint a duplicate canonical copy. The synchronous paths (make_room, drop_model,
        unlock's stale eviction) honor only the first_use_holds half — see awaiting_first_use for
        why an orphaned grace must stay reachable there."""
        return self.awaiting_first_use or self.first_use_holds > 0 or self.admission_in_flight
