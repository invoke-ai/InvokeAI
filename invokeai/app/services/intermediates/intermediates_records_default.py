"""Storage of the intermediates manager.

One classification expression decides what every intermediate is under the cleanup policy (see
`queries/intermediates.py`), whether it aggregates a summary, describes a preview, pages an operation or guards
a delete.

A preview freezes the instant its recency is judged at; the operation it confirms and the guard on every
deleting transaction classify with that same recency cutoff and a live clock for everything else. Anything
created after the preview is newer than its cutoff, so it stays ``recent`` for the whole operation. Cached-output
holds are judged the same way by the operation itself, but the hold table is shared with live-clock callers
(summaries, cache hits) that may drop a hold once its grace has passed live; that only ever drops a hold whose
grace is over.

Queue inputs are found by scanning each active item's stored session for media-name keys; the scan is cached
per item, so a queue of a thousand pending items is parsed once, not per query.
"""

import re
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Collection, Iterator, NamedTuple, Optional, Sequence

from invokeai.app.services.intermediates.intermediates_common import (
    BROWSER_HOLD_TTL_SECONDS,
    MAX_BROWSER_HOLD_LEASES_PER_USER,
    RECENT_GRACE_SECONDS,
    IntermediatesCleanupMode,
    IntermediatesKindCounts,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.intermediates import (
    TARGET_SLOTS,
    Classification,
    MediaKind,
    ScopeSpec,
    WindowRow,
)
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.types import timestamp_text
from invokeai.app.services.shared.intermediate_delete import IntermediateDeleteGuard
from invokeai.app.services.shared.media_references import IMAGE_NAME_KEYS, VIDEO_NAME_KEYS, MediaReferences

# One row of the manager: (owner, project), project None for the owner's unassigned intermediates.
ScopeTarget = tuple[str, Optional[str]]

MEDIA_KINDS: tuple[MediaKind, ...] = ("image", "video")


def _name_pattern(keys: frozenset[str]) -> re.Pattern[str]:
    alternatives = "|".join(sorted(re.escape(key) for key in keys))
    return re.compile(rf'"(?:{alternatives})"\s*:\s*"([^"\\]{{1,255}})"')


_IMAGE_NAME_RE = _name_pattern(IMAGE_NAME_KEYS)
_VIDEO_NAME_RE = _name_pattern(VIDEO_NAME_KEYS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _Clock(NamedTuple):
    """One instant for every statement of a classification."""

    now: str
    recent_cutoff: str


def _clock(recent_cutoff: Optional[str] = None) -> _Clock:
    """The live clock, or the live clock judging recency as of an earlier preview."""
    now = _utc_now()
    live_cutoff = timestamp_text(now - timedelta(seconds=RECENT_GRACE_SECONDS))
    return _Clock(timestamp_text(now), min(live_cutoff, recent_cutoff) if recent_cutoff is not None else live_cutoff)


@dataclass(frozen=True)
class _Classifying:
    """What a classification binds: its clock and the media active queue items name, per kind."""

    clock: _Clock
    active_names: dict[MediaKind, frozenset[str]]

    def parameters(self, kind: MediaKind) -> dict[str, Any]:
        return {
            "now": self.clock.now,
            "recent_cutoff": self.clock.recent_cutoff,
            "active_names": self.active_names[kind],
        }


class ReferenceOwner(NamedTuple):
    """A saved document naming a media item, as `media_references` keys it."""

    owner_kind: str
    user_id: str
    owner_id: str


def _add_count(counts: IntermediatesKindCounts, classification: Classification, n: int) -> None:
    if classification == "safe":
        counts.safe += n
    elif classification == "referenced":
        counts.referenced += n
    elif classification == "active":
        counts.active += n
    else:
        counts.recent += n


@dataclass
class ScopeCounts:
    """Aggregated intermediates of one (owner, project) row for one media kind."""

    counts: IntermediatesKindCounts = field(default_factory=IntermediatesKindCounts)
    safe_bytes: int = 0
    referenced_bytes: int = 0
    unknown_size_count: int = 0


@dataclass
class DeletableTotals:
    count: int = 0
    referenced: int = 0
    measured_bytes: int = 0
    unknown_size_count: int = 0


@dataclass
class ScopePreview:
    """A preview's scope classified once, on one transaction and one clock, so its figures agree."""

    clock: _Clock
    # (owner, project) rows holding any intermediate in scope.
    rows: set[tuple[str, Optional[str]]] = field(default_factory=set)
    counts: dict[MediaKind, IntermediatesKindCounts] = field(
        default_factory=lambda: {"image": IntermediatesKindCounts(), "video": IntermediatesKindCounts()}
    )
    deletable: dict[MediaKind, DeletableTotals] = field(
        default_factory=lambda: {"image": DeletableTotals(), "video": DeletableTotals()}
    )
    # Force mode: the documents naming the deletable referenced items, with how many each names.
    acknowledged: dict[ReferenceOwner, int] = field(default_factory=dict)
    acknowledged_overflow: bool = False


def _scopes(*, user_id: Optional[str], targets: Optional[Sequence[ScopeTarget]], per_row: bool) -> list[ScopeSpec]:
    """The statements' scopes that together cover a scope: selected rows, an owner's or everyone's intermediates.

    `per_row` gives each selected row a statement of its own, an index seek in creation order, where an OR over
    many rows would sort the whole remaining scope on every window.
    """
    if targets is not None:
        if per_row:
            return [
                ScopeSpec("project", targets=((owner, project),))
                if project is not None
                else ScopeSpec("unassigned", user_id=owner)
                for owner, project in targets
            ]
        return [
            ScopeSpec("targets", targets=tuple(targets[start : start + TARGET_SLOTS]))
            for start in range(0, len(targets), TARGET_SLOTS)
        ]
    if user_id is not None:
        return [ScopeSpec("owner", user_id=user_id)]
    return [ScopeSpec("all")]


class IntermediatesRecords:
    def __init__(self, database: Database) -> None:
        self._queries = database.queries
        # Media names an active queue item's session names, keyed by item id and stamped with the row's session
        # revision, which changes with every session rewrite: a session rewritten while the item stays active (a
        # workflow-call parent resuming with its child's outputs) is scanned again, a status change alone is not.
        # Entries live as long as the item is active. Derived from committed rows only, so it is right whichever
        # transaction rolls back; transactions on a server fill it side by side, hence the lock.
        self._active_inputs: dict[int, tuple[int, frozenset[str], frozenset[str]]] = {}
        self._active_inputs_lock = threading.Lock()

    def start(self) -> None:
        """Drops the session holds and unmeasurable marks a previous process left behind: they were its own."""
        self._queries.intermediates.clear_process_state()

    def _expire_session_holds(self, clock: _Clock) -> None:
        """Releases the holds of sessions that ended and drops those whose grace is over, in a short transaction of
        its own: a classification that wrote them would hold their rows for as long as it reads, and a cleanup's
        guard waiting on one of them would hold up every write that protects media."""
        self._queries.intermediates.expire_session_holds(now=clock.now, recent_cutoff=clock.recent_cutoff)

    def _classifying(self, q: Queries, clock: _Clock) -> _Classifying:
        """Brings the scan of active queue items up to date, on the caller's transaction, and writes nothing.

        Runs on the caller's transaction, so that a classification and an enqueue it might race are ordered by the
        database, never by a stale cache. Holds not expired yet only protect more: a hold of a session that ended
        still counts as active until it is released.
        """
        active = q.intermediates.active_items()
        with self._active_inputs_lock:
            known = {item: entry for item, entry in self._active_inputs.items() if active.get(item) == entry[0]}
        changed = [item for item in active if item not in known]
        for item, stamp, session in q.intermediates.sessions(changed):
            text = session if isinstance(session, str) else ""
            known[item] = (
                stamp,
                frozenset(_IMAGE_NAME_RE.findall(text)),
                frozenset(_VIDEO_NAME_RE.findall(text)),
            )
        with self._active_inputs_lock:
            self._active_inputs = known
        images: set[str] = set()
        videos: set[str] = set()
        for _, image_names, video_names in known.values():
            images.update(image_names)
            videos.update(video_names)
        return _Classifying(clock, {"image": frozenset(images), "video": frozenset(videos)})

    # region holds

    def hold_cached_media(self, session_id: str, references: MediaReferences) -> bool:
        if references.is_empty():
            return True

        def hold(q: Queries) -> bool:
            # Shared with every protecting write, exclusive for a cleanup's check and delete: the cleanup either
            # deletes first (a cache miss), or sees this active session's hold.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            if not q.intermediates.is_session_active(session_id):
                return False
            held: list[tuple[MediaKind, str]] = []
            wanted_by_kind: tuple[tuple[MediaKind, Collection[str]], ...] = (
                ("image", references.images),
                ("video", references.videos),
            )
            for kind, names in wanted_by_kind:
                wanted = sorted(set(names))
                if wanted and q.intermediates.count_existing(kind, wanted) != len(wanted):
                    # Deletion won the race after the cache lookup. Do not return a stale output; invoking the
                    # node again creates fresh media instead.
                    return False
                held.extend((kind, name) for name in wanted)
            q.intermediates.hold_session_media(session_id, held)
            return True

        return self._queries.run(hold)

    def replace_browser_hold(self, user_id: str, lease_id: str, images: Sequence[str], videos: Sequence[str]) -> None:
        held_by_kind: tuple[tuple[MediaKind, Sequence[str]], ...] = (("image", images), ("video", videos))

        def replace(q: Queries) -> None:
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            now = _utc_now()
            q.intermediates.sweep_browser_holds(timestamp_text(now))
            q.intermediates.release_lease(user_id, lease_id)
            expires_at = timestamp_text(now + timedelta(seconds=BROWSER_HOLD_TTL_SECONDS))
            for kind, names in held_by_kind:
                if names:
                    q.intermediates.hold_for_lease(
                        kind, user_id=user_id, lease_id=lease_id, names=names, expires_at=expires_at
                    )
            # Past the cap, the leases refreshed longest ago stop protecting their media; an editor that is still
            # open restores its lease on its next refresh. Newest first, ties by lease id.
            others = sorted(lease for lease in q.intermediates.leases(user_id) if lease[0] != lease_id)
            others.sort(key=lambda lease: lease[1], reverse=True)
            lapsed = [lease for lease, _ in others[MAX_BROWSER_HOLD_LEASES_PER_USER - 1 :]]
            if lapsed:
                q.intermediates.release_leases(user_id, lapsed)

        self._queries.run(replace)

    def release_browser_hold(self, user_id: str, lease_id: str) -> None:
        self._queries.intermediates.release_lease(user_id, lease_id)

    # endregion

    # region summary

    def summarize(
        self, user_ids: Optional[Collection[str]], kinds: Sequence[MediaKind] = MEDIA_KINDS
    ) -> dict[tuple[str, Optional[str]], dict[MediaKind, ScopeCounts]]:
        """Aggregates the ``kinds`` intermediates of ``user_ids`` (None: everyone) by (owner, project) and classification."""
        scope = ScopeSpec("all") if user_ids is None else ScopeSpec("owners", users=frozenset(user_ids))
        clock = _clock()
        self._expire_session_holds(clock)

        def summarize(q: Queries) -> dict[tuple[str, Optional[str]], dict[MediaKind, ScopeCounts]]:
            rows: dict[tuple[str, Optional[str]], dict[MediaKind, ScopeCounts]] = defaultdict(
                lambda: {"image": ScopeCounts(), "video": ScopeCounts()}
            )
            classifying = self._classifying(q, clock)
            for kind in kinds:
                for row in q.intermediates.aggregate(kind, scope, **classifying.parameters(kind)):
                    counts = rows[(row.user_id, row.project_key)][kind]
                    _add_count(counts.counts, row.classification, row.count)
                    if row.classification == "safe":
                        counts.safe_bytes += row.measured_bytes
                        counts.unknown_size_count += row.unmeasured
                    elif row.classification == "referenced":
                        counts.referenced_bytes += row.measured_bytes
                        counts.unknown_size_count += row.unmeasured
            return dict(rows)

        return self._queries.run(summarize, read_only=True)

    def get_projects(self, user_id: Optional[str]) -> dict[tuple[str, str], tuple[str, Optional[str]]]:
        """Maps (owner, project) to (name, cover image): the newest durable image on any of the project's boards."""
        return {
            (owner, project_id): (name, cover)
            for owner, project_id, name, cover in self._queries.intermediates.projects(user_id)
        }

    def has_unmeasured_intermediates(self) -> bool:
        return any(self._queries.intermediates.any_unmeasured(kind) for kind in MEDIA_KINDS)

    def next_unmeasured(self, kind: MediaKind, limit: int, *, min_age_seconds: int = 0) -> list[tuple[str, str]]:
        """Unmeasured intermediates, oldest first; rows younger than ``min_age_seconds`` are left for later.

        A row is written before its file, so a brand-new one would measure as missing. Failures are marked
        (`mark_unmeasurable`) rather than skipped by a bounded list, so later rows remain reachable.
        """
        created_before = timestamp_text(_utc_now() - timedelta(seconds=min_age_seconds))
        return self._queries.intermediates.next_unmeasured(kind, limit=limit, created_before=created_before)

    def mark_unmeasurable(self, kind: MediaKind, names: Sequence[str]) -> None:
        self._queries.intermediates.mark_unmeasurable(kind, names)

    # endregion

    # region scope

    def preview_scope(
        self,
        *,
        user_id: Optional[str],
        targets: Optional[Sequence[ScopeTarget]],
        mode: IntermediatesCleanupMode,
        is_admin: bool,
        caller_user_id: str,
        max_acknowledged: int,
    ) -> ScopePreview:
        """Counts every intermediate in scope and what a cleanup of ``mode`` may delete.

        ``targets`` narrows to selected rows; otherwise ``user_id`` narrows to an owner, and None means everyone.
        One read-only transaction, which reads one snapshot, one instant and one view of active work serve every
        figure, so the kept counts, the deletable totals and the acknowledged documents agree.
        """
        scopes = _scopes(user_id=user_id, targets=targets, per_row=False)
        clock = _clock()
        self._expire_session_holds(clock)

        def preview(q: Queries) -> ScopePreview:
            classifying = self._classifying(q, clock)
            result = ScopePreview(clock=classifying.clock)
            for kind in MEDIA_KINDS:
                totals = result.deletable[kind]
                for scope in scopes:
                    for row in q.intermediates.preview(
                        kind,
                        scope,
                        mode=mode,
                        is_admin=is_admin,
                        caller_user_id=caller_user_id,
                        **classifying.parameters(kind),
                    ):
                        result.rows.add((row.user_id, row.project_key))
                        _add_count(result.counts[kind], row.classification, row.count)
                        if row.deletable:
                            totals.count += row.count
                            totals.measured_bytes += row.measured_bytes
                            totals.unknown_size_count += row.unmeasured
                            if row.classification == "referenced":
                                totals.referenced += row.count
                    if mode != "force":
                        continue
                    for owner_kind, owner_user, owner_id, n in q.intermediates.acknowledged(
                        kind, scope, is_admin=is_admin, caller_user_id=caller_user_id, **classifying.parameters(kind)
                    ):
                        owner = ReferenceOwner(owner_kind, owner_user, owner_id)
                        if owner not in result.acknowledged and len(result.acknowledged) >= max_acknowledged:
                            result.acknowledged_overflow = True
                            continue
                        result.acknowledged[owner] = result.acknowledged.get(owner, 0) + n
            return result

        return self._queries.run(preview, read_only=True)

    def iter_deletable_batches(
        self,
        kind: MediaKind,
        *,
        user_id: Optional[str],
        targets: Optional[Sequence[ScopeTarget]],
        mode: IntermediatesCleanupMode,
        is_admin: bool,
        caller_user_id: str,
        recent_cutoff: Optional[str],
        limit: int,
    ) -> Iterator[list[tuple[str, Optional[int]]]]:
        """The deletable intermediates of a scope in creation order, a bounded window at a time.

        Each window classifies ``limit`` rows on one short transaction and yields the deletable ones (possibly
        none, so the caller can re-check its authority between windows), so a long protected stretch never holds
        the database for more than one window, and a row is classified once per operation however many are kept.
        ``recent_cutoff`` freezes recency at the preview that was confirmed; None judges it live.
        """
        for scope in _scopes(user_id=user_id, targets=targets, per_row=True):
            after = ("", "")
            while True:
                clock = _clock(recent_cutoff)
                self._expire_session_holds(clock)
                rows = self._queries.run(
                    self._window(kind, scope, mode, is_admin, caller_user_id, clock, after, limit), read_only=True
                )
                if not rows:
                    break
                after = (rows[-1].created_at, rows[-1].name)
                yield [(row.name, row.file_size_bytes) for row in rows if row.deletable]

    def _window(
        self,
        kind: MediaKind,
        scope: ScopeSpec,
        mode: IntermediatesCleanupMode,
        is_admin: bool,
        caller_user_id: str,
        clock: _Clock,
        after: tuple[str, str],
        limit: int,
    ) -> "Callable[[Queries], list[WindowRow]]":
        def window(q: Queries) -> list[WindowRow]:
            classifying = self._classifying(q, clock)
            return q.intermediates.window(
                kind,
                scope,
                mode=mode,
                is_admin=is_admin,
                caller_user_id=caller_user_id,
                after=after,
                limit=limit,
                **classifying.parameters(kind),
            )

        return window

    def get_document_names(self, owners: Collection[ReferenceOwner]) -> dict[ReferenceOwner, str]:
        """Names of the referencing documents, one query per kind; client state has none."""
        wanted: dict[str, set[ReferenceOwner]] = defaultdict(set)
        for owner in owners:
            if owner.owner_kind in ("project", "quarantined_project", "workflow"):
                wanted[owner.owner_kind].add(owner)

        def names(q: Queries) -> dict[ReferenceOwner, str]:
            found: dict[ReferenceOwner, str] = {}
            for owner_kind, kind_owners in wanted.items():
                ids = sorted({owner.owner_id for owner in kind_owners})
                named = {
                    (owner, owner_id): name
                    for owner, owner_id, name in q.intermediates.document_names(owner_kind, ids)
                    if name is not None
                }
                for owner in kind_owners:
                    name = named.get((None if owner_kind == "workflow" else owner.user_id, owner.owner_id))
                    if name is not None:
                        found[owner] = name
            return found

        return self._queries.run(names, read_only=True)

    # endregion

    # region delete guard

    def make_delete_guard(
        self,
        kind: MediaKind,
        *,
        mode: IntermediatesCleanupMode,
        allowed_user_ids: Optional[frozenset[str]],
        caller_user_id: str,
        is_admin: bool,
        recent_cutoff: Optional[str] = None,
        acknowledged: Optional[frozenset[ReferenceOwner]] = None,
    ) -> IntermediateDeleteGuard:
        """The final check of a cleanup batch, run on the deleting transaction.

        Re-applies the policy at the moment of deletion, judging recency as of ``recent_cutoff``: anything that
        became active, referenced (safe mode), non-intermediate or another account's since the batch was read is
        kept. A force clear deletes a referenced item only while every document naming it is in ``acknowledged``.
        ``allowed_user_ids`` None means every account.
        """
        acknowledged_documents = acknowledged or frozenset()

        def guard(q: Queries, names: Sequence[str]) -> list[str]:
            classifying = self._classifying(q, _clock(recent_cutoff))
            named = set(names)
            references: dict[str, set[ReferenceOwner]] = defaultdict(set)
            if mode == "force":
                for media_name, owner_kind, owner_user, owner_id in q.intermediates.reference_owners(kind, named):
                    references[media_name].add(ReferenceOwner(owner_kind, owner_user, owner_id))
            # Ownership is re-read here rather than trusted from the preview.
            allowed = {
                name
                for name, owner in q.intermediates.still_deletable(
                    kind,
                    named,
                    mode=mode,
                    is_admin=is_admin,
                    caller_user_id=caller_user_id,
                    **classifying.parameters(kind),
                )
                if (allowed_user_ids is None or owner in allowed_user_ids)
                and references[name] <= acknowledged_documents
            }
            return [name for name in names if name in allowed]

        return guard

    # endregion
