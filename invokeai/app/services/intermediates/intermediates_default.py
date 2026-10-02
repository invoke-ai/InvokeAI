"""The intermediates cleanup engine.

A preview classifies its scope once and freezes the instant recency is judged at; a confirmation
turns it into an operation that pages the same scope live, in bounded batches, on one background
worker. Every batch re-applies the policy on the deleting transaction with the preview's recency
cutoff, so nothing created after the preview is collected and a target that became active,
referenced or durable since the preview is kept. Operations live in memory: a restart forgets
them, and because the live scope is the retry, a new preview and confirmation finishes whatever an
interrupted run left.

Lock order, for anyone adding a caller: image mutation lock or video deletion lock → database
lock. Image and video deletion never hold each other's locks; queue and document writers take
only the database lock.
"""

import logging
import queue
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence, cast

from invokeai.app.services.intermediates.intermediates_base import IntermediatesCaller, IntermediatesServiceBase
from invokeai.app.services.intermediates.intermediates_common import (
    MAX_ACKNOWLEDGED_DOCUMENTS,
    MAX_AFFECTED_DOCUMENTS,
    PREVIEW_TTL_SECONDS,
    RECENT_GRACE_SECONDS,
    IntermediatesAffectedDocument,
    IntermediatesBrowserHoldRequest,
    IntermediatesImpact,
    IntermediatesOperation,
    IntermediatesOperationNotFoundError,
    IntermediatesOperationProgress,
    IntermediatesOperationRequest,
    IntermediatesPreview,
    IntermediatesPreviewNotFoundError,
    IntermediatesPreviewRequest,
    IntermediatesRow,
    IntermediatesScope,
    IntermediatesScopeForbiddenError,
    IntermediatesScopeInvalidError,
    IntermediatesSummary,
    IntermediatesSummarySort,
    IntermediatesSummaryTotals,
    IntermediatesUnavailableError,
)
from invokeai.app.services.intermediates.intermediates_measurement import IntermediatesSizeMeasurer
from invokeai.app.services.intermediates.intermediates_records_sqlite import (
    IntermediatesRecordsSqlite,
    MediaKind,
    ReferenceOwner,
    ScopeTarget,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.intermediate_delete import IntermediateDeleteGuard, IntermediateDeleteResult
from invokeai.app.services.shared.media_references import MediaReferenceOwnerKind, MediaReferences

DELETE_BATCH_SIZE = 200
# Operations run one at a time; one running cleanup per account keeps one account from filling
# the queue, and the UI offers no second Delete while one runs anyway.
MAX_ACTIVE_OPERATIONS = 8
MAX_ACTIVE_OPERATIONS_PER_CALLER = 1
# Settled operations stay readable for a while so a reopened manager can show the last result.
MAX_RETAINED_SETTLED_OPERATIONS_PER_CALLER = 5
MAX_RETAINED_OPERATIONS = 200
MAX_PREVIEWS = 200
MAX_PREVIEWS_PER_CALLER = 4
# A `matching` scope resolves to explicit rows; past this many the filter is too wide to page.
MAX_RESOLVED_SCOPE_TARGETS = 10_000
PROGRESS_EVENT_INTERVAL_SECONDS = 1.0
WORKER_STOP_TIMEOUT_SECONDS = 10.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class _ResolvedScope:
    """What a scope pages over: an owner (None: everyone) or explicit rows, and whose media it may touch."""

    user_id: Optional[str]
    # Plain tuples: a resolved scope may hold thousands of rows and outlives its preview.
    targets: Optional[list[ScopeTarget]]
    allowed_user_ids: Optional[frozenset[str]]


@dataclass
class _Preview:
    dto: IntermediatesPreview
    caller_user_id: str
    scope: _ResolvedScope
    # Recency is judged as of this instant for the whole operation the preview confirms.
    recent_cutoff: str
    # Force mode: the documents whose media the confirmer agreed to break.
    acknowledged: frozenset[ReferenceOwner]


@dataclass
class _Operation:
    dto: IntermediatesOperation
    caller: IntermediatesCaller
    scope: _ResolvedScope
    recent_cutoff: str
    acknowledged: frozenset[ReferenceOwner]
    last_progress_event_at: float = 0.0


class IntermediatesService(IntermediatesServiceBase):
    def __init__(self, records: IntermediatesRecordsSqlite, logger: Optional[logging.Logger] = None) -> None:
        self._records = records
        self._logger = logger or logging.getLogger(__name__)
        self._invoker: Optional[Invoker] = None
        self._lock = threading.Lock()
        self._previews: dict[str, _Preview] = {}
        # Insertion order is age; retention drops the oldest settled operations first.
        self._operations: dict[str, _Operation] = {}
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._measurer = IntermediatesSizeMeasurer(records, lambda: self._services, self._stop, self._logger)
        self._worker: Optional[threading.Thread] = None

    # region lifecycle

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._stop.clear()
        self._measurer.reset()

    def stop(self, invoker: Optional[Invoker] = None) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None:
            self._pending.put("")
            worker.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
            if worker.is_alive():
                self._logger.warning("Intermediates worker did not stop in time")
            else:
                self._worker = None

    def _ensure_worker(self) -> None:
        """Spawned on first use: a process that never cleans up never pays for a polling thread."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._stop.is_set():
                return
            self._worker = threading.Thread(target=self._worker_loop, name="intermediates_worker", daemon=True)
            self._worker.start()

    @property
    def _services(self):
        assert self._invoker is not None, "IntermediatesService has not been started"
        return self._invoker.services

    # endregion

    # region summary

    def hold_cached_media(self, session_id: str, references: MediaReferences) -> bool:
        return self._records.hold_cached_media(session_id, references)

    def replace_browser_hold(
        self, caller: IntermediatesCaller, lease_id: str, request: IntermediatesBrowserHoldRequest
    ) -> None:
        self._records.replace_browser_hold(caller.user_id, lease_id, request.images, request.videos)

    def release_browser_hold(self, caller: IntermediatesCaller, lease_id: str) -> None:
        self._records.release_browser_hold(caller.user_id, lease_id)

    def get_summary(
        self,
        caller: IntermediatesCaller,
        *,
        owner_id: Optional[str],
        search: Optional[str],
        sort: IntermediatesSummarySort,
        descending: bool,
        offset: int,
        limit: int,
        project_id: Optional[str] = None,
    ) -> IntermediatesSummary:
        rows = self._summary_rows(caller, owner_id=owner_id, project_id=project_id, search=search)

        rows.sort(key=lambda row: (row.project_name or "").casefold())
        if sort == "reclaimable_bytes":
            rows.sort(key=lambda row: row.reclaimable_bytes, reverse=descending)
        elif descending:
            rows.reverse()

        totals = IntermediatesSummaryTotals(rows=len(rows))
        for row in rows:
            totals.safe_images += row.images.safe
            totals.safe_videos += row.videos.safe
            totals.in_use_images += row.images.total - row.images.safe
            totals.in_use_videos += row.videos.total - row.videos.safe
            totals.reclaimable_bytes += row.reclaimable_bytes
            totals.unknown_size_count += row.unknown_size_count

        measuring = self._records.has_unmeasured_intermediates()
        if measuring:
            self._measurer.request()
            self._ensure_worker()

        return IntermediatesSummary(
            items=rows[offset : offset + limit],
            total=len(rows),
            offset=offset,
            limit=limit,
            totals=totals,
            recent_grace_seconds=RECENT_GRACE_SECONDS,
            measuring=measuring,
            can_manage_everyone=caller.is_admin,
        )

    def _summary_rows(
        self, caller: IntermediatesCaller, *, owner_id: Optional[str], project_id: Optional[str], search: Optional[str]
    ) -> list[IntermediatesRow]:
        """Every (owner, project) row the caller may see under the given filters, unordered."""
        owner_filter = self._resolve_owner_filter(caller, owner_id)
        aggregated = self._records.summarize([owner_filter] if owner_filter is not None else None)
        projects = self._records.get_projects(owner_filter)
        users = self._services.users.get_many([user_id for user_id, _ in aggregated])

        rows: list[IntermediatesRow] = []
        for (user_id, row_project_id), kinds in aggregated.items():
            user = users.get(user_id)
            project = projects.get((user_id, row_project_id)) if row_project_id is not None else None
            rows.append(
                IntermediatesRow(
                    user_id=user_id,
                    project_id=row_project_id,
                    user_display_name=user.display_name if user is not None else None,
                    user_email=user.email if user is not None else None,
                    project_name=project[0] if project is not None else None,
                    cover_image_name=project[1] if project is not None else None,
                    images=kinds["image"].counts,
                    videos=kinds["video"].counts,
                    reclaimable_bytes=kinds["image"].safe_bytes + kinds["video"].safe_bytes,
                    referenced_bytes=kinds["image"].referenced_bytes + kinds["video"].referenced_bytes,
                    unknown_size_count=kinds["image"].unknown_size_count + kinds["video"].unknown_size_count,
                )
            )

        if project_id is not None:
            rows = [row for row in rows if row.project_id == project_id]
        if search:
            needle = search.casefold()
            rows = [row for row in rows if self._matches(row, needle, include_owner=caller.is_admin)]
        return rows

    @staticmethod
    def _matches(row: IntermediatesRow, needle: str, *, include_owner: bool) -> bool:
        haystacks = [row.project_name or ""]
        if include_owner:
            haystacks.extend([row.user_display_name or "", row.user_email or ""])
        return any(needle in value.casefold() for value in haystacks)

    @staticmethod
    def _resolve_owner_filter(caller: IntermediatesCaller, owner_id: Optional[str]) -> Optional[str]:
        if not caller.is_admin:
            if owner_id is not None and owner_id != caller.user_id:
                raise IntermediatesScopeForbiddenError("Only administrators can inspect other accounts")
            return caller.user_id
        return owner_id

    # endregion

    # region previews

    def create_preview(self, request: IntermediatesPreviewRequest, caller: IntermediatesCaller) -> IntermediatesPreview:
        scope = self._resolve_scope(request.scope, caller)
        classified = self._records.preview_scope(
            user_id=scope.user_id,
            targets=scope.targets,
            mode=request.mode,
            is_admin=caller.is_admin,
            caller_user_id=caller.user_id,
            max_acknowledged=MAX_ACKNOWLEDGED_DOCUMENTS,
        )
        if classified.acknowledged_overflow:
            raise IntermediatesScopeInvalidError(
                f"This force delete would break more than {MAX_ACKNOWLEDGED_DOCUMENTS} documents; narrow the scope"
            )

        images, videos = classified.counts["image"], classified.counts["video"]
        deletable_images, deletable_videos = classified.deletable["image"], classified.deletable["video"]
        impact = IntermediatesImpact(
            delete_images=deletable_images.count,
            delete_videos=deletable_videos.count,
            keep_referenced_images=images.referenced - deletable_images.referenced,
            keep_referenced_videos=videos.referenced - deletable_videos.referenced,
            keep_active_images=images.active,
            keep_active_videos=videos.active,
            keep_recent_images=images.recent,
            keep_recent_videos=videos.recent,
            reclaimable_bytes=deletable_images.measured_bytes + deletable_videos.measured_bytes,
            unknown_size_count=deletable_images.unknown_size_count + deletable_videos.unknown_size_count,
        )
        affected = self._describe_affected(classified.acknowledged) if request.mode == "force" else []

        created = _now()
        dto = IntermediatesPreview(
            preview_id=uuid.uuid4().hex,
            mode=request.mode,
            scope=request.scope,
            created_at=created,
            expires_at=created + timedelta(seconds=PREVIEW_TTL_SECONDS),
            target_rows=len(scope.targets) if scope.targets is not None else len(classified.rows),
            impact=impact,
            affected_documents=affected,
            affected_documents_total=len(classified.acknowledged),
        )
        with self._lock:
            self._previews[dto.preview_id] = _Preview(
                dto=dto,
                caller_user_id=caller.user_id,
                scope=scope,
                recent_cutoff=classified.clock.recent_cutoff,
                acknowledged=frozenset(classified.acknowledged),
            )
            self._expire_previews_locked(created, keep=dto.preview_id)
        return dto

    def _resolve_scope(self, scope: IntermediatesScope, caller: IntermediatesCaller) -> _ResolvedScope:
        """Authorizes a scope and turns it into what the records layer pages over."""
        if scope.kind == "everyone":
            if not caller.is_admin:
                raise IntermediatesScopeForbiddenError("Only administrators can delete everyone's intermediates")
            return _ResolvedScope(user_id=None, targets=None, allowed_user_ids=None)
        if scope.kind == "owner":
            if scope.user_id is None:
                raise IntermediatesScopeInvalidError("An owner scope names the account whose intermediates to delete")
            if scope.user_id != caller.user_id and not caller.is_admin:
                raise IntermediatesScopeForbiddenError("Only administrators can delete another account's intermediates")
            return _ResolvedScope(user_id=scope.user_id, targets=None, allowed_user_ids=frozenset({scope.user_id}))
        if scope.kind == "matching":
            owner_id = self._resolve_owner_filter(caller, scope.user_id)
            rows = self._summary_rows(caller, owner_id=owner_id, project_id=scope.project_id, search=scope.search)
            excluded = {(target.user_id, target.project_id) for target in scope.excluded}
            targets = [(row.user_id, row.project_id) for row in rows if (row.user_id, row.project_id) not in excluded]
            if not targets:
                raise IntermediatesScopeInvalidError("No rows match the filter")
            if len(targets) > MAX_RESOLVED_SCOPE_TARGETS:
                raise IntermediatesScopeInvalidError(
                    f"More than {MAX_RESOLVED_SCOPE_TARGETS} rows match the filter; narrow it"
                )
        else:
            if not scope.targets:
                raise IntermediatesScopeInvalidError("A selection scope names at least one row")
            # Duplicate rows would double-count; the order is irrelevant to the SQL.
            targets = list(dict.fromkeys((t.user_id, t.project_id) for t in scope.targets))
        owners = frozenset(user_id for user_id, _ in targets)
        if not caller.is_admin and owners != frozenset({caller.user_id}):
            raise IntermediatesScopeForbiddenError("Only administrators can delete another account's intermediates")
        return _ResolvedScope(user_id=None, targets=targets, allowed_user_ids=owners)

    def _describe_affected(self, acknowledged: dict[ReferenceOwner, int]) -> list[IntermediatesAffectedDocument]:
        """The documents a force clear breaks, bounded to the first few by kind and name."""
        names = self._records.get_document_names(acknowledged)
        ordered = sorted(
            acknowledged, key=lambda owner: (owner.owner_kind, (names.get(owner) or "").casefold(), owner.owner_id)
        )[:MAX_AFFECTED_DOCUMENTS]
        users = self._services.users.get_many(sorted({owner.user_id for owner in ordered}))
        return [
            IntermediatesAffectedDocument(
                kind=cast(MediaReferenceOwnerKind, owner.owner_kind),
                user_id=owner.user_id,
                user_display_name=users[owner.user_id].display_name if owner.user_id in users else None,
                user_email=users[owner.user_id].email if owner.user_id in users else None,
                owner_id=owner.owner_id,
                name=names.get(owner),
                references=acknowledged[owner],
            )
            for owner in ordered
        ]

    def _expire_previews_locked(self, now: datetime, *, keep: Optional[str] = None) -> None:
        """Drops expired previews, then each caller's oldest past their cap, then the oldest past the global cap.

        ``keep`` (the one just created) always survives. The per-caller cap is applied first, so
        one account previewing in a loop evicts only its own pending confirmations.
        """
        for pid in [pid for pid, preview in self._previews.items() if preview.dto.expires_at <= now]:
            del self._previews[pid]
        by_caller: dict[str, list[str]] = defaultdict(list)
        for pid, preview in self._previews.items():
            by_caller[preview.caller_user_id].append(pid)
        for pids in by_caller.values():
            for pid in [pid for pid in pids if pid != keep][: max(0, len(pids) - MAX_PREVIEWS_PER_CALLER)]:
                del self._previews[pid]
        evictable = [pid for pid in self._previews if pid != keep]
        for pid in evictable[: max(0, len(self._previews) - MAX_PREVIEWS)]:
            del self._previews[pid]

    # endregion

    # region operations

    def start_operation(
        self, request: IntermediatesOperationRequest, caller: IntermediatesCaller
    ) -> IntermediatesOperation:
        with self._lock:
            self._require_cleanup_available()
            self._expire_previews_locked(_now())
            preview = self._previews.get(request.preview_id)
            # A consumed preview answers "not found" even while its run is still active, so a client
            # whose start response was lost learns to look the run up rather than to wait.
            if preview is None or preview.caller_user_id != caller.user_id:
                raise IntermediatesPreviewNotFoundError("Preview expired or unknown; request a new one")
            self._require_active_capacity_locked(caller.user_id)
            # Single use: a second confirmation of the same preview must go through a new preview.
            del self._previews[request.preview_id]
            dto = IntermediatesOperation(
                operation_id=uuid.uuid4().hex,
                user_id=caller.user_id,
                mode=preview.dto.mode,
                scope=preview.dto.scope,
                status="pending",
                created_at=_now(),
                target_images=preview.dto.impact.delete_images,
                target_videos=preview.dto.impact.delete_videos,
                progress=IntermediatesOperationProgress(),
            )
            operation = _Operation(
                dto=dto,
                caller=caller,
                scope=preview.scope,
                recent_cutoff=preview.recent_cutoff,
                acknowledged=preview.acknowledged,
            )
            self._operations[dto.operation_id] = operation
            self._retain_operations_locked()
            snapshot = dto.model_copy(deep=True)
        # The pending event goes out before the worker can see the id, so no later event precedes it.
        self._emit(snapshot)
        self._ensure_worker()
        self._pending.put(dto.operation_id)
        return snapshot

    def get_operation(self, operation_id: str, caller: IntermediatesCaller) -> IntermediatesOperation:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None or (operation.caller.user_id != caller.user_id and not caller.is_admin):
                raise IntermediatesOperationNotFoundError(operation_id)
            return operation.dto.model_copy(deep=True)

    def list_operations(self, caller: IntermediatesCaller) -> list[IntermediatesOperation]:
        with self._lock:
            own = [op for op in self._operations.values() if op.caller.user_id == caller.user_id]
            return [op.dto.model_copy(deep=True) for op in reversed(own)]

    def _require_active_capacity_locked(self, caller_user_id: str) -> None:
        active = [op for op in self._operations.values() if op.dto.status in ("pending", "running")]
        if sum(op.caller.user_id == caller_user_id for op in active) >= MAX_ACTIVE_OPERATIONS_PER_CALLER:
            raise IntermediatesUnavailableError("Wait for your running deletion to finish before starting another")
        if len(active) >= MAX_ACTIVE_OPERATIONS:
            raise IntermediatesUnavailableError("Too many cleanup operations are already running")

    def _retain_operations_locked(self) -> None:
        """Drops each account's oldest settled operations past its budget, then the oldest past the global cap."""
        settled_by_caller: dict[str, list[str]] = defaultdict(list)
        for operation_id, operation in self._operations.items():
            if operation.dto.status in ("completed", "failed"):
                settled_by_caller[operation.caller.user_id].append(operation_id)
        for operation_ids in settled_by_caller.values():
            for operation_id in operation_ids[
                : max(0, len(operation_ids) - MAX_RETAINED_SETTLED_OPERATIONS_PER_CALLER)
            ]:
                del self._operations[operation_id]
        settled = [oid for oid, op in self._operations.items() if op.dto.status in ("completed", "failed")]
        for operation_id in settled[: max(0, len(self._operations) - MAX_RETAINED_OPERATIONS)]:
            del self._operations[operation_id]

    def _emit(self, snapshot: IntermediatesOperation) -> None:
        """Events carry snapshots taken under the lock; a failing bus must not take the worker down with it."""
        try:
            self._services.events.emit_intermediates_operation_changed(snapshot)
        except Exception as error:  # pragma: no cover - defensive: the worker must survive
            self._logger.error(f"Could not publish intermediates operation {snapshot.operation_id}: {error}")

    def _require_cleanup_available(self) -> None:
        image_moves = getattr(self._services, "image_moves", None)
        if image_moves is not None and image_moves.is_maintenance_active():
            raise IntermediatesUnavailableError("Image storage maintenance is active")

    # endregion

    # region legacy

    def clear_all_images_now(self, caller: IntermediatesCaller) -> int:
        if not caller.is_admin:
            raise IntermediatesScopeForbiddenError("Only admins can clear all intermediates")
        guard = self._records.make_delete_guard(
            "image", mode="safe", allowed_user_ids=None, caller_user_id=caller.user_id, is_admin=True
        )
        deleted = 0
        # Bounded windows keep memory flat however many intermediates exist; the guard re-checks
        # each batch on the deleting transaction.
        for batch in self._records.iter_deletable_batches(
            "image",
            user_id=None,
            targets=None,
            mode="safe",
            is_admin=True,
            caller_user_id=caller.user_id,
            recent_cutoff=None,
            limit=DELETE_BATCH_SIZE,
        ):
            if batch:
                names = [name for name, _ in batch]
                deleted += len(self._services.images.delete_intermediates_by_names(names, guard).deleted_names)
        return deleted

    def count_safe_images(self, caller: IntermediatesCaller) -> int:
        counts = self._records.summarize(None if caller.is_admin else [caller.user_id], kinds=("image",))
        return sum(kinds["image"].counts.safe for kinds in counts.values())

    # endregion

    # region worker

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                # Due measurement only takes the turns operations leave free, so it polls the queue
                # instead of waiting on it.
                if self._measurer.is_due():
                    operation_id = self._pending.get_nowait()
                else:
                    operation_id = self._pending.get(timeout=0.5)
            except queue.Empty:
                if self._measurer.is_due():
                    self._measurer.measure_once()
                continue
            if not operation_id:
                continue
            with self._lock:
                operation = self._operations.get(operation_id)
            if operation is not None:
                try:
                    self._run(operation)
                except Exception as error:  # pragma: no cover - defensive: the worker must survive
                    self._logger.error(f"Intermediates operation {operation_id} crashed: {error}", exc_info=True)
                    self._finish(operation, error=str(error))

    def _run(self, operation: _Operation) -> None:
        with self._lock:
            operation.dto.status = "running"
            operation.dto.started_at = _now()
            snapshot = operation.dto.model_copy(deep=True)
        self._emit(snapshot)

        deleters: dict[MediaKind, Callable[[list[str], IntermediateDeleteGuard], IntermediateDeleteResult]] = {
            "image": self._services.images.delete_intermediates_by_names,
            "video": self._services.videos.delete_intermediates_by_names,
        }
        for media_kind, deleter in deleters.items():
            batches = self._records.iter_deletable_batches(
                media_kind,
                user_id=operation.scope.user_id,
                targets=operation.scope.targets,
                mode=operation.dto.mode,
                is_admin=operation.caller.is_admin,
                caller_user_id=operation.caller.user_id,
                recent_cutoff=operation.recent_cutoff,
                limit=DELETE_BATCH_SIZE,
            )
            for batch in batches:
                # Re-checked per window, including windows of only protected rows, so a stop or a
                # demotion is honoured within one window even across a long protected stretch.
                halt, is_admin = self._batch_authority(operation)
                if halt is not None:
                    self._finish(operation, error=halt)
                    return
                if not batch:
                    continue
                acknowledged = (
                    operation.acknowledged
                    if is_admin
                    else frozenset(o for o in operation.acknowledged if o.user_id == operation.caller.user_id)
                )
                guard = self._records.make_delete_guard(
                    media_kind,
                    mode=operation.dto.mode,
                    allowed_user_ids=operation.scope.allowed_user_ids,
                    caller_user_id=operation.caller.user_id,
                    is_admin=is_admin,
                    recent_cutoff=operation.recent_cutoff,
                    acknowledged=acknowledged,
                )
                names = [name for name, _ in batch]
                sizes = dict(batch)
                try:
                    result = deleter(names, guard)
                except Exception as error:
                    self._logger.error(f"Intermediates cleanup batch failed ({media_kind}): {error}", exc_info=True)
                    self._record_batch(operation, media_kind, names, deleted=[], deferred=[], failed=names, sizes=sizes)
                    continue
                self._record_batch(
                    operation,
                    media_kind,
                    names,
                    deleted=result.deleted_names,
                    deferred=result.purge_deferred,
                    failed=[],
                    sizes=sizes,
                )
        self._finish(operation, error=None)

    def _batch_authority(self, operation: _Operation) -> tuple[Optional[str], bool]:
        """Why the next batch must not run, if it must not, and whether the confirmer administers the instance now."""
        if self._stop.is_set():
            return "The server is shutting down", False
        image_moves = getattr(self._services, "image_moves", None)
        if image_moves is not None and image_moves.is_maintenance_active():
            return "Image storage maintenance started", False
        # Single-user mode has no accounts to re-check: every request is the local administrator,
        # and the `system` row that names it is deliberately not an admin.
        if not self._services.configuration.multiuser:
            return None, operation.caller.is_admin
        # Authorization is re-read per batch: an account demoted, deactivated or deleted
        # mid-operation stops deleting at the next batch boundary, and a demoted one no longer
        # breaks other accounts' documents even inside its own scope.
        user = self._services.users.get(operation.caller.user_id)
        if user is None or not user.is_active:
            return "The confirming account is no longer active", False
        if operation.scope.allowed_user_ids != frozenset({operation.caller.user_id}) and not user.is_admin:
            return "The confirming account no longer administers this instance", False
        return None, operation.caller.is_admin and user.is_admin

    def _record_batch(
        self,
        operation: _Operation,
        kind: MediaKind,
        names: list[str],
        *,
        deleted: Sequence[str],
        deferred: Sequence[str],
        failed: Sequence[str],
        sizes: dict[str, Optional[int]],
    ) -> None:
        deleted_set = set(deleted)
        deferred_set = set(deferred)
        with self._lock:
            progress = operation.dto.progress
            retained = len(names) - len(deleted_set) - len(failed)
            for name in deleted_set:
                if name in deferred_set:
                    progress.pending_disk_cleanup += 1
                    continue
                size = sizes.get(name)
                if size is None:
                    progress.unknown_size_count += 1
                else:
                    progress.reclaimed_bytes += size
            if kind == "image":
                progress.processed_images += len(names)
                progress.deleted_images += len(deleted_set)
                progress.failed_images += len(failed)
                progress.retained_images += retained
            else:
                progress.processed_videos += len(names)
                progress.deleted_videos += len(deleted_set)
                progress.failed_videos += len(failed)
                progress.retained_videos += retained
            # Progress events are coalesced; the final state always goes out from `_finish`.
            now = time.monotonic()
            if now - operation.last_progress_event_at < PROGRESS_EVENT_INTERVAL_SECONDS:
                return
            operation.last_progress_event_at = now
            snapshot = operation.dto.model_copy(deep=True)
        self._emit(snapshot)

    def _finish(self, operation: _Operation, *, error: Optional[str]) -> None:
        with self._lock:
            operation.dto.status = "failed" if error is not None else "completed"
            operation.dto.error = error
            operation.dto.completed_at = _now()
            self._retain_operations_locked()
            snapshot = operation.dto.model_copy(deep=True)
        self._emit(snapshot)

    # endregion
