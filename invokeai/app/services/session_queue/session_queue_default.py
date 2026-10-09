import asyncio
import hashlib
import json
from datetime import timedelta
from typing import Any, Literal, Optional

from pydantic_core import to_jsonable_python

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase, WorkflowCallChildCompletion
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    BatchStatus,
    CancelAllExceptCurrentResult,
    CancelByBatchIDsResult,
    CancelByDestinationResult,
    CancelByQueueIDResult,
    ClearResult,
    DeleteAllExceptCurrentResult,
    DeleteByDestinationResult,
    EnqueueBatchReceipt,
    EnqueueBatchResult,
    EnqueueIdempotencyConflictError,
    EnqueueProjectNotFoundError,
    EnqueueReceiptLimitError,
    IsEmptyResult,
    IsFullResult,
    ItemIdsResult,
    NodeFieldValue,
    PruneResult,
    RetryItemsResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemChangedError,
    SessionQueueItemNotFoundError,
    SessionQueueItemSummary,
    SessionQueueStatus,
    TooManySessionsError,
    ValueToInsertTuple,
    calc_session_count,
    prepare_values_to_insert,
    uuid_string,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.queries.session_queue import Scope, SettledReceipt
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.pagination import CursorPaginatedResults, SQLiteDirection

MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER = 10_000
MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER = 64 * 1024 * 1024
MAX_ENQUEUE_RECEIPTS_PER_OWNER = 100_000
MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER = 128 * 1024 * 1024
ACKNOWLEDGED_ENQUEUE_RECEIPT_RETENTION = timedelta(days=7)

# How far past the fairness-chosen candidate (in item_id distance) the device-affinity swap may
# look for a warm-model item. This bounds two things at once: the scoring query's cost (at most
# this many session blobs are scanned per dequeue) and how long a cold item can be deferred — the
# starved candidate's item_id never changes, so newly enqueued warm items eventually fall outside
# the window and the cold item runs after at most ~this many swaps.
AFFINITY_MAX_LOOKAHEAD = 32

_FINISHED = ("completed", "failed", "canceled")

# Every write that adds items takes the queue's admission lock, so that two of them neither both fit into its last
# free places nor both write one enqueue receipt, and shares the media protection: the items' sessions make the media
# they name active, which the intermediates cleanup must not delete between its check and its delete.
_ADMISSION = (DatabaseLock.SESSION_QUEUE_ADMISSION,)
_PROTECTING_MEDIA = (DatabaseLock.MEDIA_PROTECTION,)


def _session_json(session: GraphExecutionState) -> str:
    # Persisted sessions resume execution across queue boundaries.
    return json.dumps(dump_execution_state(session), default=to_jsonable_python)


def _decode_enqueue_receipt(batch_id: str, requested: int, enqueued: int, item_ids_json: str) -> EnqueueBatchReceipt:
    item_ids = json.loads(item_ids_json)
    if not isinstance(item_ids, list) or not all(isinstance(item_id, int) for item_id in item_ids):
        raise RuntimeError("Stored queue enqueue receipt contains invalid item ids")
    return EnqueueBatchReceipt(batch_id=batch_id, requested=requested, enqueued=enqueued, item_ids=item_ids)


class SessionQueue(SessionQueueBase):
    __invoker: Invoker

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self._set_in_progress_to_canceled()
        config = self.__invoker.services.configuration
        if config.clear_queue_on_startup:
            clear_result = self.clear(DEFAULT_QUEUE_ID)
            if clear_result.deleted > 0:
                self.__invoker.services.logger.info(f"Cleared all {clear_result.deleted} queue items")
            return

        if config.max_queue_history is not None:
            deleted = self._prune_terminal_to_limit(DEFAULT_QUEUE_ID, config.max_queue_history)
            if deleted > 0:
                self.__invoker.services.logger.info(
                    f"Pruned {deleted} completed/failed/canceled queue items (kept up to {config.max_queue_history})"
                )

    def _set_in_progress_to_canceled(self) -> None:
        """
        Sets all in_progress or waiting queue items to canceled. Run on app startup, not associated with any queue.
        This is necessary because the invoker may have been killed while processing a queue item or while a parent
        queue item was suspended waiting on a child workflow execution.
        """
        self._queries.session_queue.cancel_interrupted()

    def _prune_terminal_to_limit(self, queue_id: str, keep: int) -> int:
        """Prune terminal items (completed/failed/canceled) to keep at most N most-recent items."""
        return self._queries.session_queue.prune(queue_id, user_id=None, keep=keep)

    async def enqueue_batch(
        self, queue_id: str, batch: Batch, prepend: bool, user_id: str = "system"
    ) -> EnqueueBatchResult:
        # The route awaits this method, but every operation below is synchronous database/CPU work. Keep the
        # complete enqueue in one worker operation, off the event loop.
        return await asyncio.to_thread(self._enqueue_batch, queue_id, batch, prepend, user_id)

    def acknowledge_enqueue(self, queue_id: str, idempotency_key: str, user_id: str = "system") -> None:
        self._queries.session_queue.acknowledge_receipt(queue_id, user_id, idempotency_key)

    def get_enqueue_receipt(
        self, queue_id: str, idempotency_key: str, user_id: str = "system"
    ) -> EnqueueBatchReceipt | None:
        receipt = self._queries.session_queue.enqueue_receipt(queue_id, user_id, idempotency_key)
        return _decode_enqueue_receipt(*receipt) if receipt is not None else None

    def _enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool, user_id: str) -> EnqueueBatchResult:
        requested_count = calc_session_count(batch=batch)
        payload_hash = (
            hashlib.sha256(
                json.dumps(
                    {
                        "batch": batch.model_dump(mode="json", exclude={"batch_id", "idempotency_key"}),
                        "prepend": prepend,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if batch.idempotency_key is not None
            else None
        )
        key = batch.idempotency_key

        def settle(receipt: SettledReceipt) -> EnqueueBatchResult:
            if receipt.payload_hash != payload_hash:
                raise EnqueueIdempotencyConflictError(f"Idempotency key {key} is already used by another submission")
            result = _decode_enqueue_receipt(receipt.batch_id, receipt.requested, receipt.enqueued, receipt.item_ids)
            return EnqueueBatchResult(
                queue_id=queue_id,
                requested=result.requested,
                enqueued=result.enqueued,
                batch=batch.model_copy(update={"batch_id": result.batch_id}),
                priority=receipt.priority,
                item_ids=result.item_ids,
            )

        def require_project(q: Queries) -> None:
            if batch.project_id is not None and q.projects.summary(user_id, batch.project_id) is None:
                raise EnqueueProjectNotFoundError(batch.project_id)

        def survey(q: Queries) -> SettledReceipt | int:
            """A settled receipt, or the pending items' count to size the values to prepare."""
            if key is not None and (receipt := q.session_queue.settled_receipt(queue_id, user_id, key)) is not None:
                return receipt
            require_project(q)
            return q.session_queue.pending_count(queue_id)

        surveyed = self._queries.run(survey, read_only=True)
        if isinstance(surveyed, SettledReceipt):
            return settle(surveyed)

        # Preparing the values is CPU work, done outside a transaction; admission below takes what still fits.
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        prepared_values = prepare_values_to_insert(
            queue_id=queue_id,
            batch=batch,
            priority=0,
            max_new_queue_items=max(0, max_queue_size - surveyed),
            user_id=user_id,
        )

        def admit(q: Queries) -> SettledReceipt | EnqueueBatchResult:
            q.locks.acquire(*_ADMISSION, also_shared=_PROTECTING_MEDIA)
            if key is not None:
                q.session_queue.delete_acknowledged_receipts(user_id, ACKNOWLEDGED_ENQUEUE_RECEIPT_RETENTION)
                if (receipt := q.session_queue.settled_receipt(queue_id, user_id, key)) is not None:
                    return receipt
            require_project(q)

            max_new_queue_items = max(0, max_queue_size - q.session_queue.pending_count(queue_id))
            priority = q.session_queue.top_pending_priority(queue_id) + 1 if prepend else 0
            values_to_insert = prepared_values[:max_new_queue_items]
            if priority != 0:
                values_to_insert = [(*value[:5], priority, *value[6:]) for value in values_to_insert]
            enqueued_count = len(values_to_insert)
            accepted_batch = batch
            if enqueued_count > 0:
                accepted_batch_id = batch.batch_id
                while q.session_queue.batch_taken(queue_id, user_id, accepted_batch_id):
                    accepted_batch_id = uuid_string()
                if accepted_batch_id != batch.batch_id:
                    accepted_batch = batch.model_copy(update={"batch_id": accepted_batch_id})
                    values_to_insert = [(*value[:3], accepted_batch_id, *value[4:]) for value in values_to_insert]
            usage = None
            if key is not None and enqueued_count > 0:
                usage = q.session_queue.receipt_usage(user_id)
                if usage.count >= MAX_ENQUEUE_RECEIPTS_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the count limit")
                if usage.unacknowledged_count >= MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER:
                    raise EnqueueReceiptLimitError("Too many unacknowledged enqueue requests")
                if usage.bytes >= MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the storage limit")
                if usage.unacknowledged_bytes >= MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Unacknowledged enqueue receipts exceed the storage limit")
            q.session_queue.insert_items(values_to_insert)
            item_ids = q.session_queue.batch_item_ids(queue_id, user_id, accepted_batch.batch_id)
            if usage is not None:
                receipt_byte_size = len(
                    json.dumps(
                        [
                            queue_id,
                            user_id,
                            key,
                            payload_hash,
                            accepted_batch.batch_id,
                            requested_count,
                            enqueued_count,
                            priority,
                            item_ids,
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                if usage.unacknowledged_bytes + receipt_byte_size > MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Unacknowledged enqueue receipts exceed the storage limit")
                if usage.bytes + receipt_byte_size > MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the storage limit")
                q.session_queue.insert_receipt(
                    {
                        "queue_id": queue_id,
                        "user_id": user_id,
                        "idempotency_key": key,
                        "payload_hash": payload_hash,
                        "batch_id": accepted_batch.batch_id,
                        "requested": requested_count,
                        "enqueued": enqueued_count,
                        "priority": priority,
                        "item_ids": json.dumps(item_ids, separators=(",", ":")),
                        "byte_size": receipt_byte_size,
                    }
                )
            return EnqueueBatchResult(
                queue_id=queue_id,
                requested=requested_count,
                enqueued=enqueued_count,
                batch=accepted_batch,
                priority=priority,
                item_ids=item_ids,
            )

        admitted = self._queries.run(admit)
        if isinstance(admitted, SettledReceipt):
            return settle(admitted)
        self.__invoker.services.events.emit_batch_enqueued(admitted, user_id=user_id)
        return admitted

    def dequeue(self, device: Optional[str] = None) -> Optional[SessionQueueItem]:
        config = self.__invoker.services.configuration
        use_round_robin = config.multiuser and config.session_queue_mode == "round_robin"

        # Snapshot the claiming device's warm models first: the lookup touches the ModelCache lock, which other
        # threads may hold across long operations (VRAM transfers, cache clears). A slightly stale snapshot is fine
        # for a heuristic. An explicitly configured session_queue_mode=FIFO is a request for strict insertion order,
        # so it opts out of affinity reordering entirely (the setting defaults to round_robin, which keeps affinity
        # active for single-user installs even though they use the FIFO order).
        if config.session_queue_mode == "round_robin":
            resident_model_keys = self._get_device_resident_model_keys(device)
        else:
            resident_model_keys = set()

        # Several workers (multi-GPU) may pick the same candidate: the claim moves it to in_progress only if it is
        # still pending, so exactly one of them gets it and the others pick again.
        while True:
            raw_queue_item = self._queries.session_queue.next_to_run(round_robin=use_round_robin)
            if raw_queue_item is None:
                return None
            queue_item, readable = self._hydrate_queue_item(raw_queue_item, quarantine=True)
            if not readable:
                continue
            queue_item = self._apply_device_affinity(queue_item, resident_model_keys)
            claimed = self._queries.session_queue.claim(queue_item.item_id, device)
            if claimed is None:
                continue
            # Patch the item already read instead of re-reading (and re-parsing the session graph of) its row. The
            # claim's columns include the device recorded, so the UI can label the item by GPU.
            for field_name, value in claimed.items():
                setattr(queue_item, field_name, value)
            self._emit_status_changed(queue_item)
            return queue_item

    @staticmethod
    def _make_unreadable_queue_item(raw_queue_item: dict[str, Any], error: Exception) -> SessionQueueItem:
        """Build a metadata-preserving placeholder for an unreadable runtime snapshot."""
        placeholder = SessionQueueItem.model_construct(**raw_queue_item)
        # A placeholder has no trustworthy execution result. Never expose a corrupt or newer snapshot as complete.
        placeholder.status = "failed"
        placeholder.session = GraphExecutionState(graph=Graph())
        placeholder.workflow = None
        placeholder.field_values = None
        placeholder._snapshot_readable = False
        message = f"Unable to load execution state: {error}"
        placeholder.error_type = type(error).__name__
        placeholder.error_message = message
        placeholder.error_traceback = message
        return placeholder

    def _hydrate_queue_item(self, raw_queue_item: dict[str, Any], *, quarantine: bool) -> tuple[SessionQueueItem, bool]:
        """Hydrate one queue row without letting an unreadable snapshot break queue access."""
        try:
            return SessionQueueItem.queue_item_from_dict(raw_queue_item), True
        except (TypeError, ValueError) as exc:
            if quarantine:
                return self._quarantine_unreadable_queue_item(raw_queue_item, exc), False
            return self._make_unreadable_queue_item(raw_queue_item, exc), False

    def _project_queue_item_for_read(self, raw_queue_item: dict[str, Any]) -> SessionQueueItem:
        """Read queue metadata and response results without rebuilding runtime execution state."""
        try:
            return SessionQueueItem.queue_item_from_dict(raw_queue_item, hydrate_runtime=False)
        except (TypeError, ValueError) as exc:
            return self._make_unreadable_queue_item(raw_queue_item, exc)

    def _get_queue_item_for_api(self, item_id: int) -> SessionQueueItem:
        """Read queue metadata and response results without rebuilding execution runtime state."""
        raw_queue_item = self._queries.session_queue.item(item_id)
        if raw_queue_item is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return self._project_queue_item_for_read(raw_queue_item)

    def _quarantine_unreadable_queue_item(self, raw_queue_item: dict[str, Any], error: Exception) -> SessionQueueItem:
        """Fail a pending row whose runtime snapshot is newer than this worker can read.

        The real session cannot be hydrated, so use a minimal in-memory placeholder only for the
        status transition/event. The persisted session remains untouched for postmortem recovery.
        """

        placeholder = self._make_unreadable_queue_item(raw_queue_item, error)
        placeholder.status = "pending"
        return self._set_queue_item_status(
            item_id=placeholder.item_id,
            status="failed",
            error_type=placeholder.error_type,
            error_message=placeholder.error_message,
            error_traceback=placeholder.error_traceback,
            queue_item=placeholder,
        )

    def _apply_device_affinity(self, candidate: SessionQueueItem, resident_keys: set[str]) -> SessionQueueItem:
        """Swap the fairness-chosen candidate for a nearby same-user, same-priority pending item
        whose models are already cached on the claiming device, if one exists.

        Cross-device model reloads are expensive (tens of seconds for large models), so when a user
        has queued a mix of models, preferring an item whose models are warm on the freeing GPU cuts
        thrash. Fairness is preserved by construction: round-robin decides *which user* is served and
        priority ordering decides *which tier* of their items is eligible; this heuristic only
        reorders within that user's equal-priority pending items, and only within
        AFFINITY_MAX_LOOKAHEAD of the candidate's item_id, so a cold item's deferral is bounded.
        The caller passes an empty key set to disable affinity (legacy single-device mode, explicit
        FIFO mode, or cache introspection unavailable).
        """
        if not resident_keys:
            return candidate
        # Model keys are UUID strings that appear verbatim in the session JSON, so residency can be
        # scored with substring matches — no need to parse each candidate's session. Sort for
        # deterministic parameter binding.
        warm = self._queries.session_queue.warmest(
            user_id=candidate.user_id,
            priority=candidate.priority,
            after_item_id=candidate.item_id,
            lookahead=AFFINITY_MAX_LOOKAHEAD,
            resident_keys=sorted(resident_keys),
        )
        if warm is None or warm["item_id"] == candidate.item_id:
            # No warm-model item for this user (or the candidate already is one) — keep the
            # fairness-chosen candidate.
            return candidate
        queue_item, readable = self._hydrate_queue_item(warm, quarantine=True)
        return queue_item if readable else candidate

    def _get_device_resident_model_keys(self, device: Optional[str]) -> set[str]:
        """Best-effort lookup of the model keys currently cached for the given generation device."""
        if device is None:
            return set()
        try:
            cache = self.__invoker.services.model_manager.load.ram_caches.get(device)
            if cache is None:
                return set()
            return set(cache.cached_model_keys())
        except Exception:
            # Affinity is purely an optimization — dequeue must never fail because cache
            # introspection did (e.g. model manager not fully started, or mocked in tests).
            return set()

    def get_next(self, queue_id: str, origin_prefix: Optional[str] = None) -> Optional[SessionQueueItem]:
        raw_queue_item = self._queries.session_queue.item_by_status(queue_id, "pending", origin_prefix)
        return self._hydrate_queue_item(raw_queue_item, quarantine=False)[0] if raw_queue_item is not None else None

    def get_current(self, queue_id: str, origin_prefix: Optional[str] = None) -> Optional[SessionQueueItem]:
        raw_queue_item = self._queries.session_queue.item_by_status(queue_id, "in_progress", origin_prefix)
        return self._hydrate_queue_item(raw_queue_item, quarantine=False)[0] if raw_queue_item is not None else None

    def _get_queue_item_by_status_for_api(
        self, queue_id: str, status: Literal["pending", "in_progress"], origin_prefix: Optional[str]
    ) -> Optional[SessionQueueItem]:
        raw_queue_item = self._queries.session_queue.item_by_status(queue_id, status, origin_prefix)
        return self._project_queue_item_for_read(raw_queue_item) if raw_queue_item is not None else None

    def get_current_for_api(self, queue_id: str, origin_prefix: Optional[str] = None) -> Optional[SessionQueueItem]:
        return self._get_queue_item_by_status_for_api(queue_id, "in_progress", origin_prefix)

    def get_next_for_api(self, queue_id: str, origin_prefix: Optional[str] = None) -> Optional[SessionQueueItem]:
        return self._get_queue_item_by_status_for_api(queue_id, "pending", origin_prefix)

    def _set_queue_item_status(
        self,
        item_id: int,
        status: QUEUE_ITEM_STATUS,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        device: Optional[str] = None,
        queue_item: Optional[SessionQueueItem] = None,
    ) -> SessionQueueItem:
        return self._transition_queue_item_status(
            item_id=item_id,
            status=status,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            device=device,
            queue_item=queue_item,
        )[0]

    def _transition_queue_item_status(
        self,
        item_id: int,
        status: QUEUE_ITEM_STATUS,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        device: Optional[str] = None,
        queue_item: Optional[SessionQueueItem] = None,
    ) -> tuple[SessionQueueItem, bool]:
        """Move a queue item to `status` unless it is already finished (completed, failed or
        canceled), returning the item and whether THIS call performed the transition.

        The guard is the UPDATE's own condition, so two callers racing to cancel the same row cannot
        both observe themselves as the one that canceled it — exactly one sees `transitioned=True`
        (the bulk-cancel counters rely on this), on every backend. When no transition happens, the
        item is returned unchanged and no status-changed event is emitted; a vanished row raises
        SessionQueueItemNotFoundError.
        """
        if queue_item is not None and queue_item.item_id != item_id:
            raise ValueError(f"Queue item {queue_item.item_id} does not match requested item {item_id}")

        exists, changed = self._queries.session_queue.transition(
            item_id,
            status,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            device=device,
        )
        if not exists:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        if changed is None:
            # Already finished: return it unchanged (get_queue_item raises if it was deleted meanwhile).
            return self.get_queue_item(item_id), False

        if queue_item is None:
            queue_item = self.get_queue_item(item_id)
        else:
            # `device` is among the changed columns, so a caller that supplied its own queue_item still
            # sees the device this transition recorded rather than a stale value.
            for field_name, value in changed.items():
                setattr(queue_item, field_name, value)
        self._emit_status_changed(queue_item)
        return queue_item, True

    def _emit_status_changed(self, queue_item: SessionQueueItem) -> None:
        batch_status = self.get_batch_status(queue_id=queue_item.queue_id, batch_id=queue_item.batch_id)
        # The QueueItemStatusChangedEvent ships to user:{queue_item.user_id} and admin rooms.
        # acting_user_id ensures the embedded current-item identifiers are redacted when the
        # in-progress item belongs to someone else, while leaving aggregate counts global.
        # Doing this inside get_queue_status guarantees the redaction decision and the
        # embedded identifiers come from the same lightweight metadata snapshot, eliminating the
        # race where a second read could find None and skip scrubbing stale identifiers.
        # user_id additionally embeds the owner's per-user counts so the owner's client can
        # apply the event's queue_status optimistically without waiting for a refetch; the
        # sanitized companion nulls them before reaching anyone else.
        queue_status = self.get_queue_status(
            queue_id=queue_item.queue_id, user_id=queue_item.user_id, acting_user_id=queue_item.user_id
        )
        self.__invoker.services.events.emit_queue_item_status_changed(queue_item, batch_status, queue_status)

    def _get_workflow_call_descendant_ids(self, item_id: int) -> list[int]:
        return self._queries.session_queue.descendant_ids(item_id)

    def _get_workflow_call_chain_item_ids(self, item_id: int) -> list[int]:
        chain_item_ids = self._queries.session_queue.chain_item_ids(item_id)
        if chain_item_ids is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}, or one of its ancestors")
        return chain_item_ids

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        return IsEmptyResult(is_empty=self._queries.session_queue.count(queue_id) == 0)

    def is_full(self, queue_id: str) -> IsFullResult:
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        return IsFullResult(is_full=self._queries.session_queue.count(queue_id) >= max_queue_size)

    def _cancel_in_progress(self, scope: Scope) -> list[int]:
        """Cancel every in-progress item of the scope, emitting a cancel event for each.

        The bulk cancels leave in-progress items out of their single UPDATE, because a running item
        must be canceled via `_transition_queue_item_status()` so that its `QueueItemStatusChangedEvent`
        is emitted — the session processor responds to that event by setting the cancel event of the
        worker running that exact item_id. With multiple workers (multi-GPU) more than one item can be
        in_progress at once, so each is canceled individually.

        Returns the item ids of the in-progress items actually canceled.
        """
        canceled: list[int] = []
        for item_id in self._queries.session_queue.in_progress(scope):
            # Count only the items THIS call actually moved to 'canceled'. An item that finished
            # meanwhile — including one canceled by a concurrent bulk request that selected the same
            # row — is a no-op transition and must not be counted again. The transition raises if the
            # row vanished entirely (a concurrent clear/delete); such an item needs no cancellation, so
            # skip it rather than failing the whole bulk operation.
            try:
                _, transitioned = self._transition_queue_item_status(item_id, "canceled")
                if transitioned:
                    canceled.append(item_id)
            except SessionQueueItemNotFoundError:
                continue
        return canceled

    def _emit_queue_items_canceled(self, queue_id: str, item_ids_by_user: dict[str, list[int]]) -> None:
        """Emits queue_items_canceled for a bulk cancel/delete, unless nothing was affected —
        an empty result must not broadcast a pointless refetch signal to every client.

        Bulk cancel/delete operations change many rows in one statement and therefore emit no
        per-item queue_item_status_changed events; this lets every affected owner (and everyone's
        badge counts) refresh."""
        if item_ids_by_user:
            self.__invoker.services.events.emit_queue_items_canceled(queue_id, item_ids_by_user)

    def clear(self, queue_id: str, user_id: Optional[str] = None) -> ClearResult:
        # Cancel every in-progress item in scope BEFORE deleting rows, so each running worker is
        # signaled to stop via its item's own status-changed event. With multiple workers (multi-GPU)
        # more than one item can be in_progress at once, and a user-scoped clear must cancel all of
        # that user's running items — and ONLY that user's: other users' rows are out of scope and
        # their workers must keep running. See delete_by_destination for the same pattern.
        scope = Scope(queue_id, user_id=user_id)
        self._cancel_in_progress(scope)
        deleted = self._queries.session_queue.delete_scope(scope)
        self.__invoker.services.events.emit_queue_cleared(queue_id, user_id)
        return ClearResult(deleted=sum(len(item_ids) for item_ids in deleted.values()))

    def delete_queue_items_by_id(self, item_ids: list[int]) -> None:
        self._queries.session_queue.delete_items(item_ids)

    def prune(self, queue_id: str, user_id: Optional[str] = None) -> PruneResult:
        return PruneResult(deleted=self._queries.session_queue.prune(queue_id, user_id=user_id))

    def cancel_queue_item(self, item_id: int) -> SessionQueueItem:
        chain_item_ids = self._get_workflow_call_chain_item_ids(item_id)
        canceled_item: SessionQueueItem | None = None
        for chain_item_id in chain_item_ids:
            queue_item = self._set_queue_item_status(item_id=chain_item_id, status="canceled")
            if chain_item_id == item_id:
                canceled_item = queue_item
        assert canceled_item is not None
        return canceled_item

    def delete_queue_item(self, item_id: int) -> None:
        """Deletes a session queue item"""
        chain_item_ids = self._get_workflow_call_chain_item_ids(item_id)
        if any(self.get_queue_item(chain_item_id).status not in _FINISHED for chain_item_id in chain_item_ids):
            self.cancel_queue_item(item_id)
        self.delete_queue_items_by_id(chain_item_ids)

    def complete_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        return self._set_queue_item_status(item_id=item_id, status="completed", queue_item=queue_item)

    def suspend_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        return self._set_queue_item_status(item_id=item_id, status="waiting", queue_item=queue_item)

    def resume_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        return self._set_queue_item_status(item_id=item_id, status="pending", queue_item=queue_item)

    def fail_queue_item(
        self,
        item_id: int,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> SessionQueueItem:
        return self._set_queue_item_status(
            item_id=item_id,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
        )

    def cancel_by_batch_ids(
        self, queue_id: str, batch_ids: list[str], user_id: Optional[str] = None
    ) -> CancelByBatchIDsResult:
        scope = Scope(queue_id, user_id=user_id, batch_ids=batch_ids)
        canceled_item_ids_by_user = self._queries.session_queue.cancel_waiting_work(scope)
        count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
        # Cancel every in-progress item of the same scope (multi-GPU: possibly several at once). Each
        # cancel emits its own per-item queue_item_status_changed, so the bulk event below need not
        # include them.
        count += len(self._cancel_in_progress(scope))
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByBatchIDsResult(canceled=count)

    def cancel_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> CancelByDestinationResult:
        scope = Scope(queue_id, user_id=user_id, destination=destination)
        canceled_item_ids_by_user = self._queries.session_queue.cancel_waiting_work(scope)
        count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
        count += len(self._cancel_in_progress(scope))
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByDestinationResult(canceled=count)

    def delete_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> DeleteByDestinationResult:
        # Cancel every in-progress item first so each running worker is signaled to stop before we
        # delete its row. With multiple workers (multi-GPU) more than one item can be in_progress;
        # canceling only one would leave the others running (and then failing to update a deleted row).
        scope = Scope(queue_id, user_id=user_id, destination=destination)
        canceled_in_progress_ids = set(self._cancel_in_progress(scope))
        deleted_item_ids_by_user = self._queries.session_queue.delete_scope(scope)
        count = sum(len(item_ids) for item_ids in deleted_item_ids_by_user.values())
        # The in-progress items canceled above each emitted their own per-item
        # queue_item_status_changed, so the bulk event below must not signal them a second time.
        # They are left out of the event, but not of the deletion or the returned count.
        for owner_user_id, item_ids in list(deleted_item_ids_by_user.items()):
            remaining = [item_id for item_id in item_ids if item_id not in canceled_in_progress_ids]
            if remaining:
                deleted_item_ids_by_user[owner_user_id] = remaining
            else:
                del deleted_item_ids_by_user[owner_user_id]
        self._emit_queue_items_canceled(queue_id, deleted_item_ids_by_user)
        return DeleteByDestinationResult(deleted=count)

    def delete_all_except_current(self, queue_id: str, user_id: Optional[str] = None) -> DeleteAllExceptCurrentResult:
        # The chains of the items running now are read in the transaction that deletes, so none can
        # start running in between and lose its children.
        scope = Scope(queue_id, user_id=user_id, except_current=True)
        deleted_item_ids_by_user = self._queries.session_queue.delete_scope(scope, waiting_work_only=True)
        self._emit_queue_items_canceled(queue_id, deleted_item_ids_by_user)
        return DeleteAllExceptCurrentResult(
            deleted=sum(len(item_ids) for item_ids in deleted_item_ids_by_user.values())
        )

    def cancel_by_queue_id(
        self, queue_id: str, user_id: Optional[str] = None, origin_prefix: Optional[str] = None
    ) -> CancelByQueueIDResult:
        scope = Scope(queue_id, user_id=user_id, origin_prefix=origin_prefix)
        canceled_item_ids_by_user = self._queries.session_queue.cancel_waiting_work(scope)
        count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
        count += len(self._cancel_in_progress(scope))
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByQueueIDResult(canceled=count)

    def cancel_all_except_current(
        self, queue_id: str, user_id: Optional[str] = None, origin_prefix: Optional[str] = None
    ) -> CancelAllExceptCurrentResult:
        scope = Scope(queue_id, user_id=user_id, origin_prefix=origin_prefix, except_current=True)
        canceled_item_ids_by_user = self._queries.session_queue.cancel_waiting_work(scope)
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelAllExceptCurrentResult(
            canceled=sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
        )

    def _get_queue_item_with_load_status(self, item_id: int) -> tuple[SessionQueueItem, bool]:
        raw_queue_item = self._queries.session_queue.item(item_id)
        if raw_queue_item is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return self._hydrate_queue_item(raw_queue_item, quarantine=False)

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        return self._get_queue_item_with_load_status(item_id)[0]

    def get_queue_item_for_api(self, item_id: int) -> SessionQueueItem:
        return self._get_queue_item_for_api(item_id)

    def get_queue_item_workflow_json(self, item_id: int) -> str | None:
        return self._queries.session_queue.workflow(item_id)

    def _save_session(self, item_id: int, session: GraphExecutionState, *, only_unfinished: bool) -> bool:
        session_json = _session_json(session)

        def save(q: Queries) -> Optional[bool]:
            # A session names the media its item uses, which this makes active.
            q.locks.acquire(*_PROTECTING_MEDIA, shared=True)
            return q.session_queue.set_session(item_id, session_json, only_unfinished=only_unfinished)

        saved = self._queries.run(save)
        if saved is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return saved

    def save_queue_item_session(self, item_id: int, session: GraphExecutionState) -> None:
        self._save_session(item_id, session, only_unfinished=False)

    def _save_queue_item_session_if_active(self, item_id: int, session: GraphExecutionState) -> bool:
        """Persist a session only while its queue item is non-terminal; whether it was persisted."""
        return self._save_session(item_id, session, only_unfinished=True)

    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        self.save_queue_item_session(item_id, session)
        return self.get_queue_item(item_id)

    def record_workflow_call_child_completion(
        self, parent_item_id: int, child_item_id: int, output_values: dict[str, Any]
    ) -> WorkflowCallChildCompletion | None:
        def record(q: Queries) -> WorkflowCallChildCompletion | None:
            # The parent's row stays locked from the read to the write, so two children completing at
            # once each record into the session the other wrote.
            q.locks.acquire(*_PROTECTING_MEDIA, shared=True)
            row = q.session_queue.lock_item(parent_item_id)
            if row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_item_id}")
            parent_queue_item, readable = self._hydrate_queue_item(row, quarantine=False)
            if not readable:
                raise ValueError("Unable to record workflow call child completion for an unreadable parent session.")
            if parent_queue_item.status in _FINISHED:
                return None

            execution = parent_queue_item.session.waiting_workflow_call_execution
            if execution is not None and child_item_id in execution.completed_child_item_ids:
                return None
            generic_update = parent_queue_item.session.record_generic_child_completion(child_item_id, output_values)
            if generic_update is not None and not generic_update.changed:
                return None
            legacy_should_resume, legacy_values = (
                parent_queue_item.session.record_waiting_workflow_call_child_completion(child_item_id, output_values)
            )
            if generic_update is None:
                should_resume_parent, aggregated_values = legacy_should_resume, legacy_values
            else:
                should_resume_parent = generic_update.status == "completed"
                aggregated_values = {
                    key: values[0] if len(values) == 1 else values
                    for key, values in generic_update.aggregated_outputs.items()
                }
                if generic_update.status == "completed" and aggregated_values != legacy_values:
                    raise ValueError("Generic child aggregation disagrees with workflow-call aggregation.")

            q.session_queue.set_session(parent_item_id, _session_json(parent_queue_item.session), only_unfinished=False)
            return WorkflowCallChildCompletion(
                parent_queue_item=parent_queue_item,
                should_resume=should_resume_parent,
                aggregated_values=aggregated_values,
            )

        return self._queries.run(record)

    def _child_values(
        self,
        parent_queue_item: SessionQueueItem,
        child_session: GraphExecutionState,
        field_values: list[NodeFieldValue] | None,
    ) -> dict[str, Any]:
        workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
        assert workflow_call_execution is not None
        return {
            "queue_id": parent_queue_item.queue_id,
            "session": _session_json(child_session),
            "session_id": child_session.id,
            "batch_id": parent_queue_item.batch_id,
            "field_values": json.dumps(field_values, default=to_jsonable_python) if field_values is not None else None,
            "priority": parent_queue_item.priority,
            "workflow": None,
            "origin": parent_queue_item.origin,
            "destination": parent_queue_item.destination,
            "retried_from_item_id": None,
            "user_id": parent_queue_item.user_id,
            "project_id": parent_queue_item.project_id,
            "workflow_call_id": workflow_call_execution.id,
            "parent_item_id": parent_queue_item.item_id,
            "parent_session_id": parent_queue_item.session_id,
            "root_item_id": parent_queue_item.root_item_id or parent_queue_item.item_id,
            "workflow_call_depth": workflow_call_execution.depth,
            "status": "pending",
        }

    def enqueue_workflow_call_children(
        self,
        parent_queue_item: SessionQueueItem,
        child_sessions: list[tuple[GraphExecutionState, list[NodeFieldValue] | None]],
    ) -> list[SessionQueueItem]:
        if parent_queue_item.session.waiting_workflow_call_execution is None:
            raise ValueError("Parent queue item is missing active workflow call execution metadata.")
        if not child_sessions:
            raise ValueError("Workflow call must enqueue at least one child execution.")
        child_values = [
            self._child_values(parent_queue_item, child_session, field_values)
            for child_session, field_values in child_sessions
        ]
        max_queue_size = self.__invoker.services.configuration.max_queue_size

        def enqueue(q: Queries) -> list[int]:
            q.locks.acquire(*_ADMISSION, also_shared=_PROTECTING_MEDIA)
            parent_status = q.session_queue.lock_status(parent_queue_item.item_id)
            if parent_status is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_queue_item.item_id}")
            if parent_status in _FINISHED:
                raise ValueError("Cannot enqueue workflow call children for a terminal parent queue item.")
            if q.session_queue.pending_count(parent_queue_item.queue_id) + len(child_values) > max_queue_size:
                raise TooManySessionsError(
                    "call_saved_workflow exceeds remaining queue capacity for child workflow executions"
                )
            child_item_ids = [q.session_queue.insert_item(values) for values in child_values]

            # A copy: a unit run again after a lost race inserts children with other ids, which the parent's session
            # must not already name.
            waiting_session = parent_queue_item.session.model_copy(deep=True)
            waiting_session.set_waiting_workflow_call_child_item_ids(child_item_ids)
            waiting_session_json = _session_json(waiting_session)
            if not q.session_queue.wait_for_children(
                parent_queue_item.item_id,
                session=waiting_session_json,
                expected_session=getattr(parent_queue_item, "_session_json", None) or waiting_session_json,
            ):
                raise SessionQueueItemChangedError("Parent queue item changed while enqueuing workflow call children")
            return child_item_ids

        child_item_ids = self._queries.run(enqueue)
        parent_queue_item.session.set_waiting_workflow_call_child_item_ids(child_item_ids)
        parent_queue_item.status = "waiting"
        child_queue_items = [self.get_queue_item(item_id) for item_id in child_item_ids]
        for queue_item in [self.get_queue_item(parent_queue_item.item_id), *child_queue_items]:
            self._emit_status_changed(queue_item)
        return child_queue_items

    def enqueue_workflow_call_child(
        self,
        parent_queue_item: SessionQueueItem,
        child_session: GraphExecutionState,
        field_values: list[NodeFieldValue] | None = None,
    ) -> SessionQueueItem:
        if parent_queue_item.session.waiting_workflow_call_execution is None:
            raise ValueError("Parent queue item is missing active workflow call execution metadata.")
        values = self._child_values(parent_queue_item, child_session, field_values)
        max_queue_size = self.__invoker.services.configuration.max_queue_size

        def enqueue(q: Queries) -> int:
            q.locks.acquire(*_ADMISSION, also_shared=_PROTECTING_MEDIA)
            parent_status = q.session_queue.lock_status(parent_queue_item.item_id)
            if parent_status is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_queue_item.item_id}")
            if parent_status in _FINISHED:
                raise ValueError("Cannot enqueue workflow call child for a terminal parent queue item.")
            if q.session_queue.pending_count(parent_queue_item.queue_id) >= max_queue_size:
                raise TooManySessionsError(
                    "call_saved_workflow exceeds remaining queue capacity for child workflow executions"
                )
            return q.session_queue.insert_item(values)

        queue_item = self.get_queue_item(self._queries.run(enqueue))
        self._emit_status_changed(queue_item)
        return queue_item

    def cancel_workflow_call_children(
        self, workflow_call_id: str, exclude_item_ids: set[int] | None = None
    ) -> list[int]:
        exclude_item_ids = exclude_item_ids or set()
        item_ids_with_descendants: list[int] = []
        for item_id in self._queries.session_queue.workflow_call_item_ids(workflow_call_id):
            item_ids_with_descendants.append(item_id)
            item_ids_with_descendants.extend(self._get_workflow_call_descendant_ids(item_id))
        canceled_item_ids: list[int] = []
        for item_id in dict.fromkeys(item_ids_with_descendants):
            if item_id in exclude_item_ids:
                continue
            if self.get_queue_item(item_id).status in _FINISHED:
                continue
            self._set_queue_item_status(item_id=item_id, status="canceled")
            canceled_item_ids.append(item_id)
        return canceled_item_ids

    def list_queue_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
        destination: Optional[str] = None,
    ) -> CursorPaginatedResults[SessionQueueItem]:
        raw_queue_items = self._queries.session_queue.page(
            queue_id,
            limit=limit + 1,
            after=(priority, cursor) if cursor is not None else None,
            status=status,
            destination=destination,
        )
        items = [self._hydrate_queue_item(raw_queue_item, quarantine=False)[0] for raw_queue_item in raw_queue_items]
        has_more = len(items) > limit
        return CursorPaginatedResults(items=items[:limit], limit=limit, has_more=has_more)

    def list_all_queue_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        """Gets all queue items with fully rehydrated runtime sessions."""
        raw_queue_items = self._queries.session_queue.all_items(queue_id, destination)
        return [self._hydrate_queue_item(raw_queue_item, quarantine=False)[0] for raw_queue_item in raw_queue_items]

    def list_all_queue_items_for_api(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        """Gets response-shaped queue items without rebuilding runtime execution state."""
        raw_queue_items = self._queries.session_queue.all_items(queue_id, destination)
        return [self._project_queue_item_for_read(raw_queue_item) for raw_queue_item in raw_queue_items]

    def get_queue_item_ids(
        self,
        queue_id: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        user_id: Optional[str] = None,
        origin_prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> ItemIdsResult:
        item_ids = self._queries.session_queue.item_ids(
            queue_id,
            descending=order_dir == SQLiteDirection.Descending,
            user_id=user_id,
            origin_prefix=origin_prefix,
            limit=limit,
        )
        return ItemIdsResult(item_ids=item_ids, total_count=len(item_ids))

    def has_active_queue_work(self) -> bool:
        return self._queries.session_queue.has_active_work()

    def get_queue_item_summaries_by_ids(self, queue_id: str, item_ids: list[int]) -> list[SessionQueueItemSummary]:
        if not item_ids:
            return []
        summaries_by_id = {
            raw_summary["item_id"]: SessionQueueItemSummary.queue_item_summary_from_dict(raw_summary)
            for raw_summary in self._queries.session_queue.summaries(queue_id, item_ids)
        }
        return [summaries_by_id[item_id] for item_id in item_ids if item_id in summaries_by_id]

    def get_queue_status(
        self,
        queue_id: str,
        user_id: Optional[str] = None,
        acting_user_id: Optional[str] = None,
        origin_prefix: Optional[str] = None,
        is_admin: bool = False,
    ) -> SessionQueueStatus:
        # Aggregate counts are global across all users within the requested scope. With user_id, the
        # same snapshot also yields that user's own counts, for the per-user portion of the badge;
        # they are returned in separate fields and never replace the global counts. Only the current
        # item's identifiers are read, not a full SessionQueueItem: this runs on every status poll and
        # every status change, and hydrating the item would deserialize its whole session graph.
        counts, own_counts, current_item = self._queries.session_queue.status_counts(queue_id, user_id, origin_prefix)

        user_pending: Optional[int] = None
        user_in_progress: Optional[int] = None
        if user_id is not None:
            user_pending = own_counts.get("pending", 0)
            user_in_progress = own_counts.get("in_progress", 0)

        # Redaction is decided from the same current_item snapshot used to embed identifiers,
        # so a concurrent transition (e.g. B finishing while A's status changes) cannot leave
        # stale identifiers in the result. The aggregate counts stay global; only the current
        # item's identifiers are gated. acting_user_id (event path) takes precedence over
        # user_id (API path) when deciding the redaction owner; either being None means a
        # global caller who may see the current item. is_admin disables redaction outright so
        # admin callers can pass their user_id (for the per-user counts) without losing
        # visibility of other users' current item.
        owner_user_id = user_id if acting_user_id is None else acting_user_id
        current_item_id = None
        current_session_id = None
        current_batch_id = None
        if current_item is not None and (is_admin or owner_user_id is None or current_item.user_id == owner_user_id):
            current_item_id = current_item.item_id
            current_session_id = current_item.session_id
            current_batch_id = current_item.batch_id

        return SessionQueueStatus(
            queue_id=queue_id,
            item_id=current_item_id,
            session_id=current_session_id,
            batch_id=current_batch_id,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            waiting=counts.get("waiting", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=sum(counts.values()),
            user_pending=user_pending,
            user_in_progress=user_in_progress,
        )

    def get_batch_status(self, queue_id: str, batch_id: str, user_id: Optional[str] = None) -> BatchStatus:
        counts, origin, destination = self._queries.session_queue.batch_counts(queue_id, batch_id, user_id)
        return BatchStatus(
            batch_id=batch_id,
            origin=origin,
            destination=destination,
            queue_id=queue_id,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            waiting=counts.get("waiting", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=sum(counts.values()),
        )

    def get_counts_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> SessionQueueCountsByDestination:
        counts = self._queries.session_queue.destination_counts(queue_id, destination, user_id)
        return SessionQueueCountsByDestination(
            queue_id=queue_id,
            destination=destination,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            waiting=counts.get("waiting", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=sum(counts.values()),
        )

    def retry_items_by_id(self, queue_id: str, item_ids: list[int]) -> RetryItemsResult:
        """Retries the given queue items"""
        max_queue_size = self.__invoker.services.configuration.max_queue_size

        def survey(q: Queries) -> list[tuple[int, str, ValueToInsertTuple]]:
            """The failed or canceled roots to clone, each once, with its owner and the clone's values. Reading and
            validating the graphs is the costly part, done without holding the queue's admission."""
            clones: list[tuple[int, str, ValueToInsertTuple]] = []
            seen_root_item_ids: set[int] = set()
            hydrated_queue_items: dict[int, SessionQueueItem] = {}

            def read(item_id: int) -> Optional[SessionQueueItem]:
                """The item with its runtime state, which the retry validates its graph with."""
                if item_id not in hydrated_queue_items:
                    raw_queue_item = q.session_queue.item(item_id)
                    if raw_queue_item is None:
                        return None
                    hydrated_queue_items[item_id] = self._hydrate_queue_item(raw_queue_item, quarantine=False)[0]
                return hydrated_queue_items[item_id]

            for item_id in item_ids:
                queue_item = read(item_id)
                if queue_item is None or queue_item.queue_id != queue_id:
                    continue
                if queue_item.status not in ("failed", "canceled"):
                    continue

                root_item_id = queue_item.root_item_id or queue_item.item_id
                if root_item_id in seen_root_item_ids:
                    continue
                seen_root_item_ids.add(root_item_id)

                root_queue_item = read(root_item_id)
                if root_queue_item is None:
                    raise SessionQueueItemNotFoundError(f"No queue item with id {root_item_id}")
                if not root_queue_item._snapshot_readable:
                    continue
                if root_queue_item.status not in ("failed", "canceled"):
                    continue

                field_values_json = (
                    json.dumps(root_queue_item.field_values, default=to_jsonable_python)
                    if root_queue_item.field_values
                    else None
                )
                workflow_json = (
                    json.dumps(root_queue_item.workflow, default=to_jsonable_python)
                    if root_queue_item.workflow
                    else None
                )
                # Validate the graph before dumping a fresh empty execution state. The full read above already
                # rehydrated runtime state for contract-compatible validation and recovery semantics.
                root_graph = Graph.model_validate(
                    root_queue_item.session.graph.model_dump(mode="python", warnings=False), strict=False
                )
                cloned_session = GraphExecutionState(graph=root_graph)
                retried_from_item_id = (
                    root_queue_item.retried_from_item_id
                    if root_queue_item.retried_from_item_id is not None
                    else root_queue_item.item_id
                )
                clones.append(
                    (
                        root_item_id,
                        root_queue_item.user_id,
                        (
                            root_queue_item.queue_id,
                            _session_json(cloned_session),
                            cloned_session.id,
                            root_queue_item.batch_id,
                            field_values_json,
                            root_queue_item.priority,
                            workflow_json,
                            root_queue_item.origin,
                            root_queue_item.destination,
                            retried_from_item_id,
                            root_queue_item.user_id,
                            root_queue_item.project_id,
                        ),
                    )
                )
            return clones

        clones = self._queries.run(survey, read_only=True)

        def admit(q: Queries) -> Optional[list[tuple[int, str, ValueToInsertTuple]]]:
            """The clones the queue had room for, inserted; None when it is full."""
            q.locks.acquire(*_ADMISSION, also_shared=_PROTECTING_MEDIA)
            max_new_queue_items = max_queue_size - q.session_queue.pending_count(queue_id)
            if max_new_queue_items <= 0:
                return None
            admitted = clones[:max_new_queue_items]
            q.session_queue.insert_items([values for _, _, values in admitted])
            return admitted

        admitted = self._queries.run(admit)
        if admitted is None:
            return RetryItemsResult(queue_id=queue_id, retried_item_ids=[])
        retried_item_ids_by_user: dict[str, list[int]] = {}
        for root_item_id, user_id, _ in admitted:
            retried_item_ids_by_user.setdefault(user_id, []).append(root_item_id)
        retry_result = RetryItemsResult(
            queue_id=queue_id, retried_item_ids=[root_item_id for root_item_id, _, _ in admitted]
        )
        self.__invoker.services.events.emit_queue_items_retried(
            retry_result,
            user_ids=list(dict.fromkeys(user_id for _, user_id, _ in admitted)),
            retried_item_ids_by_user=retried_item_ids_by_user,
        )
        return retry_result
