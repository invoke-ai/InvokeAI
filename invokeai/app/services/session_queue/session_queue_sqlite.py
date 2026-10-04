import asyncio
import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, cast

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
from invokeai.app.services.shared.execution_state_migration import (
    dump_execution_state,
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.pagination import CursorPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

# Maximum number of ids bound into a single `IN (...)` clause. SQLite's compile-time bind limit is
# 999 on builds older than 3.32 and 32766 on newer ones; staying under the lower figure (leaving
# room for the other bind params in the statement) keeps the queries portable across both.
SQLITE_MAX_BIND_PARAMS_PER_CHUNK = 900
MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER = 10_000
MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER = 64 * 1024 * 1024
MAX_ENQUEUE_RECEIPTS_PER_OWNER = 100_000
MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER = 128 * 1024 * 1024
ACKNOWLEDGED_ENQUEUE_RECEIPT_RETENTION_DAYS = 7

# A completed child is still recovery state for its active root, including the interval
# before its returned media is persisted in its parent. Both history-pruning paths must
# retain it; intermediates cleanup also uses these rows to protect the root's media.
PRUNABLE_QUEUE_ITEMS_SQL = """
    status IN ('completed', 'failed', 'canceled')
    AND (root_item_id IS NULL OR root_item_id NOT IN (
        SELECT item_id FROM session_queue WHERE status IN ('pending', 'in_progress', 'waiting')
    ))
"""

# Round-robin dequeue (multiuser fairness): pick the next pending item from the user who was
# least-recently served.
#
# The "whose turn is it" ordering key is each candidate user's most recent started_at. We compute
# it with a *correlated* MAX subquery rather than a GROUP BY over all started rows: the candidate
# set is one row per user with pending work, and each MAX(started_at) WHERE user_id = ? is
# satisfied by an indexed seek (idx_session_queue_user_started_at) instead of scanning the full
# retained queue history. This keeps dequeue cost proportional to the number of active users, not
# to total history (which is unbounded by default). MAX() ignores NULL started_at values, so users
# with only pending items fall back to the epoch via COALESCE and are served first.
#
# Kept as a module constant so the scaling test can EXPLAIN QUERY PLAN the exact production SQL.
ROUND_ROBIN_DEQUEUE_QUERY = """--sql
    WITH user_next_item AS (
        -- For each user, select their single best pending item (highest priority, then oldest).
        SELECT
            user_id,
            item_id,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY priority DESC, item_id ASC
            ) AS rn
        FROM session_queue
        WHERE status = 'pending'
    )
    SELECT
        sq.*,
        u.display_name AS user_display_name,
        u.email AS user_email
    FROM session_queue sq
    LEFT JOIN users u ON sq.user_id = u.user_id
    JOIN user_next_item uni ON sq.item_id = uni.item_id AND uni.rn = 1
    ORDER BY
        COALESCE(
            (
                SELECT MAX(served.started_at)
                FROM session_queue served
                WHERE served.user_id = sq.user_id
            ),
            '1970-01-01'
        ) ASC,
        sq.item_id ASC
    LIMIT 1
    """

# Upper bound on how many resident model keys the device-affinity scoring query binds. Real
# per-device caches hold at most a handful of models; this only guards the SQL parameter count.
MAX_AFFINITY_MODEL_KEYS = 50

# How far past the fairness-chosen candidate (in item_id distance) the device-affinity swap may
# look for a warm-model item. This bounds two things at once: the scoring query's cost (at most
# this many session blobs are scanned per dequeue) and how long a cold item can be deferred — the
# starved candidate's item_id never changes, so newly enqueued warm items eventually fall outside
# the window and the cold item runs after at most ~this many swaps.
AFFINITY_MAX_LOOKAHEAD = 32

# FIFO dequeue (single-user mode, or round_robin explicitly disabled): strict priority then
# insertion order.
FIFO_DEQUEUE_QUERY = """--sql
    SELECT
        sq.*,
        u.display_name as user_display_name,
        u.email as user_email
    FROM session_queue sq
    LEFT JOIN users u ON sq.user_id = u.user_id
    WHERE sq.status = 'pending'
    ORDER BY
        sq.priority DESC,
        sq.item_id ASC
    LIMIT 1
    """


class SqliteSessionQueue(SessionQueueBase):
    __invoker: Invoker

    # Serializes the select-candidate-then-claim sequence in `dequeue()`. The DB connection's
    # RLock serializes individual statements, but the gap between selecting the next pending item
    # and marking it 'in_progress' is a race: with multiple session-processor workers (multi-GPU),
    # two workers could select the same item. Holding this lock across the whole claim prevents it.
    _dequeue_lock = threading.Lock()

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

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def _set_in_progress_to_canceled(self) -> None:
        """
        Sets all in_progress or waiting queue items to canceled. Run on app startup, not associated with any queue.
        This is necessary because the invoker may have been killed while processing a queue item or while a parent
        queue item was suspended waiting on a child workflow execution.
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT item_id
                FROM session_queue
                WHERE status = 'in_progress'
                   OR status = 'waiting';
                """
            )
            interrupted_item_ids = [row[0] for row in cast(list[sqlite3.Row], cursor.fetchall())]
        item_ids_to_cancel: set[int] = set()
        for item_id in interrupted_item_ids:
            item_ids_to_cancel.update(self._get_workflow_call_chain_item_ids(item_id))
        if not item_ids_to_cancel:
            return
        with self._db.transaction() as cursor:
            placeholders = ",".join("?" for _ in item_ids_to_cancel)
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled',
                    status_sequence = COALESCE(status_sequence, 0) + 1
                WHERE item_id IN ({placeholders})
                  AND (
                    status = 'pending'
                    OR status = 'in_progress'
                    OR status = 'waiting'
                  );
                """,
                tuple(item_ids_to_cancel),
            )

    def _prune_terminal_to_limit(self, queue_id: str, keep: int) -> int:
        """Prune terminal items (completed/failed/canceled) to keep at most N most-recent items."""
        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE
                queue_id = ?
                AND {PRUNABLE_QUEUE_ITEMS_SQL}
                """
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where}
                AND item_id NOT IN (
                    SELECT item_id
                    FROM session_queue
                    {where}
                    ORDER BY COALESCE(completed_at, updated_at, created_at) DESC, item_id DESC
                    LIMIT ?
                );
                """,
                (queue_id, queue_id, keep),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where}
                AND item_id NOT IN (
                    SELECT item_id
                    FROM session_queue
                    {where}
                    ORDER BY COALESCE(completed_at, updated_at, created_at) DESC, item_id DESC
                    LIMIT ?
                );
                """,
                (queue_id, queue_id, keep),
            )
        return count

    def _get_current_queue_size(self, queue_id: str) -> int:
        """Gets the current number of pending queue items"""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE
                queue_id = ?
                AND status = 'pending'
                """,
                (queue_id,),
            )
            count = cast(int, cursor.fetchone()[0])
        return count

    def _get_highest_priority(self, queue_id: str) -> int:
        """Gets the highest priority value in the queue"""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT MAX(priority)
                FROM session_queue
                WHERE
                queue_id = ?
                AND status = 'pending'
                """,
                (queue_id,),
            )
            priority = cast(Union[int, None], cursor.fetchone()[0]) or 0
        return priority

    async def enqueue_batch(
        self, queue_id: str, batch: Batch, prepend: bool, user_id: str = "system"
    ) -> EnqueueBatchResult:
        # The route awaits this method, but every operation below is synchronous SQLite/CPU
        # work. Keep the complete transaction in one worker operation; otherwise the queue-size
        # check before the first await and the insert/event work after later awaits execute on the
        # event loop.
        return await asyncio.to_thread(self._enqueue_batch, queue_id, batch, prepend, user_id)

    def acknowledge_enqueue(self, queue_id: str, idempotency_key: str, user_id: str = "system") -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE session_queue_enqueue_receipts
                SET acknowledged_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE queue_id = ? AND user_id = ? AND idempotency_key = ?
                """,
                (queue_id, user_id, idempotency_key),
            )

    def get_enqueue_receipt(
        self, queue_id: str, idempotency_key: str, user_id: str = "system"
    ) -> EnqueueBatchReceipt | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT batch_id, requested, enqueued, item_ids
                FROM session_queue_enqueue_receipts
                WHERE queue_id = ? AND user_id = ? AND idempotency_key = ?
                """,
                (queue_id, user_id, idempotency_key),
            )
            receipt = cursor.fetchone()
        return self._decode_enqueue_receipt(receipt) if receipt is not None else None

    @staticmethod
    def _decode_enqueue_receipt(receipt: sqlite3.Row) -> EnqueueBatchReceipt:
        item_ids = json.loads(receipt["item_ids"])
        if not isinstance(item_ids, list) or not all(isinstance(item_id, int) for item_id in item_ids):
            raise RuntimeError("Stored queue enqueue receipt contains invalid item ids")
        return EnqueueBatchReceipt(
            batch_id=receipt["batch_id"],
            requested=receipt["requested"],
            enqueued=receipt["enqueued"],
            item_ids=item_ids,
        )

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

        def get_receipt(cursor: sqlite3.Cursor) -> sqlite3.Row | None:
            if batch.idempotency_key is None:
                return None
            cursor.execute(
                """--sql
                SELECT payload_hash, batch_id, requested, enqueued, priority, item_ids
                FROM session_queue_enqueue_receipts
                WHERE queue_id = ? AND user_id = ? AND idempotency_key = ?
                """,
                (queue_id, user_id, batch.idempotency_key),
            )
            return cast(sqlite3.Row | None, cursor.fetchone())

        def settle_receipt(receipt: sqlite3.Row) -> EnqueueBatchResult:
            if receipt["payload_hash"] != payload_hash:
                raise EnqueueIdempotencyConflictError(
                    f"Idempotency key {batch.idempotency_key} is already used by another submission"
                )
            result = self._decode_enqueue_receipt(receipt)
            return EnqueueBatchResult(
                queue_id=queue_id,
                requested=result.requested,
                enqueued=result.enqueued,
                batch=batch.model_copy(update={"batch_id": result.batch_id}),
                priority=receipt["priority"],
                item_ids=result.item_ids,
            )

        def require_project(cursor: sqlite3.Cursor) -> None:
            if batch.project_id is None:
                return
            cursor.execute(
                "SELECT 1 FROM projects WHERE user_id = ? AND project_id = ?;",
                (user_id, batch.project_id),
            )
            if cursor.fetchone() is None:
                raise EnqueueProjectNotFoundError(batch.project_id)

        with self._db.transaction() as cursor:
            if batch.idempotency_key is not None:
                receipt = get_receipt(cursor)
                if receipt is not None:
                    return settle_receipt(receipt)
            require_project(cursor)
            cursor.execute(
                "SELECT count(*) FROM session_queue WHERE queue_id = ? AND status = 'pending';",
                (queue_id,),
            )
            preliminary_queue_size = cast(int, cursor.fetchone()[0])

        max_queue_size = self.__invoker.services.configuration.max_queue_size
        preliminary_capacity = max(0, max_queue_size - preliminary_queue_size)
        prepared_values = prepare_values_to_insert(
            queue_id=queue_id,
            batch=batch,
            priority=0,
            max_new_queue_items=preliminary_capacity,
            user_id=user_id,
        )

        with self._db.transaction() as cursor:
            if batch.idempotency_key is not None:
                cursor.execute(
                    """--sql
                    DELETE FROM session_queue_enqueue_receipts
                    WHERE user_id = ?
                      AND acknowledged_at IS NOT NULL
                      AND acknowledged_at <= STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', ?)
                    """,
                    (user_id, f"-{ACKNOWLEDGED_ENQUEUE_RECEIPT_RETENTION_DAYS} days"),
                )
                receipt = get_receipt(cursor)
                if receipt is not None:
                    return settle_receipt(receipt)

            require_project(cursor)

            cursor.execute(
                """--sql
                SELECT count(*) FROM session_queue WHERE queue_id = ? AND status = 'pending';
                """,
                (queue_id,),
            )
            current_queue_size = cast(int, cursor.fetchone()[0])
            max_new_queue_items = max(0, max_queue_size - current_queue_size)
            priority = 0
            if prepend:
                cursor.execute(
                    "SELECT MAX(priority) FROM session_queue WHERE queue_id = ? AND status = 'pending';",
                    (queue_id,),
                )
                priority = (cast(int | None, cursor.fetchone()[0]) or 0) + 1
            values_to_insert = prepared_values[:max_new_queue_items]
            if priority != 0:
                values_to_insert = [(*value[:5], priority, *value[6:]) for value in values_to_insert]
            enqueued_count = len(values_to_insert)
            accepted_batch = batch
            if enqueued_count > 0:
                accepted_batch_id = batch.batch_id
                while True:
                    cursor.execute(
                        """--sql
                        SELECT 1 FROM session_queue
                        WHERE queue_id = ? AND user_id = ? AND batch_id = ?
                        LIMIT 1;
                        """,
                        (queue_id, user_id, accepted_batch_id),
                    )
                    if cursor.fetchone() is None:
                        break
                    accepted_batch_id = uuid_string()
                if accepted_batch_id != batch.batch_id:
                    accepted_batch = batch.model_copy(update={"batch_id": accepted_batch_id})
                    values_to_insert = [(*value[:3], accepted_batch_id, *value[4:]) for value in values_to_insert]
            receipt_bytes = 0
            unacknowledged_bytes = 0
            if batch.idempotency_key is not None and enqueued_count > 0:
                cursor.execute(
                    """--sql
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(byte_size), 0),
                        COALESCE(SUM(CASE WHEN acknowledged_at IS NULL THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN acknowledged_at IS NULL THEN byte_size ELSE 0 END), 0)
                    FROM session_queue_enqueue_receipts
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                receipt_count, receipt_bytes, unacknowledged_count, unacknowledged_bytes = cursor.fetchone()
                if cast(int, receipt_count) >= MAX_ENQUEUE_RECEIPTS_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the count limit")
                if cast(int, unacknowledged_count) >= MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER:
                    raise EnqueueReceiptLimitError("Too many unacknowledged enqueue requests")
                if cast(int, receipt_bytes) >= MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the storage limit")
                if cast(int, unacknowledged_bytes) >= MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Unacknowledged enqueue receipts exceed the storage limit")
            cursor.executemany(
                """--sql
                INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )
            cursor.execute(
                """--sql
                SELECT item_id
                FROM session_queue
                WHERE queue_id = ? AND user_id = ? AND batch_id = ?
                ORDER BY item_id ASC;
                """,
                (queue_id, user_id, accepted_batch.batch_id),
            )
            item_ids = [row[0] for row in cursor.fetchall()]
            if batch.idempotency_key is not None and enqueued_count > 0:
                item_ids_json = json.dumps(item_ids, separators=(",", ":"))
                receipt_byte_size = len(
                    json.dumps(
                        [
                            queue_id,
                            user_id,
                            batch.idempotency_key,
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
                if (
                    cast(int, unacknowledged_bytes) + receipt_byte_size
                    > MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER
                ):
                    raise EnqueueReceiptLimitError("Unacknowledged enqueue receipts exceed the storage limit")
                if cast(int, receipt_bytes) + receipt_byte_size > MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER:
                    raise EnqueueReceiptLimitError("Enqueue receipts exceed the storage limit")
                cursor.execute(
                    """--sql
                    INSERT INTO session_queue_enqueue_receipts (
                        queue_id, user_id, idempotency_key, payload_hash, batch_id,
                        requested, enqueued, priority, item_ids, byte_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        queue_id,
                        user_id,
                        batch.idempotency_key,
                        payload_hash,
                        accepted_batch.batch_id,
                        requested_count,
                        enqueued_count,
                        priority,
                        item_ids_json,
                        receipt_byte_size,
                    ),
                )
        enqueue_result = EnqueueBatchResult(
            queue_id=queue_id,
            requested=requested_count,
            enqueued=enqueued_count,
            batch=accepted_batch,
            priority=priority,
            item_ids=item_ids,
        )
        self.__invoker.services.events.emit_batch_enqueued(enqueue_result, user_id=user_id)
        return enqueue_result

    def dequeue(self, device: Optional[str] = None) -> Optional[SessionQueueItem]:
        config = self.__invoker.services.configuration
        use_round_robin = config.multiuser and config.session_queue_mode == "round_robin"

        query = ROUND_ROBIN_DEQUEUE_QUERY if use_round_robin else FIFO_DEQUEUE_QUERY

        # Snapshot the claiming device's warm models BEFORE taking the dequeue lock: the lookup
        # touches the ModelCache lock, which other threads may hold across long operations (VRAM
        # transfers, cache clears). A slightly stale snapshot is fine for a heuristic; stalling
        # every worker's dequeue is not. An explicitly configured session_queue_mode=FIFO is a
        # request for strict insertion order, so it opts out of affinity reordering entirely
        # (the setting defaults to round_robin, which keeps affinity active for single-user
        # installs even though they use the FIFO query).
        if config.session_queue_mode == "round_robin":
            resident_model_keys = self._get_device_resident_model_keys(device)
        else:
            resident_model_keys = set()

        # Hold the dequeue lock across the select-then-claim so concurrent workers (multi-GPU)
        # cannot select and claim the same pending item. `_set_queue_item_status` already no-ops
        # if the item was concurrently moved to a terminal state (e.g. canceled), so we only need
        # to guard against two dequeues racing for the same pending row.
        with self._dequeue_lock:
            while True:
                with self._db.transaction() as cursor:
                    cursor.execute(query)
                    result = cast(Union[sqlite3.Row, None], cursor.fetchone())
                if result is None:
                    return None
                raw_result = dict(result)
                queue_item, readable = self._hydrate_queue_item(raw_result, quarantine=True)
                if not readable:
                    continue
                queue_item = self._apply_device_affinity(queue_item, resident_model_keys)
                # Record the claiming worker's device so the UI can label the item by GPU. Passing the
                # item we already materialized lets _set_queue_item_status patch it in place instead of
                # re-reading (and re-parsing the session graph of) the row we just read.
                queue_item = self._set_queue_item_status(
                    item_id=queue_item.item_id, status="in_progress", device=device, queue_item=queue_item
                )
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

    def _get_queue_item_for_read(
        self,
        item_id: int,
        *,
        cursor: sqlite3.Cursor | None = None,
        hydrate_runtime: bool = False,
    ) -> SessionQueueItem:
        """Read queue metadata and response results without rebuilding execution runtime state."""
        if cursor is None:
            with self._db.transaction() as transaction_cursor:
                return self._get_queue_item_for_read(
                    item_id, cursor=transaction_cursor, hydrate_runtime=hydrate_runtime
                )

        cursor.execute(
            """--sql
            SELECT
                sq.*,
                u.display_name AS user_display_name,
                u.email AS user_email
            FROM session_queue sq
            LEFT JOIN users u ON sq.user_id = u.user_id
            WHERE sq.item_id = ?
            """,
            (item_id,),
        )
        result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        raw_queue_item = dict(result)
        if hydrate_runtime:
            return self._hydrate_queue_item(raw_queue_item, quarantine=False)[0]
        return self._project_queue_item_for_read(raw_queue_item)

    def _get_queue_item_for_api(self, item_id: int, *, cursor: sqlite3.Cursor | None = None) -> SessionQueueItem:
        """Read one queue row with either full runtime hydration or response projection."""
        return self._get_queue_item_for_read(item_id, cursor=cursor, hydrate_runtime=False)

    def _get_queue_item_for_retry(self, item_id: int, *, cursor: sqlite3.Cursor | None = None) -> SessionQueueItem:
        """Read the graph and retry metadata with full runtime hydration."""
        return self._get_queue_item_for_read(item_id, cursor=cursor, hydrate_runtime=True)

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
        # deterministic parameter binding; cap to bound the query if a cache is unexpectedly large.
        keys = sorted(resident_keys)[:MAX_AFFINITY_MODEL_KEYS]
        score = " + ".join(["(instr(sq.session, ?) > 0)"] * len(keys))
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT
                    sq.*,
                    u.display_name AS user_display_name,
                    u.email AS user_email,
                    ({score}) AS affinity
                FROM session_queue sq
                LEFT JOIN users u ON sq.user_id = u.user_id
                WHERE sq.status = 'pending'
                    AND sq.user_id IS ?
                    AND sq.priority = ?
                    AND sq.item_id >= ?
                    AND sq.item_id <= ?
                ORDER BY affinity DESC, sq.item_id ASC
                LIMIT 1
                """,
                (
                    *keys,
                    candidate.user_id,
                    candidate.priority,
                    candidate.item_id,
                    candidate.item_id + AFFINITY_MAX_LOOKAHEAD,
                ),
            )
            row = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if row is None:
            return candidate
        row_dict = dict(row)
        if not row_dict.pop("affinity", 0) or row_dict["item_id"] == candidate.item_id:
            # No warm-model item for this user (or the candidate already is one) — keep the
            # fairness-chosen candidate.
            return candidate
        queue_item, readable = self._hydrate_queue_item(row_dict, quarantine=True)
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
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT
                    sq.*,
                    u.display_name as user_display_name,
                    u.email as user_email
                FROM session_queue sq
                LEFT JOIN users u ON sq.user_id = u.user_id
                WHERE
                    sq.queue_id = ?
                    AND sq.status = 'pending'
                """
            params = [queue_id]
            if origin_prefix is not None:
                query += """--sql
                    AND sq.origin LIKE ?
                    """
                params.append(f"{origin_prefix}%")
            query += """--sql
                ORDER BY
                    sq.priority DESC,
                    sq.created_at ASC
                LIMIT 1
                """
            cursor.execute(query, params)
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        return self._hydrate_queue_item(dict(result), quarantine=False)[0]

    def get_current(self, queue_id: str, origin_prefix: Optional[str] = None) -> Optional[SessionQueueItem]:
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT
                    sq.*,
                    u.display_name as user_display_name,
                    u.email as user_email
                FROM session_queue sq
                LEFT JOIN users u ON sq.user_id = u.user_id
                WHERE
                    sq.queue_id = ?
                    AND sq.status = 'in_progress'
                """
            params = [queue_id]
            if origin_prefix is not None:
                query += """--sql
                    AND sq.origin LIKE ?
                    """
                params.append(f"{origin_prefix}%")
            query += """--sql
                LIMIT 1
                """
            cursor.execute(query, params)
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        return self._hydrate_queue_item(dict(result), quarantine=False)[0]

    def _get_queue_item_by_status_for_api(
        self, queue_id: str, status: Literal["pending", "in_progress"], origin_prefix: Optional[str]
    ) -> Optional[SessionQueueItem]:
        query = """--sql
            SELECT
                sq.*,
                u.display_name as user_display_name,
                u.email as user_email
            FROM session_queue sq
            LEFT JOIN users u ON sq.user_id = u.user_id
            WHERE
                sq.queue_id = ?
                AND sq.status = ?
            """
        params: list[str] = [queue_id, status]
        if origin_prefix is not None:
            query += " AND sq.origin LIKE ?"
            params.append(f"{origin_prefix}%")
        if status == "pending":
            query += " ORDER BY sq.priority DESC, sq.created_at ASC"
        query += " LIMIT 1"
        with self._db.transaction() as cursor:
            cursor.execute(query, params)
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        return self._project_queue_item_for_read(dict(result))

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

        The terminal guard and the UPDATE run in one transaction, so two callers racing to cancel
        the same row cannot both observe themselves as the one that canceled it — exactly one sees
        `transitioned=True` (the bulk-cancel counters rely on this). Note the atomicity comes from
        SqliteDatabase.transaction() holding its process-wide lock for the whole block, not from
        the SQL: the guard is a Python check between the SELECT and the UPDATE, valid only while
        every writer goes through that lock. When no transition happens, the item is returned
        unchanged and no status-changed event is emitted; a vanished row raises
        SessionQueueItemNotFoundError.
        """
        if queue_item is not None and queue_item.item_id != item_id:
            raise ValueError(f"Queue item {queue_item.item_id} does not match requested item {item_id}")

        updated_status_row: sqlite3.Row | None = None
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status FROM session_queue WHERE item_id = ?
                """,
                (item_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
            current_status = row[0]

            # Only update if not already finished (completed, failed or canceled).
            if current_status not in ("completed", "failed", "canceled"):
                cursor.execute(
                    """--sql
                    UPDATE session_queue
                    SET status = ?, status_sequence = COALESCE(status_sequence, 0) + 1, error_type = ?, error_message = ?, error_traceback = ?, device = COALESCE(?, device)
                    WHERE item_id = ?
                    """,
                    (status, error_type, error_message, error_traceback, device, item_id),
                )
                cursor.execute(
                    """--sql
                    SELECT
                        status,
                        status_sequence,
                        error_type,
                        error_message,
                        error_traceback,
                        created_at,
                        updated_at,
                        started_at,
                        completed_at,
                        device
                    FROM session_queue
                    WHERE item_id = ?
                    """,
                    (item_id,),
                )
                updated_status_row = cast(sqlite3.Row, cursor.fetchone())

        if updated_status_row is None:
            # Already finished (return it unchanged) or deleted (get_queue_item raises).
            return self.get_queue_item(item_id), False

        if queue_item is None:
            queue_item = self.get_queue_item(item_id)
        else:
            # `device` is in the SELECT above, so a caller that supplied its own queue_item still
            # sees the device this claim just recorded rather than a stale value.
            for field_name in updated_status_row.keys():
                setattr(queue_item, field_name, updated_status_row[field_name])
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
        return queue_item, True

    def _get_workflow_call_child_ids(self, item_id: int) -> list[int]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT item_id
                FROM session_queue
                WHERE parent_item_id = ?
                ORDER BY item_id ASC
                """,
                (item_id,),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
        return [row[0] for row in rows]

    def _get_workflow_call_descendant_ids(self, item_id: int) -> list[int]:
        descendant_ids: list[int] = []
        queue: list[int] = [item_id]
        while queue:
            current_item_id = queue.pop(0)
            child_ids = self._get_workflow_call_child_ids(current_item_id)
            descendant_ids.extend(child_ids)
            queue.extend(child_ids)
        return descendant_ids

    def _get_workflow_call_ancestor_ids(self, item_id: int) -> list[int]:
        ancestor_ids: list[int] = []
        current_item_id = item_id
        while True:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    SELECT parent_item_id
                    FROM session_queue
                    WHERE item_id = ?
                    """,
                    (current_item_id,),
                )
                row = cursor.fetchone()
            if row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {current_item_id}")
            parent_item_id = row[0]
            if parent_item_id is None:
                break
            ancestor_ids.append(parent_item_id)
            current_item_id = parent_item_id
        return ancestor_ids

    def _get_workflow_call_chain_item_ids(self, item_id: int) -> list[int]:
        ancestor_ids = self._get_workflow_call_ancestor_ids(item_id)
        root_item_id = ancestor_ids[-1] if ancestor_ids else item_id
        descendant_ids = self._get_workflow_call_descendant_ids(root_item_id)
        chain_item_ids = ancestor_ids + [item_id] + descendant_ids
        deduped_chain_item_ids = list(dict.fromkeys(chain_item_ids))
        return deduped_chain_item_ids

    def _get_current_workflow_call_chain_item_ids(self, queue_id: str) -> set[int]:
        """Item ids in the workflow-call chains of every currently executing item.

        With multiple GPU workers, several items can be in_progress at once. Every one of them is
        "current": the "except current" operations must leave each active worker's complete
        ancestor/descendant chain intact, not just the chain of one arbitrarily selected item.
        Waiting chains with no in-progress item remain cancellable backlog, exactly as in
        single-worker mode.
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT item_id
                FROM session_queue
                WHERE queue_id = ? AND status = 'in_progress'
                ORDER BY item_id ASC
                """,
                (queue_id,),
            )
            in_progress_ids = [cast(int, row[0]) for row in cursor.fetchall()]

        if in_progress_ids:
            chain_item_ids: set[int] = set()
            for item_id in in_progress_ids:
                chain_item_ids.update(self._get_workflow_call_chain_item_ids(item_id))
            return chain_item_ids

        # Nothing is in progress. A chain can still be mid-execution in the gap between one child
        # finishing and the next being dequeued (parent 'waiting', next child 'pending'); protect
        # the chain of the next item that would be dequeued, as before.
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT child.item_id
                FROM session_queue child
                JOIN session_queue parent ON parent.item_id = child.parent_item_id
                WHERE
                    child.queue_id = ?
                    AND child.status = 'pending'
                    AND parent.status = 'waiting'
                ORDER BY child.priority DESC, child.created_at ASC, child.item_id ASC
                LIMIT 1
                """,
                (queue_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return set()
        return set(self._get_workflow_call_chain_item_ids(cast(int, row[0])))

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            is_empty = cast(int, cursor.fetchone()[0]) == 0
        return IsEmptyResult(is_empty=is_empty)

    def is_full(self, queue_id: str) -> IsFullResult:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            max_queue_size = self.__invoker.services.configuration.max_queue_size
            is_full = cast(int, cursor.fetchone()[0]) >= max_queue_size
        return IsFullResult(is_full=is_full)

    def clear(self, queue_id: str, user_id: Optional[str] = None) -> ClearResult:
        user_filter = "AND user_id = ?" if user_id is not None else ""
        # Cancel every in-progress item in scope BEFORE deleting rows, so each running
        # worker is signaled to stop via its item's own status-changed event. With
        # multiple workers (multi-GPU) more than one item can be in_progress at once, and
        # a user-scoped clear must cancel all of that user's running items — and ONLY
        # that user's: other users' rows are out of scope and their workers must keep
        # running. See delete_by_destination for the same pattern.
        match_filter = f"queue_id == ? {user_filter}"
        cancel_params: list[Any] = [queue_id]
        if user_id is not None:
            cancel_params.append(user_id)
        self._cancel_in_progress_matching(match_filter, cancel_params)

        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE queue_id = ?
                {user_filter}
                """
            params: list[str] = [queue_id]
            if user_id is not None:
                params.append(user_id)
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where}
                """,
                tuple(params),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where}
                """,
                tuple(params),
            )
        self.__invoker.services.events.emit_queue_cleared(queue_id, user_id)
        return ClearResult(deleted=count)

    def delete_queue_items_by_id(self, item_ids: list[int]) -> None:
        if not item_ids:
            return
        placeholders = ", ".join(["?" for _ in item_ids])
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                WHERE item_id IN ({placeholders})
                """,
                tuple(item_ids),
            )

    def prune(self, queue_id: str, user_id: Optional[str] = None) -> PruneResult:
        with self._db.transaction() as cursor:
            # Build WHERE clause with optional user_id filter
            user_filter = "AND user_id = ?" if user_id is not None else ""
            where = f"""--sql
                WHERE
                queue_id = ?
                AND {PRUNABLE_QUEUE_ITEMS_SQL}
                {user_filter}
                """
            params: list[Any] = [queue_id]
            if user_id is not None:
                params.append(user_id)

            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
        return PruneResult(deleted=count)

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
        if any(
            self.get_queue_item(chain_item_id).status not in {"completed", "failed", "canceled"}
            for chain_item_id in chain_item_ids
        ):
            self.cancel_queue_item(item_id)
        self.delete_queue_items_by_id(chain_item_ids)

    def complete_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(item_id=item_id, status="completed", queue_item=queue_item)
        return queue_item

    def suspend_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(item_id=item_id, status="waiting", queue_item=queue_item)
        return queue_item

    def resume_queue_item(self, item_id: int, queue_item: Optional[SessionQueueItem] = None) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(item_id=item_id, status="pending", queue_item=queue_item)
        return queue_item

    def fail_queue_item(
        self,
        item_id: int,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(
            item_id=item_id,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
        )
        return queue_item

    def _cancel_in_progress_matching(self, match_filter: str, params: list[Any]) -> list[int]:
        """Cancel every in-progress item matching `match_filter`, emitting a cancel event for each.

        The bulk-cancel methods exclude in-progress items from their single UPDATE statement, because
        a running item must be canceled via `_set_queue_item_status()` so that its
        `QueueItemStatusChangedEvent` is emitted — the session processor responds to that event by
        setting the cancel event of the worker running that exact item_id. With multiple workers
        (multi-GPU) more than one item can be in_progress at once, so each matching item is canceled
        individually here rather than relying on a single `get_current()` (which returns only one).

        `match_filter` is a WHERE fragment without the leading WHERE (e.g.
        "queue_id == ? AND batch_id IN (?, ?)"); `params` are its bound values.

        Returns the item ids of the in-progress items actually canceled.
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT item_id
                FROM session_queue
                WHERE status == 'in_progress' AND {match_filter};
                """,
                tuple(params),
            )
            item_ids = [row[0] for row in cursor.fetchall()]

        canceled: list[int] = []
        for item_id in item_ids:
            # Count only the items THIS call actually moved to 'canceled'. An item that finished
            # between the SELECT and now — including one canceled by a concurrent bulk request
            # that selected the same row — is a no-op transition and must not be counted again.
            # The transition raises if the row vanished entirely (a concurrent clear/delete);
            # such an item needs no cancellation, so skip it rather than failing the whole bulk
            # operation.
            try:
                _, transitioned = self._transition_queue_item_status(item_id, "canceled")
                if transitioned:
                    canceled.append(item_id)
            except SessionQueueItemNotFoundError:
                continue
        return canceled

    def _collect_item_ids_by_user(
        self, cursor: sqlite3.Cursor, where: str, params: Sequence[Any]
    ) -> dict[str, list[int]]:
        """Groups the item ids matched by `where` by their owner's user id.

        Bulk cancel/delete operations mutate many rows in a single SQL statement and therefore
        emit no per-item queue_item_status_changed events. Call this with the operation's WHERE
        clause before mutating, then pass the result to _emit_queue_items_canceled after the
        transaction commits so every affected owner (and everyone's badge counts) can refresh.
        """
        cursor.execute(
            f"""--sql
            SELECT item_id, user_id
            FROM session_queue
            {where};
            """,
            tuple(params),
        )
        item_ids_by_user: dict[str, list[int]] = {}
        for item_id, owner_user_id in cursor.fetchall():
            item_ids_by_user.setdefault(owner_user_id, []).append(item_id)
        return item_ids_by_user

    def _emit_queue_items_canceled(self, queue_id: str, item_ids_by_user: dict[str, list[int]]) -> None:
        """Emits queue_items_canceled for a bulk cancel/delete, unless nothing was affected —
        an empty result must not broadcast a pointless refetch signal to every client."""
        if item_ids_by_user:
            self.__invoker.services.events.emit_queue_items_canceled(queue_id, item_ids_by_user)

    def cancel_by_batch_ids(
        self, queue_id: str, batch_ids: list[str], user_id: Optional[str] = None
    ) -> CancelByBatchIDsResult:
        placeholders = ", ".join(["?" for _ in batch_ids])
        # Build the match filter (with optional user_id filter) shared by the bulk update and the
        # in-progress cancellation below.
        user_filter = "AND user_id = ?" if user_id is not None else ""
        match_filter = f"queue_id == ? AND batch_id IN ({placeholders}) {user_filter}"
        params: list[Any] = [queue_id] + batch_ids
        if user_id is not None:
            params.append(user_id)

        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE {match_filter}
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- In-progress items are canceled individually below so each worker is signaled.
                  AND status != 'in_progress'
                """
            canceled_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled',
                    status_sequence = COALESCE(status_sequence, 0) + 1
                {where};
                """,
                tuple(params),
            )

        # Cancel every in-progress item matching the same filter (multi-GPU: possibly several at
        # once). Each cancel emits its own per-item queue_item_status_changed, so the bulk event
        # below need not include them (the WHERE above already excludes in-progress rows).
        count += len(self._cancel_in_progress_matching(match_filter, params))

        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByBatchIDsResult(canceled=count)

    def cancel_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> CancelByDestinationResult:
        user_filter = "AND user_id = ?" if user_id is not None else ""
        match_filter = f"queue_id == ? AND destination == ? {user_filter}"
        params: list[Any] = [queue_id, destination]
        if user_id is not None:
            params.append(user_id)

        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE {match_filter}
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- In-progress items are canceled individually below so each worker is signaled.
                  AND status != 'in_progress'
                """
            canceled_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled',
                    status_sequence = COALESCE(status_sequence, 0) + 1
                {where};
                """,
                tuple(params),
            )

        # Cancel every in-progress item matching the same filter (multi-GPU: possibly several at
        # once). Each cancel emits its own per-item queue_item_status_changed, so the bulk event
        # below need not include them (the WHERE above already excludes in-progress rows).
        count += len(self._cancel_in_progress_matching(match_filter, params))

        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByDestinationResult(canceled=count)

    def delete_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> DeleteByDestinationResult:
        user_filter = "AND user_id = ?" if user_id is not None else ""
        match_filter = f"queue_id == ? AND destination == ? {user_filter}"
        params: list[Any] = [queue_id, destination]
        if user_id is not None:
            params.append(user_id)

        # Cancel every in-progress item first so each running worker is signaled to stop before we
        # delete its row. With multiple workers (multi-GPU) more than one item can be in_progress;
        # canceling only get_current() would leave the others running (and then failing to update a
        # deleted row). See _cancel_in_progress_matching.
        canceled_in_progress_ids = set(self._cancel_in_progress_matching(match_filter, params))

        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE
                  queue_id == ?
                  AND destination == ?
                  {user_filter}
                """
            deleted_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in deleted_item_ids_by_user.values())
            # The in-progress items canceled above each emitted their own per-item
            # queue_item_status_changed, so the bulk event below must not signal them a second
            # time. Their rows still match the destination WHERE (the cancel only flipped their
            # status), so they are excluded from the collected ids — but not from the DELETE or
            # the returned count.
            if canceled_in_progress_ids:
                for owner_user_id in list(deleted_item_ids_by_user):
                    remaining = [
                        i for i in deleted_item_ids_by_user[owner_user_id] if i not in canceled_in_progress_ids
                    ]
                    if remaining:
                        deleted_item_ids_by_user[owner_user_id] = remaining
                    else:
                        del deleted_item_ids_by_user[owner_user_id]
            cursor.execute(
                f"""--sql
                DELETE FROM session_queue
                {where};
                """,
                tuple(params),
            )
        self._emit_queue_items_canceled(queue_id, deleted_item_ids_by_user)
        return DeleteByDestinationResult(deleted=count)

    def delete_all_except_current(self, queue_id: str, user_id: Optional[str] = None) -> DeleteAllExceptCurrentResult:
        current_chain_item_ids = self._get_current_workflow_call_chain_item_ids(queue_id)
        with self._db.transaction() as cursor:
            # Build WHERE clause with optional user_id filter
            user_filter = "AND user_id = ?" if user_id is not None else ""
            current_chain_filter = ""
            if current_chain_item_ids:
                placeholders = ", ".join(["?" for _ in current_chain_item_ids])
                current_chain_filter = f"AND item_id NOT IN ({placeholders})"
            where = f"""--sql
                WHERE
                  queue_id == ?
                  AND status IN ('pending', 'waiting')
                  {user_filter}
                  {current_chain_filter}
                """
            params: list[Any] = [queue_id]
            if user_id is not None:
                params.append(user_id)
            params.extend(current_chain_item_ids)

            deleted_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in deleted_item_ids_by_user.values())
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
        self._emit_queue_items_canceled(queue_id, deleted_item_ids_by_user)
        return DeleteAllExceptCurrentResult(deleted=count)

    def cancel_by_queue_id(
        self, queue_id: str, user_id: Optional[str] = None, origin_prefix: Optional[str] = None
    ) -> CancelByQueueIDResult:
        user_filter = "AND user_id = ?" if user_id is not None else ""
        origin_filter = "AND origin LIKE ?" if origin_prefix is not None else ""
        match_filter = f"queue_id == ? {user_filter} {origin_filter}"
        params: list[Any] = [queue_id]
        if user_id is not None:
            params.append(user_id)
        if origin_prefix is not None:
            params.append(f"{origin_prefix}%")

        with self._db.transaction() as cursor:
            where = f"""--sql
                WHERE {match_filter}
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- In-progress items are canceled individually below so each worker is signaled.
                  AND status != 'in_progress'
                """
            canceled_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled',
                    status_sequence = COALESCE(status_sequence, 0) + 1
                {where};
                """,
                tuple(params),
            )

        # Cancel every in-progress item in the queue (multi-GPU: possibly several at once). Each
        # cancel emits its own per-item queue_item_status_changed; the bulk event below covers the
        # silently-updated rows.
        count += len(self._cancel_in_progress_matching(match_filter, params))
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelByQueueIDResult(canceled=count)

    def cancel_all_except_current(
        self, queue_id: str, user_id: Optional[str] = None, origin_prefix: Optional[str] = None
    ) -> CancelAllExceptCurrentResult:
        current_chain_item_ids = self._get_current_workflow_call_chain_item_ids(queue_id)
        with self._db.transaction() as cursor:
            # Build WHERE clause with optional user_id and origin_prefix filters
            user_filter = "AND user_id = ?" if user_id is not None else ""
            origin_filter = "AND origin LIKE ?" if origin_prefix is not None else ""
            current_chain_filter = ""
            if current_chain_item_ids:
                placeholders = ", ".join(["?" for _ in current_chain_item_ids])
                current_chain_filter = f"AND item_id NOT IN ({placeholders})"
            where = f"""--sql
                WHERE
                  queue_id == ?
                  AND status IN ('pending', 'waiting')
                  {user_filter}
                  {origin_filter}
                  {current_chain_filter}
                """
            params: list[Any] = [queue_id]
            if user_id is not None:
                params.append(user_id)
            if origin_prefix is not None:
                params.append(f"{origin_prefix}%")
            params.extend(current_chain_item_ids)

            canceled_item_ids_by_user = self._collect_item_ids_by_user(cursor, where, params)
            count = sum(len(item_ids) for item_ids in canceled_item_ids_by_user.values())
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled',
                    status_sequence = COALESCE(status_sequence, 0) + 1
                {where};
                """,
                tuple(params),
            )
        self._emit_queue_items_canceled(queue_id, canceled_item_ids_by_user)
        return CancelAllExceptCurrentResult(canceled=count)

    def _get_queue_item_with_load_status(self, item_id: int) -> tuple[SessionQueueItem, bool]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT
                    sq.*,
                    u.display_name as user_display_name,
                    u.email as user_email
                FROM session_queue sq
                LEFT JOIN users u ON sq.user_id = u.user_id
                WHERE sq.item_id = ?
                """,
                (item_id,),
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return self._hydrate_queue_item(dict(result), quarantine=False)

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        return self._get_queue_item_with_load_status(item_id)[0]

    def get_queue_item_for_api(self, item_id: int) -> SessionQueueItem:
        return self._get_queue_item_for_api(item_id)

    def get_queue_item_workflow_json(self, item_id: int) -> str | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT workflow
                FROM session_queue
                WHERE item_id = ?
                """,
                (item_id,),
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        return cast(str | None, result["workflow"]) if result is not None else None

    def save_queue_item_session(self, item_id: int, session: GraphExecutionState) -> None:
        with self._db.transaction() as cursor:
            # Use exclude_none so we don't end up with a bunch of nulls in the graph - this can cause validation errors
            # when the graph is loaded. Persisted sessions are used to resume execution across queue boundaries.
            session_json = json.dumps(dump_execution_state(session), default=to_jsonable_python)
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET session = ?
                WHERE item_id = ?
                """,
                (session_json, item_id),
            )
            if cursor.rowcount == 0:
                raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")

    def _save_queue_item_session_if_active(self, item_id: int, session: GraphExecutionState) -> bool:
        """Persist a session only while its queue item is non-terminal.

        This is an internal race guard. The existing transaction lock makes the status check and
        write atomic for all queue mutations in this process without adding a persistence column.
        """
        session_json = json.dumps(dump_execution_state(session), default=to_jsonable_python)
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET session = ?
                WHERE item_id = ? AND status NOT IN ('completed', 'failed', 'canceled')
                """,
                (session_json, item_id),
            )
            if cursor.rowcount != 0:
                return True
            cursor.execute("SELECT 1 FROM session_queue WHERE item_id = ?", (item_id,))
            if cursor.fetchone() is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
            return False

    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        self.save_queue_item_session(item_id, session)
        return self.get_queue_item(item_id)

    def record_workflow_call_child_completion(
        self, parent_item_id: int, child_item_id: int, output_values: dict[str, Any]
    ) -> WorkflowCallChildCompletion | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE item_id = ?
                """,
                (parent_item_id,),
            )
            row = cast(sqlite3.Row | None, cursor.fetchone())
            if row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_item_id}")
            parent_queue_item, readable = self._hydrate_queue_item(dict(row), quarantine=False)
            if not readable:
                raise ValueError("Unable to record workflow call child completion for an unreadable parent session.")
            if parent_queue_item.status in ("completed", "failed", "canceled"):
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

            session_json = json.dumps(dump_execution_state(parent_queue_item.session), default=to_jsonable_python)
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET session = ?
                WHERE item_id = ?
                """,
                (session_json, parent_item_id),
            )
            if cursor.rowcount == 0:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_item_id}")

        return WorkflowCallChildCompletion(
            parent_queue_item=parent_queue_item,
            should_resume=should_resume_parent,
            aggregated_values=aggregated_values,
        )

    def enqueue_workflow_call_children(
        self,
        parent_queue_item: SessionQueueItem,
        child_sessions: list[tuple[GraphExecutionState, list[NodeFieldValue] | None]],
    ) -> list[SessionQueueItem]:
        workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
        if workflow_call_execution is None:
            raise ValueError("Parent queue item is missing active workflow call execution metadata.")
        if not child_sessions:
            raise ValueError("Workflow call must enqueue at least one child execution.")

        serialized_children = [
            (
                json.dumps(dump_execution_state(child_session), default=to_jsonable_python),
                json.dumps(field_values, default=to_jsonable_python) if field_values is not None else None,
            )
            for child_session, field_values in child_sessions
        ]
        root_item_id = parent_queue_item.root_item_id or parent_queue_item.item_id
        child_item_ids: list[int] = []
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status
                FROM session_queue
                WHERE item_id = ?
                """,
                (parent_queue_item.item_id,),
            )
            parent_status_row = cursor.fetchone()
            if parent_status_row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_queue_item.item_id}")
            if parent_status_row[0] in ("completed", "failed", "canceled"):
                raise ValueError("Cannot enqueue workflow call children for a terminal parent queue item.")

            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE queue_id = ? AND status = 'pending'
                """,
                (parent_queue_item.queue_id,),
            )
            pending_count = cast(int, cursor.fetchone()[0])
            max_queue_size = self.__invoker.services.configuration.max_queue_size
            if pending_count + len(child_sessions) > max_queue_size:
                raise TooManySessionsError(
                    "call_saved_workflow exceeds remaining queue capacity for child workflow executions"
                )

            for (session_json, field_values_json), (child_session, _field_values) in zip(
                serialized_children, child_sessions, strict=True
            ):
                cursor.execute(
                    """--sql
                    INSERT INTO session_queue (
                        queue_id,
                        session,
                        session_id,
                        batch_id,
                        field_values,
                        priority,
                        workflow,
                        origin,
                        destination,
                        retried_from_item_id,
                        user_id,
                        workflow_call_id,
                        parent_item_id,
                        parent_session_id,
                        root_item_id,
                        workflow_call_depth,
                        project_id,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        parent_queue_item.queue_id,
                        session_json,
                        child_session.id,
                        parent_queue_item.batch_id,
                        field_values_json,
                        parent_queue_item.priority,
                        None,
                        parent_queue_item.origin,
                        parent_queue_item.destination,
                        None,
                        parent_queue_item.user_id,
                        workflow_call_execution.id,
                        parent_queue_item.item_id,
                        parent_queue_item.session_id,
                        root_item_id,
                        workflow_call_execution.depth,
                        parent_queue_item.project_id,
                    ),
                )
                child_item_ids.append(cast(int, cursor.lastrowid))

            parent_queue_item.session.set_waiting_workflow_call_child_item_ids(child_item_ids)
            session_json = json.dumps(dump_execution_state(parent_queue_item.session), default=to_jsonable_python)
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET session = ?,
                    status = 'waiting', status_sequence = COALESCE(status_sequence, 0) + 1
                WHERE item_id = ? AND session = ?
                    AND status NOT IN ('completed', 'failed', 'canceled')
                """,
                (
                    session_json,
                    parent_queue_item.item_id,
                    getattr(parent_queue_item, "_session_json", None)
                    or json.dumps(dump_execution_state(parent_queue_item.session), default=to_jsonable_python),
                ),
            )
            if cursor.rowcount == 0:
                raise SessionQueueItemChangedError("Parent queue item changed while enqueuing workflow call children")

        parent_queue_item.status = "waiting"
        child_queue_items = [self.get_queue_item(item_id) for item_id in child_item_ids]
        for queue_item in [self.get_queue_item(parent_queue_item.item_id), *child_queue_items]:
            batch_status = self.get_batch_status(queue_id=queue_item.queue_id, batch_id=queue_item.batch_id)
            queue_status = self.get_queue_status(
                queue_id=queue_item.queue_id, user_id=queue_item.user_id, acting_user_id=queue_item.user_id
            )
            self.__invoker.services.events.emit_queue_item_status_changed(queue_item, batch_status, queue_status)
        return child_queue_items

    def enqueue_workflow_call_child(
        self,
        parent_queue_item: SessionQueueItem,
        child_session: GraphExecutionState,
        field_values: list[NodeFieldValue] | None = None,
    ) -> SessionQueueItem:
        workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
        if workflow_call_execution is None:
            raise ValueError("Parent queue item is missing active workflow call execution metadata.")

        session_json = json.dumps(dump_execution_state(child_session), default=to_jsonable_python)
        field_values_json = json.dumps(field_values, default=to_jsonable_python) if field_values is not None else None
        root_item_id = parent_queue_item.root_item_id or parent_queue_item.item_id

        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status
                FROM session_queue
                WHERE item_id = ?
                """,
                (parent_queue_item.item_id,),
            )
            parent_row = cursor.fetchone()
            if parent_row is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {parent_queue_item.item_id}")
            if parent_row[0] in ("completed", "failed", "canceled"):
                raise ValueError("Cannot enqueue workflow call child for a terminal parent queue item.")

            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE queue_id = ? AND status = 'pending'
                """,
                (parent_queue_item.queue_id,),
            )
            pending_count = cast(int, cursor.fetchone()[0])
            if pending_count >= self.__invoker.services.configuration.max_queue_size:
                raise TooManySessionsError(
                    "call_saved_workflow exceeds remaining queue capacity for child workflow executions"
                )

            cursor.execute(
                """--sql
                INSERT INTO session_queue (
                    queue_id,
                    session,
                    session_id,
                    batch_id,
                    field_values,
                    priority,
                    workflow,
                    origin,
                    destination,
                    retried_from_item_id,
                    user_id,
                    project_id,
                    workflow_call_id,
                    parent_item_id,
                    parent_session_id,
                    root_item_id,
                    workflow_call_depth,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    parent_queue_item.queue_id,
                    session_json,
                    child_session.id,
                    parent_queue_item.batch_id,
                    field_values_json,
                    parent_queue_item.priority,
                    None,
                    parent_queue_item.origin,
                    parent_queue_item.destination,
                    None,
                    parent_queue_item.user_id,
                    parent_queue_item.project_id,
                    workflow_call_execution.id,
                    parent_queue_item.item_id,
                    parent_queue_item.session_id,
                    root_item_id,
                    workflow_call_execution.depth,
                ),
            )
            item_id = cursor.lastrowid

        queue_item = self.get_queue_item(item_id)
        batch_status = self.get_batch_status(queue_id=queue_item.queue_id, batch_id=queue_item.batch_id)
        queue_status = self.get_queue_status(
            queue_id=queue_item.queue_id, user_id=queue_item.user_id, acting_user_id=queue_item.user_id
        )
        self.__invoker.services.events.emit_queue_item_status_changed(queue_item, batch_status, queue_status)
        return queue_item

    def cancel_workflow_call_children(
        self, workflow_call_id: str, exclude_item_ids: set[int] | None = None
    ) -> list[int]:
        exclude_item_ids = exclude_item_ids or set()
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT item_id
                FROM session_queue
                WHERE workflow_call_id = ?
                ORDER BY item_id ASC
                """,
                (workflow_call_id,),
            )
            item_ids = [row[0] for row in cast(list[sqlite3.Row], cursor.fetchall())]
        item_ids_with_descendants: list[int] = []
        for item_id in item_ids:
            item_ids_with_descendants.append(item_id)
            item_ids_with_descendants.extend(self._get_workflow_call_descendant_ids(item_id))
        item_ids = list(dict.fromkeys(item_ids_with_descendants))
        canceled_item_ids: list[int] = []
        for item_id in item_ids:
            if item_id in exclude_item_ids:
                continue
            queue_item = self.get_queue_item(item_id)
            if queue_item.status in {"completed", "failed", "canceled"}:
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
        with self._db.transaction() as cursor_:
            item_id = cursor
            query = """--sql
                SELECT *
                FROM session_queue
                WHERE queue_id = ?
            """
            params: list[Union[str, int]] = [queue_id]

            if status is not None:
                query += """--sql
                    AND status = ?
                    """
                params.append(status)

            if destination is not None:
                query += """---sql
                    AND destination = ?
                """
                params.append(destination)

            if item_id is not None:
                query += """--sql
                    AND (priority < ?) OR (priority = ? AND item_id > ?)
                    """
                params.extend([priority, priority, item_id])

            query += """--sql
                ORDER BY
                    priority DESC,
                    item_id ASC
                LIMIT ?
                """
            params.append(limit + 1)
            cursor_.execute(query, params)
            results = cast(list[sqlite3.Row], cursor_.fetchall())
        items = [self._hydrate_queue_item(dict(result), quarantine=False)[0] for result in results]
        has_more = False
        if len(items) > limit:
            # remove the extra item
            items.pop()
            has_more = True
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def _list_all_queue_item_rows(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT
                    sq.*,
                    u.display_name as user_display_name,
                    u.email as user_email
                FROM session_queue sq
                LEFT JOIN users u ON sq.user_id = u.user_id
                WHERE sq.queue_id = ?
            """
            params: list[Union[str, int]] = [queue_id]

            if destination is not None:
                query += """---sql
                    AND sq.destination = ?
                """
                params.append(destination)

            query += """--sql
                ORDER BY
                    sq.priority DESC,
                    sq.item_id ASC
                ;
                """
            cursor.execute(query, params)
            return cast(list[sqlite3.Row], cursor.fetchall())

    def list_all_queue_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        """Gets all queue items with fully rehydrated runtime sessions."""
        results = self._list_all_queue_item_rows(queue_id=queue_id, destination=destination)
        return [self._hydrate_queue_item(dict(result), quarantine=False)[0] for result in results]

    def list_all_queue_items_for_api(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        """Gets response-shaped queue items without rebuilding runtime execution state."""
        results = self._list_all_queue_item_rows(queue_id=queue_id, destination=destination)
        return [self._project_queue_item_for_read(dict(result)) for result in results]

    def get_queue_item_ids(
        self,
        queue_id: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        user_id: Optional[str] = None,
        origin_prefix: Optional[str] = None,
    ) -> ItemIdsResult:
        with self._db.transaction() as cursor_:
            query = """--sql
                SELECT item_id
                FROM session_queue
                WHERE queue_id = ?
                """
            query_params: list[str] = [queue_id]

            if user_id is not None:
                query += " AND user_id = ?"
                query_params.append(user_id)

            if origin_prefix is not None:
                query += " AND origin LIKE ?"
                query_params.append(f"{origin_prefix}%")

            query += f" ORDER BY created_at {order_dir.value}"

            cursor_.execute(query, query_params)
            result = cast(list[sqlite3.Row], cursor_.fetchall())
        item_ids = [row[0] for row in result]

        return ItemIdsResult(item_ids=item_ids, total_count=len(item_ids))

    def get_queue_item_summaries_by_ids(self, queue_id: str, item_ids: list[int]) -> list[SessionQueueItemSummary]:
        if not item_ids:
            return []

        rows: list[sqlite3.Row] = []
        with self._db.transaction() as cursor:
            # Each id becomes one bind parameter, so a single IN (...) would blow past SQLite's
            # per-statement variable limit for large id lists. Query in chunks instead - callers
            # are bounded at the API layer, but this keeps any caller from hitting that ceiling.
            for chunk_start in range(0, len(item_ids), SQLITE_MAX_BIND_PARAMS_PER_CHUNK):
                chunk = item_ids[chunk_start : chunk_start + SQLITE_MAX_BIND_PARAMS_PER_CHUNK]
                placeholders = ", ".join("?" for _ in chunk)
                cursor.execute(
                    f"""--sql
                    SELECT
                        sq.item_id,
                        sq.created_at,
                        sq.status,
                        sq.device,
                        sq.started_at,
                        sq.completed_at,
                        sq.origin,
                        sq.destination,
                        sq.batch_id,
                        sq.user_id,
                        u.display_name AS user_display_name,
                        u.email AS user_email,
                        sq.field_values,
                        sq.parent_item_id
                    FROM session_queue sq
                    LEFT JOIN users u ON sq.user_id = u.user_id
                    WHERE sq.queue_id = ? AND sq.item_id IN ({placeholders})
                    """,
                    (queue_id, *chunk),
                )
                rows.extend(cast(list[sqlite3.Row], cursor.fetchall()))

        summaries_by_id = {
            row["item_id"]: SessionQueueItemSummary.queue_item_summary_from_dict(dict(row)) for row in rows
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
        with self._db.transaction() as cursor:
            # Aggregate counts are global across all users within the requested scope.
            query = """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ?
                """
            params: list[str] = [queue_id]

            if origin_prefix is not None:
                query += " AND origin LIKE ?"
                params.append(f"{origin_prefix}%")

            query += " GROUP BY status"
            cursor.execute(query, params)
            counts_result = cast(list[sqlite3.Row], cursor.fetchall())

            # When user_id is provided, additionally compute that user's own counts so the
            # caller can render the per-user portion of the badge. These are returned in the
            # separate user_pending/user_in_progress fields and never replace the global counts.
            user_counts_result: list[sqlite3.Row] = []
            if user_id is not None:
                user_query = """--sql
                    SELECT status, count(*)
                    FROM session_queue
                    WHERE queue_id = ? AND user_id = ?
                    """
                user_params = [queue_id, user_id]

                if origin_prefix is not None:
                    user_query += " AND origin LIKE ?"
                    user_params.append(f"{origin_prefix}%")

                user_query += """--sql
                    GROUP BY status
                    """
                cursor.execute(user_query, user_params)
                user_counts_result = cast(list[sqlite3.Row], cursor.fetchall())

            # Only the four identifier columns, not a full SessionQueueItem: this runs on
            # every status poll, and hydrating the item would deserialize its whole session
            # graph. The origin filter mirrors the aggregate-count queries above so a
            # scoped caller never sees another scope's current item.
            current_item_query = """--sql
                SELECT item_id, session_id, batch_id, user_id
                FROM session_queue
                WHERE queue_id = ? AND status = 'in_progress'
                """
            current_item_params: list[str] = [queue_id]

            if origin_prefix is not None:
                current_item_query += " AND origin LIKE ?"
                current_item_params.append(f"{origin_prefix}%")

            current_item_query += " LIMIT 1"
            cursor.execute(current_item_query, current_item_params)
            current_item = cast(Union[sqlite3.Row, None], cursor.fetchone())

        total = sum(row[1] or 0 for row in counts_result)
        counts: dict[str, int] = {row[0]: row[1] for row in counts_result}

        user_pending: Optional[int] = None
        user_in_progress: Optional[int] = None
        if user_id is not None:
            user_counts: dict[str, int] = {row[0]: row[1] for row in user_counts_result}
            user_pending = user_counts.get("pending", 0)
            user_in_progress = user_counts.get("in_progress", 0)

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
        if current_item is not None and (is_admin or owner_user_id is None or current_item["user_id"] == owner_user_id):
            current_item_id = current_item["item_id"]
            current_session_id = current_item["session_id"]
            current_batch_id = current_item["batch_id"]

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
            total=total,
            user_pending=user_pending,
            user_in_progress=user_in_progress,
        )

    def get_batch_status(self, queue_id: str, batch_id: str, user_id: Optional[str] = None) -> BatchStatus:
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT status, count(*), origin, destination
                FROM session_queue
                WHERE queue_id = ? AND batch_id = ?
                """
            params: list[str] = [queue_id, batch_id]
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " GROUP BY status"
            cursor.execute(query, params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
        total = sum(row[1] or 0 for row in result)
        counts: dict[str, int] = {row[0]: row[1] for row in result}
        origin = result[0]["origin"] if result else None
        destination = result[0]["destination"] if result else None

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
            total=total,
        )

    def get_counts_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> SessionQueueCountsByDestination:
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ? AND destination = ?
                """
            params: list[str] = [queue_id, destination]
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " GROUP BY status"
            cursor.execute(query, params)
            counts_result = cast(list[sqlite3.Row], cursor.fetchall())

        total = sum(row[1] or 0 for row in counts_result)
        counts: dict[str, int] = {row[0]: row[1] for row in counts_result}

        return SessionQueueCountsByDestination(
            queue_id=queue_id,
            destination=destination,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            waiting=counts.get("waiting", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def retry_items_by_id(self, queue_id: str, item_ids: list[int]) -> RetryItemsResult:
        """Retries the given queue items"""
        with self._db.transaction() as cursor:
            values_to_insert: list[ValueToInsertTuple] = []
            retried_root_item_ids: list[int] = []
            retried_user_ids: list[str] = []
            retried_item_ids_by_user: dict[str, list[int]] = {}
            seen_root_item_ids: set[int] = set()
            hydrated_queue_items: dict[int, SessionQueueItem] = {}
            max_new_queue_items = self.__invoker.services.configuration.max_queue_size - self._get_current_queue_size(
                queue_id
            )

            if max_new_queue_items <= 0:
                return RetryItemsResult(queue_id=queue_id, retried_item_ids=[])

            for item_id in item_ids:
                if item_id in hydrated_queue_items:
                    queue_item = hydrated_queue_items[item_id]
                else:
                    try:
                        queue_item = self._get_queue_item_for_retry(item_id, cursor=cursor)
                    except SessionQueueItemNotFoundError:
                        continue
                    hydrated_queue_items[item_id] = queue_item
                if queue_item.queue_id != queue_id:
                    continue

                if queue_item.status not in ("failed", "canceled"):
                    continue

                root_item_id = queue_item.root_item_id or queue_item.item_id
                if root_item_id in seen_root_item_ids:
                    continue
                seen_root_item_ids.add(root_item_id)

                if root_item_id in hydrated_queue_items:
                    root_queue_item = hydrated_queue_items[root_item_id]
                else:
                    root_queue_item = self._get_queue_item_for_retry(root_item_id, cursor=cursor)
                    hydrated_queue_items[root_item_id] = root_queue_item
                if not root_queue_item._snapshot_readable:
                    continue
                if root_queue_item.status not in ("failed", "canceled"):
                    continue

                retried_root_item_ids.append(root_item_id)
                retried_user_ids.append(root_queue_item.user_id)
                retried_item_ids_by_user.setdefault(root_queue_item.user_id, []).append(root_item_id)

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
                # Validate the graph before dumping a fresh empty execution state. The full retry read above already
                # rehydrated runtime state for contract-compatible validation and recovery semantics.
                root_graph = Graph.model_validate(
                    root_queue_item.session.graph.model_dump(mode="python", warnings=False), strict=False
                )
                cloned_session = GraphExecutionState(graph=root_graph)
                cloned_session_json = json.dumps(dump_execution_state(cloned_session), default=to_jsonable_python)

                retried_from_item_id = (
                    root_queue_item.retried_from_item_id
                    if root_queue_item.retried_from_item_id is not None
                    else root_queue_item.item_id
                )

                value_to_insert: ValueToInsertTuple = (
                    root_queue_item.queue_id,
                    cloned_session_json,
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
                )
                values_to_insert.append(value_to_insert)

                if len(values_to_insert) >= max_new_queue_items:
                    break

            cursor.executemany(
                """--sql
                INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )

        retry_result = RetryItemsResult(
            queue_id=queue_id,
            retried_item_ids=retried_root_item_ids,
        )
        self.__invoker.services.events.emit_queue_items_retried(
            retry_result,
            user_ids=list(dict.fromkeys(retried_user_ids)),
            retried_item_ids_by_user=retried_item_ids_by_user,
        )
        return retry_result
