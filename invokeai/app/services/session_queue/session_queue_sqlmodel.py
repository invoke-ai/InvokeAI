"""SQLModel-backed implementation of the session queue service.

All SQL lives in the central ``DbQueries`` wrapper (``db.queries``); this service keeps
the orchestration: event emission, config access, async value preparation and the
current-item handling around bulk cancel/delete operations.
"""

import asyncio
import json
from typing import Optional

from pydantic_core import to_jsonable_python

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
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
    EnqueueBatchResult,
    IsEmptyResult,
    IsFullResult,
    ItemIdsResult,
    PruneResult,
    RetryItemsResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
    SessionQueueStatus,
    ValueToInsertTuple,
    calc_session_count,
    prepare_values_to_insert,
)
from invokeai.app.services.shared.graph import GraphExecutionState
from invokeai.app.services.shared.pagination import CursorPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

_TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "canceled")


class SqlModelSessionQueue(SessionQueueBase):
    __invoker: Invoker

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self._q.queue_set_in_progress_to_canceled()
        config = self.__invoker.services.configuration
        if config.clear_queue_on_startup:
            clear_result = self.clear(DEFAULT_QUEUE_ID)
            if clear_result.deleted > 0:
                self.__invoker.services.logger.info(f"Cleared all {clear_result.deleted} queue items")
            return

        if config.max_queue_history is not None:
            deleted = self._q.queue_prune_terminal_to_limit(DEFAULT_QUEUE_ID, config.max_queue_history)
            if deleted > 0:
                self.__invoker.services.logger.info(
                    f"Pruned {deleted} completed/failed/canceled queue items "
                    f"(kept up to {config.max_queue_history})"
                )

    # region: enqueue / dequeue / read single

    async def enqueue_batch(
        self, queue_id: str, batch: Batch, prepend: bool, user_id: str = "system"
    ) -> EnqueueBatchResult:
        current_queue_size = self._q.queue_pending_count(queue_id)
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        max_new_queue_items = max_queue_size - current_queue_size

        priority = 0
        if prepend:
            priority = self._q.queue_highest_pending_priority(queue_id) + 1

        requested_count = await asyncio.to_thread(calc_session_count, batch=batch)
        values_to_insert = await asyncio.to_thread(
            prepare_values_to_insert,
            queue_id=queue_id,
            batch=batch,
            priority=priority,
            max_new_queue_items=max_new_queue_items,
            user_id=user_id,
        )
        enqueued_count = len(values_to_insert)

        item_ids = self._q.queue_enqueue_values(values_to_insert, batch.batch_id)

        enqueue_result = EnqueueBatchResult(
            queue_id=queue_id,
            requested=requested_count,
            enqueued=enqueued_count,
            batch=batch,
            priority=priority,
            item_ids=item_ids,
        )
        self.__invoker.services.events.emit_batch_enqueued(enqueue_result, user_id=user_id)
        return enqueue_result

    def dequeue(self) -> Optional[SessionQueueItem]:
        queue_item = self._q.queue_get_next_pending_any_queue()
        if queue_item is None:
            return None
        return self._set_queue_item_status(item_id=queue_item.item_id, status="in_progress")

    def get_next(self, queue_id: str) -> Optional[SessionQueueItem]:
        return self._q.queue_get_next_pending(queue_id)

    def get_current(self, queue_id: str) -> Optional[SessionQueueItem]:
        return self._q.queue_get_in_progress(queue_id)

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        return self._q.queue_get_item(item_id)

    # endregion

    # region: status mutation

    def _set_queue_item_status(
        self,
        item_id: int,
        status: QUEUE_ITEM_STATUS,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> SessionQueueItem:
        current_status = self._q.queue_set_status_returning_prior(
            item_id=item_id,
            status=status,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
        )

        queue_item = self.get_queue_item(item_id)

        # If we did not update (item was already terminal), do not emit a status change event.
        if current_status not in _TERMINAL_STATUSES:
            batch_status = self.get_batch_status(queue_id=queue_item.queue_id, batch_id=queue_item.batch_id)
            queue_status = self.get_queue_status(queue_id=queue_item.queue_id)
            self.__invoker.services.events.emit_queue_item_status_changed(queue_item, batch_status, queue_status)
        return queue_item

    def cancel_queue_item(self, item_id: int) -> SessionQueueItem:
        return self._set_queue_item_status(item_id=item_id, status="canceled")

    def complete_queue_item(self, item_id: int) -> SessionQueueItem:
        return self._set_queue_item_status(item_id=item_id, status="completed")

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

    def delete_queue_item(self, item_id: int) -> None:
        try:
            self.cancel_queue_item(item_id)
        except SessionQueueItemNotFoundError:
            pass
        self._q.queue_delete_item(item_id)

    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        # Use exclude_none so we don't end up with a bunch of nulls in the graph - this can cause
        # validation errors when the graph is loaded. Graph execution occurs purely in memory - the
        # session saved here is not referenced during execution.
        session_json = session.model_dump_json(warnings=False, exclude_none=True)
        self._q.queue_update_session_json(item_id, session_json)
        return self.get_queue_item(item_id)

    # endregion

    # region: simple status checks

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        return IsEmptyResult(is_empty=self._q.queue_count(queue_id) == 0)

    def is_full(self, queue_id: str) -> IsFullResult:
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        return IsFullResult(is_full=self._q.queue_count(queue_id) >= max_queue_size)

    # endregion

    # region: bulk delete

    def clear(self, queue_id: str, user_id: Optional[str] = None) -> ClearResult:
        deleted = self._q.queue_clear(queue_id, user_id)
        self.__invoker.services.events.emit_queue_cleared(queue_id)
        return ClearResult(deleted=deleted)

    def prune(self, queue_id: str, user_id: Optional[str] = None) -> PruneResult:
        return PruneResult(deleted=self._q.queue_prune_terminal(queue_id, user_id))

    def delete_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> DeleteByDestinationResult:
        # Handle current in-progress item BEFORE the bulk delete, so it is properly
        # canceled (with events) instead of silently removed.
        current_queue_item = self.get_current(queue_id)
        if current_queue_item is not None and current_queue_item.destination == destination:
            if user_id is None or current_queue_item.user_id == user_id:
                self.cancel_queue_item(current_queue_item.item_id)

        deleted = self._q.queue_delete_by_destination(queue_id, destination, user_id)
        return DeleteByDestinationResult(deleted=deleted)

    def delete_all_except_current(self, queue_id: str, user_id: Optional[str] = None) -> DeleteAllExceptCurrentResult:
        return DeleteAllExceptCurrentResult(deleted=self._q.queue_delete_pending(queue_id, user_id))

    # endregion

    # region: bulk cancel

    def cancel_by_batch_ids(
        self, queue_id: str, batch_ids: list[str], user_id: Optional[str] = None
    ) -> CancelByBatchIDsResult:
        current_queue_item = self.get_current(queue_id)
        count = self._q.queue_cancel_by_batch_ids(queue_id, batch_ids, user_id)

        # Handle current item separately - check ownership if user_id is provided
        if current_queue_item is not None and current_queue_item.batch_id in batch_ids:
            if user_id is None or current_queue_item.user_id == user_id:
                self._set_queue_item_status(current_queue_item.item_id, "canceled")

        return CancelByBatchIDsResult(canceled=count)

    def cancel_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> CancelByDestinationResult:
        current_queue_item = self.get_current(queue_id)
        count = self._q.queue_cancel_by_destination(queue_id, destination, user_id)

        if current_queue_item is not None and current_queue_item.destination == destination:
            if user_id is None or current_queue_item.user_id == user_id:
                self._set_queue_item_status(current_queue_item.item_id, "canceled")

        return CancelByDestinationResult(canceled=count)

    def cancel_by_queue_id(self, queue_id: str) -> CancelByQueueIDResult:
        current_queue_item = self.get_current(queue_id)
        count = self._q.queue_cancel_by_queue_id(queue_id)

        if current_queue_item is not None and current_queue_item.queue_id == queue_id:
            self._set_queue_item_status(current_queue_item.item_id, "canceled")
        return CancelByQueueIDResult(canceled=count)

    def cancel_all_except_current(
        self, queue_id: str, user_id: Optional[str] = None
    ) -> CancelAllExceptCurrentResult:
        return CancelAllExceptCurrentResult(canceled=self._q.queue_cancel_pending(queue_id, user_id))

    # endregion

    # region: list / pagination

    def list_queue_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
        destination: Optional[str] = None,
    ) -> CursorPaginatedResults[SessionQueueItem]:
        return self._q.queue_list_items(
            queue_id=queue_id,
            limit=limit,
            priority=priority,
            cursor=cursor,
            status=status,
            destination=destination,
        )

    def list_all_queue_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        return self._q.queue_list_all_items(queue_id, destination)

    def get_queue_item_ids(
        self,
        queue_id: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        user_id: Optional[str] = None,
    ) -> ItemIdsResult:
        item_ids = self._q.queue_get_item_ids(queue_id, order_dir, user_id)
        return ItemIdsResult(item_ids=item_ids, total_count=len(item_ids))

    # endregion

    # region: aggregations

    def get_queue_status(
        self,
        queue_id: str,
        user_id: Optional[str] = None,
        acting_user_id: Optional[str] = None,
    ) -> SessionQueueStatus:
        counts = self._q.queue_status_counts(queue_id, user_id)
        current_item = self.get_current(queue_id=queue_id)
        total = sum(counts.values())

        # user_id filters the counts; acting_user_id decides current-item redaction (who is asking),
        # falling back to user_id. Hide current item details for non-admins who don't own it.
        owner_user_id = user_id if acting_user_id is None else acting_user_id
        show_current_item = current_item is not None and (
            owner_user_id is None or current_item.user_id == owner_user_id
        )

        return SessionQueueStatus(
            queue_id=queue_id,
            item_id=current_item.item_id if show_current_item else None,
            session_id=current_item.session_id if show_current_item else None,
            batch_id=current_item.batch_id if show_current_item else None,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def get_batch_status(self, queue_id: str, batch_id: str, user_id: Optional[str] = None) -> BatchStatus:
        return self._q.queue_get_batch_status(queue_id, batch_id, user_id)

    def get_counts_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> SessionQueueCountsByDestination:
        return self._q.queue_get_counts_by_destination(queue_id, destination, user_id)

    # endregion

    # region: retry

    def retry_items_by_id(self, queue_id: str, item_ids: list[int]) -> RetryItemsResult:
        values_to_insert: list[ValueToInsertTuple] = []
        retried_item_ids: list[int] = []

        for item_id in item_ids:
            queue_item = self.get_queue_item(item_id)
            if queue_item.status not in ("failed", "canceled"):
                continue
            retried_item_ids.append(item_id)

            field_values_json = (
                json.dumps(queue_item.field_values, default=to_jsonable_python) if queue_item.field_values else None
            )
            workflow_json = (
                json.dumps(queue_item.workflow, default=to_jsonable_python) if queue_item.workflow else None
            )
            cloned_session = GraphExecutionState(graph=queue_item.session.graph)
            cloned_session_json = cloned_session.model_dump_json(warnings=False, exclude_none=True)

            retried_from_item_id = (
                queue_item.retried_from_item_id
                if queue_item.retried_from_item_id is not None
                else queue_item.item_id
            )

            values_to_insert.append(
                (
                    queue_item.queue_id,
                    cloned_session_json,
                    cloned_session.id,
                    queue_item.batch_id,
                    field_values_json,
                    queue_item.priority,
                    workflow_json,
                    queue_item.origin,
                    queue_item.destination,
                    retried_from_item_id,
                    queue_item.user_id,
                )
            )

        # TODO(psyche): Handle max queue size?
        self._q.queue_insert_values(values_to_insert)

        retry_result = RetryItemsResult(queue_id=queue_id, retried_item_ids=retried_item_ids)
        self.__invoker.services.events.emit_queue_items_retried(retry_result)
        return retry_result

    # endregion
