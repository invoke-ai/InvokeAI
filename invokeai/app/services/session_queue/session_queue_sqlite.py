import asyncio
import json
import sqlite3
from typing import Optional, Union, cast

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
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteSessionQueue(SessionQueueBase):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self._set_in_progress_to_canceled()
        if self.__invoker.services.configuration.clear_queue_on_startup:
            clear_result = self.clear(DEFAULT_QUEUE_ID)
            if clear_result.deleted > 0:
                self.__invoker.services.logger.info(f"Cleared all {clear_result.deleted} queue items")

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def _set_in_progress_to_canceled(self) -> None:
        """
        Sets all in_progress queue items to canceled. Run on app startup, not associated with any queue.
        This is necessary because the invoker may have been killed while processing a queue item.
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = 'canceled'
                WHERE status = 'in_progress';
                """
            )

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

    async def enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool) -> EnqueueBatchResult:
        current_queue_size = self._get_current_queue_size(queue_id)
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        max_new_queue_items = max_queue_size - current_queue_size

        priority = 0
        if prepend:
            priority = self._get_highest_priority(queue_id) + 1

        requested_count = await asyncio.to_thread(
            calc_session_count,
            batch=batch,
        )
        values_to_insert = await asyncio.to_thread(
            prepare_values_to_insert,
            queue_id=queue_id,
            batch=batch,
            priority=priority,
            max_new_queue_items=max_new_queue_items,
        )
        enqueued_count = len(values_to_insert)

        with self._db.transaction() as cursor:
            cursor.executemany(
                """--sql
                    INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                values_to_insert,
            )
            cursor.execute(
                """--sql
                    SELECT item_id
                    FROM session_queue
                    WHERE batch_id = ?
                    ORDER BY item_id DESC;
                    """,
                (batch.batch_id,),
            )
            item_ids = [row[0] for row in cursor.fetchall()]
        enqueue_result = EnqueueBatchResult(
            queue_id=queue_id,
            requested=requested_count,
            enqueued=enqueued_count,
            batch=batch,
            priority=priority,
            item_ids=item_ids,
        )
        self.__invoker.services.events.emit_batch_enqueued(enqueue_result)
        return enqueue_result

    def dequeue(self) -> Optional[SessionQueueItem]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE status = 'pending'
                ORDER BY
                    priority DESC,
                    item_id ASC
                LIMIT 1
                """
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        queue_item = SessionQueueItem.queue_item_from_dict(dict(result))
        queue_item = self._set_queue_item_status(item_id=queue_item.item_id, status="in_progress")
        return queue_item

    def get_next(self, queue_id: str) -> Optional[SessionQueueItem]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE
                    queue_id = ?
                    AND status = 'pending'
                ORDER BY
                    priority DESC,
                    created_at ASC
                LIMIT 1
                """,
                (queue_id,),
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        return SessionQueueItem.queue_item_from_dict(dict(result))

    def get_current(self, queue_id: str) -> Optional[SessionQueueItem]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE
                    queue_id = ?
                    AND status = 'in_progress'
                LIMIT 1
                """,
                (queue_id,),
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            return None
        return SessionQueueItem.queue_item_from_dict(dict(result))

    def _set_queue_item_status(
        self,
        item_id: int,
        status: QUEUE_ITEM_STATUS,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> SessionQueueItem:
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

        # Only update if not already finished (completed, failed or canceled)
        if current_status in ("completed", "failed", "canceled"):
            return self.get_queue_item(item_id)

        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = ?, error_type = ?, error_message = ?, error_traceback = ?
                WHERE item_id = ?
                """,
                (status, error_type, error_message, error_traceback, item_id),
            )

        queue_item = self.get_queue_item(item_id)
        batch_status = self.get_batch_status(queue_id=queue_item.queue_id, batch_id=queue_item.batch_id)
        queue_status = self.get_queue_status(queue_id=queue_item.queue_id)
        self.__invoker.services.events.emit_queue_item_status_changed(queue_item, batch_status, queue_status)
        return queue_item

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

    def clear(self, queue_id: str) -> ClearResult:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                """--sql
                DELETE
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
        self.__invoker.services.events.emit_queue_cleared(queue_id)
        return ClearResult(deleted=count)

    def prune(self, queue_id: str) -> PruneResult:
        with self._db.transaction() as cursor:
            where = """--sql
                WHERE
                queue_id = ?
                AND (
                    status = 'completed'
                    OR status = 'failed'
                    OR status = 'canceled'
                )
                """
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
        return PruneResult(deleted=count)

    def cancel_queue_item(self, item_id: int) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(item_id=item_id, status="canceled")
        return queue_item

    def delete_queue_item(self, item_id: int) -> None:
        """Deletes a session queue item"""
        try:
            self.cancel_queue_item(item_id)
        except SessionQueueItemNotFoundError:
            pass
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE
                FROM session_queue
                WHERE item_id = ?
                """,
                (item_id,),
            )

    def complete_queue_item(self, item_id: int) -> SessionQueueItem:
        queue_item = self._set_queue_item_status(item_id=item_id, status="completed")
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

    def cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str]) -> CancelByBatchIDsResult:
        with self._db.transaction() as cursor:
            current_queue_item = self.get_current(queue_id)
            placeholders = ", ".join(["?" for _ in batch_ids])
            where = f"""--sql
                WHERE
                  queue_id == ?
                  AND batch_id IN ({placeholders})
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- We will cancel the current item separately below - skip it here
                  AND status != 'in_progress'
                """
            params = [queue_id] + batch_ids
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
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                tuple(params),
            )

        if current_queue_item is not None and current_queue_item.batch_id in batch_ids:
            self._set_queue_item_status(current_queue_item.item_id, "canceled")

        return CancelByBatchIDsResult(canceled=count)

    def cancel_by_destination(self, queue_id: str, destination: str) -> CancelByDestinationResult:
        with self._db.transaction() as cursor:
            current_queue_item = self.get_current(queue_id)
            where = """--sql
                WHERE
                  queue_id == ?
                  AND destination == ?
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- We will cancel the current item separately below - skip it here
                  AND status != 'in_progress'
                """
            params = (queue_id, destination)
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                params,
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                params,
            )
        if current_queue_item is not None and current_queue_item.destination == destination:
            self._set_queue_item_status(current_queue_item.item_id, "canceled")
        return CancelByDestinationResult(canceled=count)

    def delete_by_destination(self, queue_id: str, destination: str) -> DeleteByDestinationResult:
        with self._db.transaction() as cursor:
            current_queue_item = self.get_current(queue_id)
            if current_queue_item is not None and current_queue_item.destination == destination:
                self.cancel_queue_item(current_queue_item.item_id)
            params = (queue_id, destination)
            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE
                  queue_id = ?
                  AND destination = ?;
                """,
                params,
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                """--sql
                DELETE
                FROM session_queue
                WHERE
                  queue_id = ?
                  AND destination = ?;
                """,
                params,
            )
        return DeleteByDestinationResult(deleted=count)

    def delete_all_except_current(self, queue_id: str) -> DeleteAllExceptCurrentResult:
        with self._db.transaction() as cursor:
            where = """--sql
                WHERE
                  queue_id == ?
                  AND status == 'pending'
                """
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
        return DeleteAllExceptCurrentResult(deleted=count)

    def cancel_by_queue_id(self, queue_id: str) -> CancelByQueueIDResult:
        with self._db.transaction() as cursor:
            current_queue_item = self.get_current(queue_id)
            where = """--sql
                WHERE
                  queue_id is ?
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                  -- We will cancel the current item separately below - skip it here
                  AND status != 'in_progress'
                """
            params = [queue_id]
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
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                tuple(params),
            )

        if current_queue_item is not None and current_queue_item.queue_id == queue_id:
            self._set_queue_item_status(current_queue_item.item_id, "canceled")
        return CancelByQueueIDResult(canceled=count)

    def cancel_all_except_current(self, queue_id: str) -> CancelAllExceptCurrentResult:
        with self._db.transaction() as cursor:
            where = """--sql
                WHERE
                  queue_id == ?
                  AND status == 'pending'
                """
            cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                (queue_id,),
            )
        return CancelAllExceptCurrentResult(canceled=count)

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE
                    item_id = ?
                """,
                (item_id,),
            )
            result = cast(Union[sqlite3.Row, None], cursor.fetchone())
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return SessionQueueItem.queue_item_from_dict(dict(result))

    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        with self._db.transaction() as cursor:
            # Use exclude_none so we don't end up with a bunch of nulls in the graph - this can cause validation errors
            # when the graph is loaded. Graph execution occurs purely in memory - the session saved here is not referenced
            # during execution.
            session_json = session.model_dump_json(warnings=False, exclude_none=True)
            cursor.execute(
                """--sql
                UPDATE session_queue
                SET session = ?
                WHERE item_id = ?
                """,
                (session_json, item_id),
            )
        return self.get_queue_item(item_id)

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
        items = [SessionQueueItem.queue_item_from_dict(dict(result)) for result in results]
        has_more = False
        if len(items) > limit:
            # remove the extra item
            items.pop()
            has_more = True
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def list_all_queue_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        """Gets all queue items that match the given parameters"""
        with self._db.transaction() as cursor:
            query = """--sql
                SELECT *
                FROM session_queue
                WHERE queue_id = ?
            """
            params: list[Union[str, int]] = [queue_id]

            if destination is not None:
                query += """---sql
                    AND destination = ?
                """
                params.append(destination)

            query += """--sql
                ORDER BY
                    priority DESC,
                    item_id ASC
                ;
                """
            cursor.execute(query, params)
            results = cast(list[sqlite3.Row], cursor.fetchall())
        items = [SessionQueueItem.queue_item_from_dict(dict(result)) for result in results]
        return items

    def get_queue_status(self, queue_id: str) -> SessionQueueStatus:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ?
                GROUP BY status
                """,
                (queue_id,),
            )
            counts_result = cast(list[sqlite3.Row], cursor.fetchall())

        current_item = self.get_current(queue_id=queue_id)
        total = sum(row[1] or 0 for row in counts_result)
        counts: dict[str, int] = {row[0]: row[1] for row in counts_result}
        return SessionQueueStatus(
            queue_id=queue_id,
            item_id=current_item.item_id if current_item else None,
            session_id=current_item.session_id if current_item else None,
            batch_id=current_item.batch_id if current_item else None,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def get_batch_status(self, queue_id: str, batch_id: str) -> BatchStatus:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status, count(*), origin, destination
                FROM session_queue
                WHERE
                    queue_id = ?
                    AND batch_id = ?
                GROUP BY status
                """,
                (queue_id, batch_id),
            )
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
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def get_counts_by_destination(self, queue_id: str, destination: str) -> SessionQueueCountsByDestination:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ?
                AND destination = ?
                GROUP BY status
                """,
                (queue_id, destination),
            )
            counts_result = cast(list[sqlite3.Row], cursor.fetchall())

        total = sum(row[1] or 0 for row in counts_result)
        counts: dict[str, int] = {row[0]: row[1] for row in counts_result}

        return SessionQueueCountsByDestination(
            queue_id=queue_id,
            destination=destination,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def retry_items_by_id(self, queue_id: str, item_ids: list[int]) -> RetryItemsResult:
        """Retries the given queue items"""
        with self._db.transaction() as cursor:
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

                value_to_insert: ValueToInsertTuple = (
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
                )
                values_to_insert.append(value_to_insert)

            # TODO(psyche): Handle max queue size?

            cursor.executemany(
                """--sql
                INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )

        retry_result = RetryItemsResult(
            queue_id=queue_id,
            retried_item_ids=retried_item_ids,
        )
        self.__invoker.services.events.emit_queue_items_retried(retry_result)
        return retry_result
