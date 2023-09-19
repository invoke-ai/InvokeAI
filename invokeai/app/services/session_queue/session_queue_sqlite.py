import sqlite3
import threading
from typing import Optional, Union, cast

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.graph import Graph
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    BatchStatus,
    CancelByBatchIDsResult,
    CancelByQueueIDResult,
    ClearResult,
    EnqueueBatchResult,
    EnqueueGraphResult,
    IsEmptyResult,
    IsFullResult,
    PruneResult,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueItemNotFoundError,
    SessionQueueStatus,
    calc_session_count,
    prepare_values_to_insert,
)
from invokeai.app.services.shared.models import CursorPaginatedResults


class SqliteSessionQueue(SessionQueueBase):
    __invoker: Invoker
    __conn: sqlite3.Connection
    __cursor: sqlite3.Cursor
    __lock: threading.Lock

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self._set_in_progress_to_canceled()
        prune_result = self.prune(DEFAULT_QUEUE_ID)
        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._on_session_event)
        self.__invoker.services.logger.info(f"Pruned {prune_result.deleted} finished queue items")

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock) -> None:
        super().__init__()
        self.__conn = conn
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self.__conn.row_factory = sqlite3.Row
        self.__cursor = self.__conn.cursor()
        self.__lock = lock
        self._create_tables()

    def _match_event_name(self, event: FastAPIEvent, match_in: list[str]) -> bool:
        return event[1]["event"] in match_in

    async def _on_session_event(self, event: FastAPIEvent) -> FastAPIEvent:
        event_name = event[1]["event"]
        match event_name:
            case "graph_execution_state_complete":
                await self._handle_complete_event(event)
            case "invocation_error" | "session_retrieval_error" | "invocation_retrieval_error":
                await self._handle_error_event(event)
            case "session_canceled":
                await self._handle_cancel_event(event)
        return event

    async def _handle_complete_event(self, event: FastAPIEvent) -> None:
        try:
            item_id = event[1]["data"]["queue_item_id"]
            # When a queue item has an error, we get an error event, then a completed event.
            # Mark the queue item completed only if it isn't already marked completed, e.g.
            # by a previously-handled error event.
            queue_item = self.get_queue_item(item_id)
            if queue_item.status not in ["completed", "failed", "canceled"]:
                queue_item = self._set_queue_item_status(item_id=queue_item.item_id, status="completed")
                self.__invoker.services.events.emit_queue_item_status_changed(queue_item)
        except SessionQueueItemNotFoundError:
            return

    async def _handle_error_event(self, event: FastAPIEvent) -> None:
        try:
            item_id = event[1]["data"]["queue_item_id"]
            error = event[1]["data"]["error"]
            queue_item = self.get_queue_item(item_id)
            queue_item = self._set_queue_item_status(item_id=queue_item.item_id, status="failed", error=error)
            self.__invoker.services.events.emit_queue_item_status_changed(queue_item)
        except SessionQueueItemNotFoundError:
            return

    async def _handle_cancel_event(self, event: FastAPIEvent) -> None:
        try:
            item_id = event[1]["data"]["queue_item_id"]
            queue_item = self.get_queue_item(item_id)
            queue_item = self._set_queue_item_status(item_id=queue_item.item_id, status="canceled")
            self.__invoker.services.events.emit_queue_item_status_changed(queue_item)
        except SessionQueueItemNotFoundError:
            return

    def _create_tables(self) -> None:
        """Creates the session queue tables, indicies, and triggers"""
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS session_queue (
                    item_id TEXT NOT NULL PRIMARY KEY, -- the unique identifier of this queue item
                    order_id INTEGER NOT NULL, -- used for ordering, cursor pagination
                    batch_id TEXT NOT NULL, -- identifier of the batch this queue item belongs to
                    queue_id TEXT NOT NULL, -- identifier of the queue this queue item belongs to
                    session_id TEXT NOT NULL UNIQUE, -- duplicated data from the session column, for ease of access
                    field_values TEXT, -- NULL if no values are associated with this queue item
                    session TEXT NOT NULL, -- the session to be executed
                    status TEXT NOT NULL DEFAULT 'pending', -- the status of the queue item, one of 'pending', 'in_progress', 'complete', 'error', 'canceled'
                    priority INTEGER NOT NULL DEFAULT 0, -- the priority, higher is more important
                    error TEXT, -- any errors associated with this queue item
                    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')), -- updated via trigger
                    started_at DATETIME, -- updated via trigger
                    completed_at DATETIME -- updated via trigger, completed items are cleaned up on application startup
                    -- Ideally this is a FK, but graph_executions uses INSERT OR REPLACE, and REPLACE triggers the ON DELETE CASCADE...
                    -- FOREIGN KEY (session_id) REFERENCES graph_executions (id) ON DELETE CASCADE
                );
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_item_id ON session_queue(item_id);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_order_id ON session_queue(order_id);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_session_id ON session_queue(session_id);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_session_queue_batch_id ON session_queue(batch_id);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_session_queue_created_priority ON session_queue(priority);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_session_queue_created_status ON session_queue(status);
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE TRIGGER IF NOT EXISTS tg_session_queue_completed_at
                AFTER UPDATE OF status ON session_queue
                FOR EACH ROW
                WHEN
                  NEW.status = 'completed'
                  OR NEW.status = 'failed'
                  OR NEW.status = 'canceled'
                BEGIN
                  UPDATE session_queue
                  SET completed_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                  WHERE item_id = NEW.item_id;
                END;
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE TRIGGER IF NOT EXISTS tg_session_queue_started_at
                AFTER UPDATE OF status ON session_queue
                FOR EACH ROW
                WHEN
                  NEW.status = 'in_progress'
                BEGIN
                  UPDATE session_queue
                  SET started_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                  WHERE item_id = NEW.item_id;
                END;
                """
            )

            self.__cursor.execute(
                """--sql
                CREATE TRIGGER IF NOT EXISTS tg_session_queue_updated_at
                AFTER UPDATE
                ON session_queue FOR EACH ROW
                BEGIN
                    UPDATE session_queue
                    SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE item_id = old.item_id;
                END;
                """
            )

            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()

    def _set_in_progress_to_canceled(self) -> None:
        """
        Sets all in_progress queue items to canceled.
        This is necessary because the invoker may have been killed while processing a queue item.
        """
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = 'canceled'
                WHERE status = 'in_progress';
                """
            )
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()

    def _get_current_queue_size(self, queue_id: str) -> int:
        """Gets the current number of pending queue items"""
        self.__cursor.execute(
            """--sql
            SELECT count(*)
            FROM session_queue
            WHERE
              queue_id = ?
              AND status = 'pending'
            """,
            (queue_id,),
        )
        return cast(int, self.__cursor.fetchone()[0])

    def _get_highest_priority(self, queue_id: str) -> int:
        """Gets the highest priority value in the queue"""
        self.__cursor.execute(
            """--sql
            SELECT MAX(priority)
            FROM session_queue
            WHERE
              queue_id = ?
              AND status = 'pending'
            """,
            (queue_id,),
        )
        return cast(Union[int, None], self.__cursor.fetchone()[0]) or 0

    def enqueue_graph(self, queue_id: str, graph: Graph, prepend: bool) -> EnqueueGraphResult:
        enqueue_result = self.enqueue_batch(queue_id=queue_id, batch=Batch(graph=graph), prepend=prepend)
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE queue_id = ?
                AND batch_id = ?
                """,
                (queue_id, enqueue_result.batch.batch_id),
            )
            result = cast(Union[sqlite3.Row, None], self.__cursor.fetchone())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with batch id {enqueue_result.batch.batch_id}")
        return EnqueueGraphResult(
            **enqueue_result.dict(),
            queue_item=SessionQueueItemDTO.from_dict(dict(result)),
        )

    def enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool) -> EnqueueBatchResult:
        try:
            self.__lock.acquire()

            # TODO: how does this work in a multi-user scenario?
            current_queue_size = self._get_current_queue_size(queue_id)
            max_queue_size = self.__invoker.services.configuration.get_config().max_queue_size
            max_new_queue_items = max_queue_size - current_queue_size

            priority = 0
            if prepend:
                priority = self._get_highest_priority(queue_id) + 1

            self.__cursor.execute(
                """--sql
                SELECT MAX(order_id)
                FROM session_queue
                """
            )
            max_order_id = cast(Optional[int], self.__cursor.fetchone()[0]) or 0

            requested_count = calc_session_count(batch)
            values_to_insert = prepare_values_to_insert(
                queue_id=queue_id,
                batch=batch,
                priority=priority,
                max_new_queue_items=max_new_queue_items,
                order_id=max_order_id + 1,
            )
            enqueued_count = len(values_to_insert)

            if requested_count > enqueued_count:
                values_to_insert = values_to_insert[:max_new_queue_items]

            self.__cursor.executemany(
                """--sql
                INSERT INTO session_queue (item_id, queue_id, session, session_id, batch_id, field_values, priority, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )
            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        enqueue_result = EnqueueBatchResult(
            queue_id=queue_id,
            requested=requested_count,
            enqueued=enqueued_count,
            batch=batch,
            priority=priority,
        )
        self.__invoker.services.events.emit_batch_enqueued(enqueue_result)
        return enqueue_result

    def dequeue(self) -> Optional[SessionQueueItem]:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE status = 'pending'
                ORDER BY
                  priority DESC,
                  order_id ASC
                LIMIT 1
                """
            )
            result = cast(Union[sqlite3.Row, None], self.__cursor.fetchone())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        if result is None:
            return None
        queue_item = SessionQueueItem.from_dict(dict(result))
        queue_item = self._set_queue_item_status(item_id=queue_item.item_id, status="in_progress")
        self.__invoker.services.events.emit_queue_item_status_changed(queue_item)
        return queue_item

    def get_next(self, queue_id: str) -> Optional[SessionQueueItem]:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
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
            result = cast(Union[sqlite3.Row, None], self.__cursor.fetchone())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        if result is None:
            return None
        return SessionQueueItem.from_dict(dict(result))

    def get_current(self, queue_id: str) -> Optional[SessionQueueItem]:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
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
            result = cast(Union[sqlite3.Row, None], self.__cursor.fetchone())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        if result is None:
            return None
        return SessionQueueItem.from_dict(dict(result))

    def _set_queue_item_status(
        self, item_id: str, status: QUEUE_ITEM_STATUS, error: Optional[str] = None
    ) -> SessionQueueItem:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = ?, error = ?
                WHERE
                  item_id = ?
                """,
                (status, error, item_id),
            )
            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return self.get_queue_item(item_id)

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            is_empty = cast(int, self.__cursor.fetchone()[0]) == 0
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return IsEmptyResult(is_empty=is_empty)

    def is_full(self, queue_id: str) -> IsFullResult:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            max_queue_size = self.__invoker.services.configuration.max_queue_size
            is_full = cast(int, self.__cursor.fetchone()[0]) >= max_queue_size
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return IsFullResult(is_full=is_full)

    def delete_queue_item(self, item_id: str) -> SessionQueueItem:
        queue_item = self.get_queue_item(item_id=item_id)
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                DELETE FROM session_queue
                WHERE
                  item_id = ?
                """,
                (item_id,),
            )
            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return queue_item

    def clear(self, queue_id: str) -> ClearResult:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            count = self.__cursor.fetchone()[0]
            self.__cursor.execute(
                """--sql
                DELETE
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        self.__invoker.services.events.emit_queue_cleared(queue_id)
        return ClearResult(deleted=count)

    def prune(self, queue_id: str) -> PruneResult:
        try:
            where = """--sql
                WHERE
                  queue_id = ?
                  AND (
                    status = 'completed'
                    OR status = 'failed'
                    OR status = 'canceled'
                  )
                """
            self.__lock.acquire()
            self.__cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            count = self.__cursor.fetchone()[0]
            self.__cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            self.__conn.commit()
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return PruneResult(deleted=count)

    def cancel_queue_item(self, item_id: str) -> SessionQueueItem:
        queue_item = self.get_queue_item(item_id)
        if queue_item.status not in ["canceled", "failed", "completed"]:
            queue_item = self._set_queue_item_status(item_id=item_id, status="canceled")
            self.__invoker.services.queue.cancel(queue_item.session_id)
            self.__invoker.services.events.emit_session_canceled(
                queue_item_id=queue_item.item_id,
                queue_id=queue_item.queue_id,
                graph_execution_state_id=queue_item.session_id,
            )
            self.__invoker.services.events.emit_queue_item_status_changed(queue_item)
        return queue_item

    def cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str]) -> CancelByBatchIDsResult:
        try:
            current_queue_item = self.get_current(queue_id)
            self.__lock.acquire()
            placeholders = ", ".join(["?" for _ in batch_ids])
            where = f"""--sql
                WHERE
                  queue_id == ?
                  AND batch_id IN ({placeholders})
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                """
            params = [queue_id] + batch_ids
            self.__cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
            count = self.__cursor.fetchone()[0]
            self.__cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                tuple(params),
            )
            self.__conn.commit()
            if current_queue_item is not None and current_queue_item.batch_id in batch_ids:
                self.__invoker.services.queue.cancel(current_queue_item.session_id)
                self.__invoker.services.events.emit_session_canceled(
                    queue_item_id=current_queue_item.item_id,
                    queue_id=current_queue_item.queue_id,
                    graph_execution_state_id=current_queue_item.session_id,
                )
                self.__invoker.services.events.emit_queue_item_status_changed(current_queue_item)
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return CancelByBatchIDsResult(canceled=count)

    def cancel_by_queue_id(self, queue_id: str) -> CancelByQueueIDResult:
        try:
            current_queue_item = self.get_current(queue_id)
            self.__lock.acquire()
            where = """--sql
                WHERE
                  queue_id is ?
                  AND status != 'canceled'
                  AND status != 'completed'
                  AND status != 'failed'
                """
            params = [queue_id]
            self.__cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
            count = self.__cursor.fetchone()[0]
            self.__cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                tuple(params),
            )
            self.__conn.commit()
            if current_queue_item is not None and current_queue_item.queue_id == queue_id:
                self.__invoker.services.queue.cancel(current_queue_item.session_id)
                self.__invoker.services.events.emit_session_canceled(
                    queue_item_id=current_queue_item.item_id,
                    queue_id=current_queue_item.queue_id,
                    graph_execution_state_id=current_queue_item.session_id,
                )
                self.__invoker.services.events.emit_queue_item_status_changed(current_queue_item)
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return CancelByQueueIDResult(canceled=count)

    def get_queue_item(self, item_id: str) -> SessionQueueItem:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE
                  item_id = ?
                """,
                (item_id,),
            )
            result = cast(Union[sqlite3.Row, None], self.__cursor.fetchone())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return SessionQueueItem.from_dict(dict(result))

    def list_queue_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        order_id: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
    ) -> CursorPaginatedResults[SessionQueueItemDTO]:
        try:
            self.__lock.acquire()
            query = """--sql
                SELECT item_id,
                    order_id,
                    status,
                    priority,
                    field_values,
                    error,
                    created_at,
                    updated_at,
                    completed_at,
                    started_at,
                    session_id,
                    batch_id,
                    queue_id
                FROM session_queue
                WHERE queue_id = ?
            """
            params: list[Union[str, int]] = [queue_id]

            if status is not None:
                query += """--sql
                    AND status = ?
                    """
                params.append(status)

            if order_id is not None:
                query += """--sql
                    AND (priority < ?) OR (priority = ? AND order_id > ?)
                    """
                params.extend([priority, priority, order_id])

            query += """--sql
                ORDER BY
                  priority DESC,
                  order_id ASC
                LIMIT ?
                """
            params.append(limit + 1)
            self.__cursor.execute(query, params)
            results = cast(list[sqlite3.Row], self.__cursor.fetchall())
            items = [SessionQueueItemDTO.from_dict(dict(result)) for result in results]
            has_more = False
            if len(items) > limit:
                # remove the extra item
                items.pop()
                has_more = True
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def get_queue_status(self, queue_id: str) -> SessionQueueStatus:
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ?
                GROUP BY status
                """,
                (queue_id,),
            )
            counts_result = cast(list[sqlite3.Row], self.__cursor.fetchall())
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()

        current_item = self.get_current(queue_id=queue_id)
        total = sum(row[1] for row in counts_result)
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
        try:
            self.__lock.acquire()
            self.__cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE
                  queue_id = ?
                  AND batch_id = ?
                GROUP BY status
                """,
                (queue_id, batch_id),
            )
            result = cast(list[sqlite3.Row], self.__cursor.fetchall())
            total = sum(row[1] for row in result)
            counts: dict[str, int] = {row[0]: row[1] for row in result}
        except Exception:
            self.__conn.rollback()
            raise
        finally:
            self.__lock.release()

        return BatchStatus(
            batch_id=batch_id,
            queue_id=queue_id,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )
