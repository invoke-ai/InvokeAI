import sqlite3
import threading
from typing import Optional, Union, cast

from invokeai.app.services.graph import Graph
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    CancelByBatchIDsResult,
    ClearResult,
    EnqueueBatchResult,
    EnqueueGraphResult,
    IsEmptyResult,
    IsFullResult,
    PruneResult,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueItemNotFoundError,
    SessionQueueStatusResult,
    SetManyQueueItemStatusResult,
    calc_session_count,
    prepare_values_to_insert,
)
from invokeai.app.services.shared.models import CursorPaginatedResults


class SqliteSessionQueue(SessionQueueBase):
    _invoker: Invoker
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock) -> None:
        super().__init__()
        self._conn = conn
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = lock

        try:
            self._lock.acquire()
            self._create_tables()
            self._conn.commit()
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        self._cursor.execute(
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
                completed_at DATETIME -- completed items are cleaned up on application startup
                -- Ideally this is a FK, but graph_executions uses INSERT OR REPLACE, and REPLACE triggers the ON DELETE CASCADE...
                -- FOREIGN KEY (session_id) REFERENCES graph_executions (id) ON DELETE CASCADE
            );
            """
        )

        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_item_id ON session_queue(item_id);
            """
        )

        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_order_id ON session_queue(order_id);
            """
        )

        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_session_id ON session_queue(session_id);
            """
        )

        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_batch_id ON session_queue(batch_id);
            """
        )

        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_created_priority ON session_queue(priority);
            """
        )

        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_created_status ON session_queue(status);
            """
        )

        self._cursor.execute(
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

        self._cursor.execute(
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

    def _set_in_progress_to_canceled(self) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = 'canceled'
                WHERE status = 'in_progress';
                """
            )
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _get_current_queue_size(self, queue_id: str) -> int:
        self._cursor.execute(
            """--sql
            SELECT count(*)
            FROM session_queue
            WHERE
              queue_id = ?
              AND status = 'pending'
            """,
            (queue_id,),
        )
        return cast(int, self._cursor.fetchone()[0])

    def _get_highest_priority(self, queue_id: str) -> int:
        self._cursor.execute(
            """--sql
            SELECT MAX(priority)
            FROM session_queue
            WHERE
              queue_id = ?
              AND status = 'pending'
            """,
            (queue_id,),
        )
        return cast(Union[int, None], self._cursor.fetchone()[0]) or 0

    def start_service(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._set_in_progress_to_canceled()
        prune_result = self.prune(DEFAULT_QUEUE_ID)
        self._invoker.services.logger.info(f"Pruned {prune_result.deleted} finished queue items")

    def enqueue_graph(self, queue_id: str, graph: Graph, prepend: bool) -> EnqueueGraphResult:
        enqueue_result = self.enqueue_batch(queue_id=queue_id, batch=Batch(graph=graph), prepend=prepend)
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM session_queue
                WHERE queue_id = ?
                AND batch_id = ?
                """,
                (queue_id, enqueue_result.batch.batch_id),
            )
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with batch id {enqueue_result.batch.batch_id}")
        return EnqueueGraphResult(
            **enqueue_result.dict(),
            queue_item=SessionQueueItemDTO.from_dict(dict(result)),
        )

    def enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool) -> EnqueueBatchResult:
        try:
            self._lock.acquire()

            # TODO: how does this work in a multi-user scenario?
            current_queue_size = self._get_current_queue_size(queue_id=queue_id)
            max_queue_size = self._invoker.services.configuration.get_config().max_queue_size
            max_new_queue_items = max_queue_size - current_queue_size

            priority = 0
            if prepend:
                priority = self._get_highest_priority(queue_id=queue_id) + 1

            self._cursor.execute(
                """--sql
                SELECT MAX(order_id)
                FROM session_queue
                """
            )
            max_order_id = cast(Optional[int], self._cursor.fetchone()[0]) or 0

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

            self._cursor.executemany(
                """--sql
                INSERT INTO session_queue (item_id, queue_id, session, session_id, batch_id, field_values, priority, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return EnqueueBatchResult(
            requested=requested_count,
            enqueued=enqueued_count,
            batch=batch,
            priority=priority,
        )

    def dequeue(self) -> Optional[SessionQueueItem]:
        try:
            self._lock.acquire()
            self._cursor.execute(
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
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            return None
        queue_item = SessionQueueItem.from_dict(dict(result))
        return self.set_queue_item_status(
            queue_id=queue_item.queue_id, item_id=queue_item.item_id, status="in_progress"
        )

    def peek(self, queue_id: str) -> Optional[SessionQueueItem]:
        try:
            self._lock.acquire()
            self._cursor.execute(
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
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            return None
        return SessionQueueItem.from_dict(dict(result))

    def set_queue_item_status(self, queue_id: str, item_id: str, status: QUEUE_ITEM_STATUS) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = ?
                WHERE
                  queue_id = ?
                  AND item_id = ?
                """,
                (status, queue_id, item_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get_queue_item(queue_id=queue_id, item_id=item_id)

    def set_many_queue_item_status(
        self, queue_id: str, item_ids: list[str], status: QUEUE_ITEM_STATUS
    ) -> SetManyQueueItemStatusResult:
        try:
            self._lock.acquire()

            # update the queue items
            placeholders = ", ".join(["?" for _ in item_ids])

            update_query = f"""--sql
            UPDATE session_queue
            SET status = ?
            WHERE
              queue_id in ?
              AND item_id IN ({placeholders})
            """

            self._cursor.execute(update_query, [queue_id, status] + item_ids)
            self._conn.commit()

            # get queue items from list which were set to the status successfully
            fetch_query = f"""--sql
            SELECT item_id
            FROM session_queue
            WHERE
              queue_id = ?
              AND status = ?
              AND item_id IN ({placeholders})
            """

            self._cursor.execute(fetch_query, [queue_id, status] + item_ids)
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

        updated_ids = [row[0] for row in result]
        return SetManyQueueItemStatusResult(item_ids=updated_ids, status=status)

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            is_empty = cast(int, self._cursor.fetchone()[0]) == 0
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return IsEmptyResult(is_empty=is_empty)

    def is_full(self, queue_id: str) -> IsFullResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            max_queue_size = self._invoker.services.configuration.max_queue_size
            is_full = cast(int, self._cursor.fetchone()[0]) >= max_queue_size
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return IsFullResult(is_full=is_full)

    def delete_queue_item(self, queue_id: str, item_id: str) -> SessionQueueItem:
        queue_item = self.get_queue_item(queue_id=queue_id, item_id=item_id)
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM session_queue
                WHERE
                  queue_id = ?
                  AND item_id = ?
                """,
                (queue_id, item_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return queue_item

    def clear(self, queue_id: str) -> ClearResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            count = self._cursor.fetchone()[0]
            self._cursor.execute(
                """--sql
                DELETE
                FROM session_queue
                WHERE queue_id = ?
                """,
                (queue_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
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
            self._lock.acquire()
            self._cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            count = self._cursor.fetchone()[0]
            self._cursor.execute(
                f"""--sql
                DELETE
                FROM session_queue
                {where};
                """,
                (queue_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return PruneResult(deleted=count)

    def cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str]) -> CancelByBatchIDsResult:
        try:
            self._lock.acquire()
            placeholders = ", ".join(["?" for _ in batch_ids])
            where = f"""--sql
                WHERE
                  queue_id = ?
                  AND batch_id IN ({placeholders})
                  AND status != 'canceled'
                  AND status != 'completed'
                """
            params = [queue_id] + batch_ids
            self._cursor.execute(
                f"""--sql
                SELECT COUNT(*)
                FROM session_queue
                {where};
                """,
                tuple(params),
            )
            count = self._cursor.fetchone()[0]
            self._cursor.execute(
                f"""--sql
                UPDATE session_queue
                SET status = 'canceled'
                {where};
                """,
                tuple(params),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return CancelByBatchIDsResult(canceled=count)

    def get_queue_item(self, queue_id: str, item_id: str) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE
                  queue_id = ?
                  AND item_id = ?
                """,
                (queue_id, item_id),
            )
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return SessionQueueItem.from_dict(dict(result))

    def get_queue_item_by_session_id(self, session_id: str) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE
                  session_id = ?
                """,
                (session_id,),
            )
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with session id {session_id}")
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
            self._lock.acquire()
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
            self._cursor.execute(query, params)
            results = cast(list[sqlite3.Row], self._cursor.fetchall())
            items = [SessionQueueItemDTO.from_dict(dict(result)) for result in results]
            has_more = False
            if len(items) > limit:
                # remove the extra item
                items.pop()
                has_more = True
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def get_status(self, queue_id: str) -> SessionQueueStatusResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                WHERE queue_id = ?
                GROUP BY status
                """,
                (queue_id,),
            )
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            total = sum(row[1] for row in result)
            counts: dict[str, int] = {row[0]: row[1] for row in result}
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

        return SessionQueueStatusResult(
            queue_id=queue_id,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
            max_queue_size=self._invoker.services.configuration.get_config().max_queue_size,
        )
