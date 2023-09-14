import sqlite3
import threading
from typing import Optional, Union, cast

from invokeai.app.services.graph import Graph
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    ClearResult,
    EnqueueResult,
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
                id INTEGER PRIMARY KEY AUTOINCREMENT, -- used for ordering, cursor pagination
                batch_id TEXT NOT NULL, -- identifier of the batch this queue item belongs to
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
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_id ON session_queue(id);
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
            WHEN NEW.status = 'completed' OR NEW.status = 'failed' or NEW.status = 'canceled'
            BEGIN
              UPDATE session_queue
              SET completed_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
              WHERE id = NEW.id;
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
                WHERE id = old.id;
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

    def start_service(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._set_in_progress_to_canceled()
        prune_result = self.prune()
        self._invoker.services.logger.info(f"Pruned {prune_result.deleted} finished queue items")

    def enqueue(self, graph: Graph, prepend: bool) -> EnqueueResult:
        return self.enqueue_batch(Batch(graph=graph), prepend=prepend)

    def _get_current_queue_size(self) -> int:
        self._cursor.execute(
            """--sql
            SELECT count(*)
            FROM session_queue
            WHERE status = 'pending'
            """
        )
        return cast(int, self._cursor.fetchone()[0])

    def _get_highest_priority(self) -> int:
        self._cursor.execute(
            """--sql
            SELECT MAX(priority)
            FROM session_queue
            WHERE status = 'pending'
            """
        )
        return cast(Union[int, None], self._cursor.fetchone()[0]) or 0

    def enqueue_batch(self, batch: Batch, prepend: bool) -> EnqueueResult:
        try:
            self._lock.acquire()

            # TODO: how does this work in a multi-user scenario?
            current_queue_size = self._get_current_queue_size()
            max_queue_size = self._invoker.services.configuration.get_config().max_queue_size
            max_new_queue_items = max_queue_size - current_queue_size

            priority = 0
            if prepend:
                priority = self._get_highest_priority() + 1

            requested_count = calc_session_count(batch)
            values_to_insert = prepare_values_to_insert(batch, priority, max_new_queue_items)
            enqueued_count = len(values_to_insert)

            if requested_count > enqueued_count:
                values_to_insert = values_to_insert[:max_new_queue_items]

            self._cursor.executemany(
                """--sql
                INSERT INTO session_queue (session, session_id, batch_id, field_values, priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                values_to_insert,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return EnqueueResult(
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
                SELECT id FROM session_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC -- created_at doesn't have high enough precision to be used for ordering
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
        return self.set_queue_item_status(result[0], "in_progress")

    def peek(self) -> Optional[SessionQueueItem]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
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
        return SessionQueueItem.from_dict(dict(result))

    def set_queue_item_status(self, id: int, status: QUEUE_ITEM_STATUS) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE session_queue
                SET status = ?
                WHERE id = ?
                """,
                (status, id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get_queue_item(id)

    def set_many_queue_item_status(self, ids: list[str], status: QUEUE_ITEM_STATUS) -> SetManyQueueItemStatusResult:
        try:
            self._lock.acquire()

            # update the queue items
            placeholders = ", ".join(["?" for _ in ids])

            update_query = f"""--sql
            UPDATE session_queue
            SET status = ?
            WHERE id IN ({placeholders})
            """

            self._cursor.execute(update_query, [status] + ids)
            self._conn.commit()

            # get queue items from list which were set to the status successfully
            fetch_query = f"""--sql
            SELECT id
            FROM session_queue
            WHERE status = ? AND id IN ({placeholders})
            """

            self._cursor.execute(fetch_query, [status] + ids)
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

        updated_ids = [row[0] for row in result]
        return SetManyQueueItemStatusResult(ids=updated_ids, status=status)

    def is_empty(self) -> IsEmptyResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                """
            )
            is_empty = cast(int, self._cursor.fetchone()[0]) == 0
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return IsEmptyResult(is_empty=is_empty)

    def is_full(self) -> IsFullResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT count(*)
                FROM session_queue
                """
            )
            max_queue_size = self._invoker.services.configuration.max_queue_size
            is_full = cast(int, self._cursor.fetchone()[0]) >= max_queue_size
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return IsFullResult(is_full=is_full)

    def delete_queue_item(self, id: int) -> SessionQueueItem:
        queue_item = self.get_queue_item(id)
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM session_queue
                WHERE id = ?
                """,
                (id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return queue_item

    def clear(self) -> ClearResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT COUNT(*) FROM session_queue
                """
            )
            count = self._cursor.fetchone()[0]
            self._cursor.execute(
                """--sql
                DELETE FROM session_queue
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return ClearResult(deleted=count)

    def prune(self) -> PruneResult:
        try:
            where = "WHERE status = 'completed' OR status = 'failed' OR status = 'canceled'"
            self._lock.acquire()
            self._cursor.execute(
                f"""--sql
                SELECT COUNT(*) FROM session_queue
                {where};
                """
            )
            count = self._cursor.fetchone()[0]
            self._cursor.execute(
                f"""--sql
                DELETE FROM session_queue
                {where};
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return PruneResult(deleted=count)

    def get_queue_item(self, id: int) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE id = ?
                """,
                (id,),
            )
            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        if result is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {id}")
        return SessionQueueItem.from_dict(dict(result))

    def get_queue_item_by_session_id(self, session_id: str) -> SessionQueueItem:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT * FROM session_queue
                WHERE session_id = ?
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
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
    ) -> CursorPaginatedResults[SessionQueueItemDTO]:
        try:
            self._lock.acquire()
            query = """--sql
                SELECT id,
                    status,
                    priority,
                    field_values,
                    error,
                    created_at,
                    updated_at,
                    completed_at,
                    session_id,
                    batch_id
                FROM session_queue
                WHERE 1 = 1
            """
            params = []

            if status is not None:
                query += " AND status = ?"
                params.append(status)

            if cursor is not None:
                query += " AND (priority < ?) OR (priority = ? AND id > ?)"
                params.extend([priority, priority, cursor])

            query += " ORDER BY priority DESC, id ASC LIMIT ?"
            params.append(limit + 1)
            self._cursor.execute(query, tuple(params))
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

    def get_status(self) -> SessionQueueStatusResult:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT status, count(*)
                FROM session_queue
                GROUP BY status
                """
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
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
            max_queue_size=self._invoker.services.configuration.get_config().max_queue_size,
        )
