"""SQLModel-backed implementation of the session queue service.

This module is the Phase 3 sibling of `session_queue_sqlite.py`. It uses
SQLAlchemy Core for the hot paths (bulk enqueue/cancel/delete, dequeue, list
with cursor pagination, aggregations) and keeps the same external behaviour as
the raw-SQL implementation, including reliance on the existing DB triggers for
`started_at`, `completed_at` and `updated_at`.
"""

import asyncio
import json
from typing import Any, Optional

from pydantic_core import to_jsonable_python
from sqlalchemy import and_, delete, func, insert, or_, select, update
from sqlalchemy.engine import Row

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
from invokeai.app.services.shared.sqlite.models import SessionQueueTable, UserTable
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

_TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "canceled")

_QUEUE_COLUMNS = (
    SessionQueueTable.item_id,
    SessionQueueTable.batch_id,
    SessionQueueTable.queue_id,
    SessionQueueTable.session_id,
    SessionQueueTable.field_values,
    SessionQueueTable.session,
    SessionQueueTable.status,
    SessionQueueTable.priority,
    SessionQueueTable.error_traceback,
    SessionQueueTable.created_at,
    SessionQueueTable.updated_at,
    SessionQueueTable.started_at,
    SessionQueueTable.completed_at,
    SessionQueueTable.error_type,
    SessionQueueTable.error_message,
    SessionQueueTable.origin,
    SessionQueueTable.destination,
    SessionQueueTable.retried_from_item_id,
    SessionQueueTable.user_id,
)


def _row_to_queue_item_dict(row: Row) -> dict[str, Any]:
    """Convert a Row produced by `_select_queue_item_with_user` to a plain dict
    that `SessionQueueItem.queue_item_from_dict` expects."""
    mapping = dict(row._mapping)
    # Stringify datetime columns so the Pydantic union (`datetime | str`) accepts them
    # consistently across queries that JOIN datetime columns from multiple tables.
    for ts_key in ("created_at", "updated_at", "started_at", "completed_at"):
        ts_value = mapping.get(ts_key)
        if ts_value is not None and not isinstance(ts_value, str):
            mapping[ts_key] = str(ts_value)
    mapping.setdefault("user_display_name", None)
    mapping.setdefault("user_email", None)
    mapping.setdefault("workflow", None)
    return mapping


def _select_queue_item_with_user():
    """Build a SELECT that mirrors `sq.*, u.display_name, u.email` with LEFT JOIN."""
    return (
        select(
            *_QUEUE_COLUMNS,
            SessionQueueTable.workflow,
            UserTable.display_name.label("user_display_name"),
            UserTable.email.label("user_email"),
        )
        .select_from(SessionQueueTable)
        .join(UserTable, SessionQueueTable.user_id == UserTable.user_id, isouter=True)
    )


def _value_tuple_to_dict(t: ValueToInsertTuple) -> dict[str, Any]:
    """Adapt the positional tuple from `prepare_values_to_insert` to a dict that
    SQLAlchemy Core's `insert(...).values([...])` expects."""
    return {
        "queue_id": t[0],
        "session": t[1],
        "session_id": t[2],
        "batch_id": t[3],
        "field_values": t[4],
        "priority": t[5],
        "workflow": t[6],
        "origin": t[7],
        "destination": t[8],
        "retried_from_item_id": t[9],
        "user_id": t[10],
    }


class SqlModelSessionQueue(SessionQueueBase):
    __invoker: Invoker

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

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
                    f"Pruned {deleted} completed/failed/canceled queue items "
                    f"(kept up to {config.max_queue_history})"
                )

    # region: internal helpers

    def _set_in_progress_to_canceled(self) -> None:
        """Sets all in_progress queue items to canceled. Run on app startup."""
        with self._db.get_session() as session:
            session.execute(
                update(SessionQueueTable)
                .where(SessionQueueTable.status == "in_progress")
                .values(status="canceled")
            )

    def _prune_terminal_to_limit(self, queue_id: str, keep: int) -> int:
        """Prune terminal items (completed/failed/canceled) to keep at most N most-recent items."""
        terminal_filter = and_(
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.in_(_TERMINAL_STATUSES),
        )
        # Subquery: ids of the items we want to keep (most recent N)
        keep_ids_stmt = (
            select(SessionQueueTable.item_id)
            .where(terminal_filter)
            .order_by(
                func.coalesce(
                    SessionQueueTable.completed_at,
                    SessionQueueTable.updated_at,
                    SessionQueueTable.created_at,
                ).desc(),
                SessionQueueTable.item_id.desc(),
            )
            .limit(keep)
        )
        with self._db.get_session() as session:
            count_stmt = (
                select(func.count())
                .select_from(SessionQueueTable)
                .where(terminal_filter)
                .where(~SessionQueueTable.item_id.in_(keep_ids_stmt))
            )
            count = session.execute(count_stmt).scalar_one()
            session.execute(
                delete(SessionQueueTable)
                .where(terminal_filter)
                .where(~SessionQueueTable.item_id.in_(keep_ids_stmt))
            )
        return int(count)

    def _get_current_queue_size(self, queue_id: str) -> int:
        """Gets the current number of pending queue items."""
        with self._db.get_readonly_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(SessionQueueTable)
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
            ).scalar_one()
        return int(count)

    def _get_highest_priority(self, queue_id: str) -> int:
        """Gets the highest priority value in the queue."""
        with self._db.get_readonly_session() as session:
            priority = session.execute(
                select(func.max(SessionQueueTable.priority)).where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
            ).scalar()
        return int(priority) if priority is not None else 0

    # endregion

    # region: enqueue / dequeue / read single

    async def enqueue_batch(
        self, queue_id: str, batch: Batch, prepend: bool, user_id: str = "system"
    ) -> EnqueueBatchResult:
        current_queue_size = self._get_current_queue_size(queue_id)
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        max_new_queue_items = max_queue_size - current_queue_size

        priority = 0
        if prepend:
            priority = self._get_highest_priority(queue_id) + 1

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

        with self._db.get_session() as session:
            if values_to_insert:
                session.execute(
                    insert(SessionQueueTable),
                    [_value_tuple_to_dict(v) for v in values_to_insert],
                )
            item_ids_rows = session.execute(
                select(SessionQueueTable.item_id)
                .where(SessionQueueTable.batch_id == batch.batch_id)
                .order_by(SessionQueueTable.item_id.desc())
            ).all()
        item_ids = [row[0] for row in item_ids_rows]

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
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _select_queue_item_with_user()
                .where(SessionQueueTable.status == "pending")
                .order_by(SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc())
                .limit(1)
            ).first()
        if row is None:
            return None
        queue_item = SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(row))
        return self._set_queue_item_status(item_id=queue_item.item_id, status="in_progress")

    def get_next(self, queue_id: str) -> Optional[SessionQueueItem]:
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _select_queue_item_with_user()
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
                .order_by(SessionQueueTable.priority.desc(), SessionQueueTable.created_at.asc())
                .limit(1)
            ).first()
        if row is None:
            return None
        return SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(row))

    def get_current(self, queue_id: str) -> Optional[SessionQueueItem]:
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _select_queue_item_with_user()
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "in_progress",
                )
                .limit(1)
            ).first()
        if row is None:
            return None
        return SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(row))

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _select_queue_item_with_user().where(SessionQueueTable.item_id == item_id)
            ).first()
        if row is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(row))

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
        with self._db.get_session() as session:
            current_status = session.execute(
                select(SessionQueueTable.status).where(SessionQueueTable.item_id == item_id)
            ).scalar()
            if current_status is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")

            # Only update if not already finished (completed, failed or canceled)
            if current_status in _TERMINAL_STATUSES:
                # No update; fall through to fetch + return below.
                pass
            else:
                session.execute(
                    update(SessionQueueTable)
                    .where(SessionQueueTable.item_id == item_id)
                    .values(
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                        error_traceback=error_traceback,
                    )
                )

        queue_item = self.get_queue_item(item_id)

        # If we did not update, do not emit a status change event.
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
        with self._db.get_session() as session:
            session.execute(delete(SessionQueueTable).where(SessionQueueTable.item_id == item_id))

    def set_queue_item_session(self, item_id: int, session_state: GraphExecutionState) -> SessionQueueItem:
        # Use exclude_none so we don't end up with a bunch of nulls in the graph - this can cause
        # validation errors when the graph is loaded. Graph execution occurs purely in memory - the
        # session saved here is not referenced during execution.
        session_json = session_state.model_dump_json(warnings=False, exclude_none=True)
        with self._db.get_session() as session:
            session.execute(
                update(SessionQueueTable)
                .where(SessionQueueTable.item_id == item_id)
                .values(session=session_json)
            )
        return self.get_queue_item(item_id)

    # endregion

    # region: simple status checks

    def is_empty(self, queue_id: str) -> IsEmptyResult:
        with self._db.get_readonly_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(SessionQueueTable)
                .where(SessionQueueTable.queue_id == queue_id)
            ).scalar_one()
        return IsEmptyResult(is_empty=int(count) == 0)

    def is_full(self, queue_id: str) -> IsFullResult:
        with self._db.get_readonly_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(SessionQueueTable)
                .where(SessionQueueTable.queue_id == queue_id)
            ).scalar_one()
        max_queue_size = self.__invoker.services.configuration.max_queue_size
        return IsFullResult(is_full=int(count) >= max_queue_size)

    # endregion

    # region: bulk delete

    def clear(self, queue_id: str, user_id: Optional[str] = None) -> ClearResult:
        where = [SessionQueueTable.queue_id == queue_id]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)

        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(delete(SessionQueueTable).where(*where))
        self.__invoker.services.events.emit_queue_cleared(queue_id)
        return ClearResult(deleted=int(count))

    def prune(self, queue_id: str, user_id: Optional[str] = None) -> PruneResult:
        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.in_(_TERMINAL_STATUSES),
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)

        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(delete(SessionQueueTable).where(*where))
        return PruneResult(deleted=int(count))

    def delete_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> DeleteByDestinationResult:
        # Handle current in-progress item BEFORE opening a write session of our own,
        # to avoid nested writes on the single StaticPool connection.
        current_queue_item = self.get_current(queue_id)
        if current_queue_item is not None and current_queue_item.destination == destination:
            if user_id is None or current_queue_item.user_id == user_id:
                self.cancel_queue_item(current_queue_item.item_id)

        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.destination == destination,
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)

        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(delete(SessionQueueTable).where(*where))
        return DeleteByDestinationResult(deleted=int(count))

    def delete_all_except_current(
        self, queue_id: str, user_id: Optional[str] = None
    ) -> DeleteAllExceptCurrentResult:
        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status == "pending",
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)

        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(delete(SessionQueueTable).where(*where))
        return DeleteAllExceptCurrentResult(deleted=int(count))

    # endregion

    # region: bulk cancel

    def _cancel_skip_in_progress_filter(
        self, queue_id: str, user_id: Optional[str], extra: list
    ) -> list:
        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.notin_(("canceled", "completed", "failed", "in_progress")),
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        where.extend(extra)
        return where

    def cancel_by_batch_ids(
        self, queue_id: str, batch_ids: list[str], user_id: Optional[str] = None
    ) -> CancelByBatchIDsResult:
        current_queue_item = self.get_current(queue_id)
        where = self._cancel_skip_in_progress_filter(
            queue_id, user_id, [SessionQueueTable.batch_id.in_(batch_ids)]
        )
        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(update(SessionQueueTable).where(*where).values(status="canceled"))

        # Handle current item separately - check ownership if user_id is provided
        if current_queue_item is not None and current_queue_item.batch_id in batch_ids:
            if user_id is None or current_queue_item.user_id == user_id:
                self._set_queue_item_status(current_queue_item.item_id, "canceled")

        return CancelByBatchIDsResult(canceled=int(count))

    def cancel_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> CancelByDestinationResult:
        current_queue_item = self.get_current(queue_id)
        where = self._cancel_skip_in_progress_filter(
            queue_id, user_id, [SessionQueueTable.destination == destination]
        )
        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(update(SessionQueueTable).where(*where).values(status="canceled"))

        if current_queue_item is not None and current_queue_item.destination == destination:
            if user_id is None or current_queue_item.user_id == user_id:
                self._set_queue_item_status(current_queue_item.item_id, "canceled")

        return CancelByDestinationResult(canceled=int(count))

    def cancel_by_queue_id(self, queue_id: str) -> CancelByQueueIDResult:
        current_queue_item = self.get_current(queue_id)
        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.notin_(("canceled", "completed", "failed", "in_progress")),
        ]
        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(update(SessionQueueTable).where(*where).values(status="canceled"))

        if current_queue_item is not None and current_queue_item.queue_id == queue_id:
            self._set_queue_item_status(current_queue_item.item_id, "canceled")
        return CancelByQueueIDResult(canceled=int(count))

    def cancel_all_except_current(
        self, queue_id: str, user_id: Optional[str] = None
    ) -> CancelAllExceptCurrentResult:
        where = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status == "pending",
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)

        with self._db.get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SessionQueueTable).where(*where)
            ).scalar_one()
            session.execute(update(SessionQueueTable).where(*where).values(status="canceled"))
        return CancelAllExceptCurrentResult(canceled=int(count))

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
        # NOTE: this preserves the (somewhat surprising) cursor semantics of the original
        # raw-SQL implementation, including the unparenthesised `AND ... OR ...` precedence.
        item_id = cursor

        stmt = select(*_QUEUE_COLUMNS, SessionQueueTable.workflow).where(
            SessionQueueTable.queue_id == queue_id
        )
        if status is not None:
            stmt = stmt.where(SessionQueueTable.status == status)
        if destination is not None:
            stmt = stmt.where(SessionQueueTable.destination == destination)
        if item_id is not None:
            stmt = stmt.where(
                or_(
                    SessionQueueTable.priority < priority,
                    and_(
                        SessionQueueTable.priority == priority,
                        SessionQueueTable.item_id > item_id,
                    ),
                )
            )
        stmt = stmt.order_by(
            SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc()
        ).limit(limit + 1)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        items = [SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(r)) for r in rows]
        has_more = False
        if len(items) > limit:
            items.pop()
            has_more = True
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def list_all_queue_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        stmt = _select_queue_item_with_user().where(SessionQueueTable.queue_id == queue_id)
        if destination is not None:
            stmt = stmt.where(SessionQueueTable.destination == destination)
        stmt = stmt.order_by(
            SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc()
        )
        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()
        return [SessionQueueItem.queue_item_from_dict(_row_to_queue_item_dict(r)) for r in rows]

    def get_queue_item_ids(
        self,
        queue_id: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        user_id: Optional[str] = None,
    ) -> ItemIdsResult:
        stmt = select(SessionQueueTable.item_id).where(SessionQueueTable.queue_id == queue_id)
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)
        if order_dir == SQLiteDirection.Descending:
            stmt = stmt.order_by(SessionQueueTable.created_at.desc())
        else:
            stmt = stmt.order_by(SessionQueueTable.created_at.asc())

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()
        item_ids = [row[0] for row in rows]
        return ItemIdsResult(item_ids=item_ids, total_count=len(item_ids))

    # endregion

    # region: aggregations

    def get_queue_status(self, queue_id: str, user_id: Optional[str] = None) -> SessionQueueStatus:
        stmt = (
            select(SessionQueueTable.status, func.count())
            .where(SessionQueueTable.queue_id == queue_id)
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        current_item = self.get_current(queue_id=queue_id)
        total = sum(int(row[1] or 0) for row in rows)
        counts: dict[str, int] = {row[0]: int(row[1]) for row in rows}

        # For non-admin users, hide current item details if they don't own it
        show_current_item = current_item is not None and (
            user_id is None or current_item.user_id == user_id
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

    def get_batch_status(
        self, queue_id: str, batch_id: str, user_id: Optional[str] = None
    ) -> BatchStatus:
        stmt = (
            select(
                SessionQueueTable.status,
                func.count(),
                SessionQueueTable.origin,
                SessionQueueTable.destination,
            )
            .where(
                SessionQueueTable.queue_id == queue_id,
                SessionQueueTable.batch_id == batch_id,
            )
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        total = sum(int(row[1] or 0) for row in rows)
        counts: dict[str, int] = {row[0]: int(row[1]) for row in rows}
        origin = rows[0][2] if rows else None
        destination = rows[0][3] if rows else None

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

    def get_counts_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> SessionQueueCountsByDestination:
        stmt = (
            select(SessionQueueTable.status, func.count())
            .where(
                SessionQueueTable.queue_id == queue_id,
                SessionQueueTable.destination == destination,
            )
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        total = sum(int(row[1] or 0) for row in rows)
        counts: dict[str, int] = {row[0]: int(row[1]) for row in rows}

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
                json.dumps(queue_item.field_values, default=to_jsonable_python)
                if queue_item.field_values
                else None
            )
            workflow_json = (
                json.dumps(queue_item.workflow, default=to_jsonable_python)
                if queue_item.workflow
                else None
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
        if values_to_insert:
            with self._db.get_session() as session:
                session.execute(
                    insert(SessionQueueTable),
                    [_value_tuple_to_dict(v) for v in values_to_insert],
                )

        retry_result = RetryItemsResult(queue_id=queue_id, retried_item_ids=retried_item_ids)
        self.__invoker.services.events.emit_queue_items_retried(retry_result)
        return retry_result

    # endregion
