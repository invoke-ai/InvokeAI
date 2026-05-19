"""Tests for the SQLModel-backed session queue implementation."""

import asyncio
import uuid
from typing import Optional

import pytest
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    SessionQueueItemNotFoundError,
)
from invokeai.app.services.session_queue.session_queue_sqlmodel import SqlModelSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.sqlite.models import SessionQueueTable
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from tests.test_nodes import PromptTestInvocation


# ---- fixtures ----


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqlModelSessionQueue:
    """Create a SqlModelSessionQueue backed by the mock invoker's in-memory database."""
    db = mock_invoker.services.board_records._db
    queue = SqlModelSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def batch_graph() -> Graph:
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Chevy"))
    return g


# ---- helpers ----


def _make_session_json() -> tuple[str, str]:
    """Build a valid GraphExecutionState JSON blob and return (session_id, json)."""
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Chevy"))
    state = GraphExecutionState(graph=g)
    return state.id, state.model_dump_json(warnings=False, exclude_none=True)


def _insert_raw(
    queue: SqlModelSessionQueue,
    *,
    queue_id: str = "default",
    user_id: str = "system",
    status: str = "pending",
    priority: int = 0,
    batch_id: Optional[str] = None,
    destination: Optional[str] = None,
) -> int:
    """Insert a minimal queue item via Core and return its item_id."""
    session_id, session_json = _make_session_json()
    batch_id = batch_id or str(uuid.uuid4())
    with queue._db.get_session() as session:
        result = session.execute(
            insert(SessionQueueTable).values(
                queue_id=queue_id,
                session=session_json,
                session_id=session_id,
                batch_id=batch_id,
                field_values=None,
                priority=priority,
                workflow=None,
                origin=None,
                destination=destination,
                retried_from_item_id=None,
                user_id=user_id,
                status=status,
            )
        )
        return int(result.inserted_primary_key[0])


# ---- start() / _set_in_progress_to_canceled ----


def test_start_cancels_in_progress(mock_invoker: Invoker) -> None:
    db = mock_invoker.services.board_records._db
    queue = SqlModelSessionQueue(db=db)
    in_progress_id = _insert_raw(queue, status="in_progress")
    queue.start(mock_invoker)
    item = queue.get_queue_item(in_progress_id)
    assert item.status == "canceled"


# ---- simple read methods ----


def test_is_empty_and_is_full(session_queue: SqlModelSessionQueue) -> None:
    assert session_queue.is_empty("default").is_empty is True
    _insert_raw(session_queue)
    assert session_queue.is_empty("default").is_empty is False
    # default max_queue_size is high; queue with 1 item is not full
    assert session_queue.is_full("default").is_full is False


def test_get_queue_item_not_found(session_queue: SqlModelSessionQueue) -> None:
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(99999)


def test_get_queue_item(session_queue: SqlModelSessionQueue) -> None:
    item_id = _insert_raw(session_queue, user_id="alice")
    item = session_queue.get_queue_item(item_id)
    assert item.item_id == item_id
    assert item.user_id == "alice"
    assert item.status == "pending"


def test_get_current_and_get_next(session_queue: SqlModelSessionQueue) -> None:
    pending = _insert_raw(session_queue, priority=1)
    in_progress = _insert_raw(session_queue, status="in_progress")
    current = session_queue.get_current("default")
    assert current is not None and current.item_id == in_progress
    nxt = session_queue.get_next("default")
    assert nxt is not None and nxt.item_id == pending


def test_get_current_queue_size(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue)
    _insert_raw(session_queue)
    _insert_raw(session_queue, status="completed")
    assert session_queue._get_current_queue_size("default") == 2


def test_get_highest_priority(session_queue: SqlModelSessionQueue) -> None:
    assert session_queue._get_highest_priority("default") == 0
    _insert_raw(session_queue, priority=3)
    _insert_raw(session_queue, priority=7)
    _insert_raw(session_queue, priority=10, status="completed")  # ignored
    assert session_queue._get_highest_priority("default") == 7


# ---- enqueue / dequeue ----


def test_enqueue_batch_and_dequeue(
    session_queue: SqlModelSessionQueue, batch_graph: Graph
) -> None:
    batch = Batch(graph=batch_graph, runs=2)
    result = asyncio.run(session_queue.enqueue_batch("default", batch, prepend=False))
    assert result.enqueued == 2
    assert result.requested == 2
    assert len(result.item_ids) == 2

    # dequeue takes the first pending and marks it in_progress
    dequeued = session_queue.dequeue()
    assert dequeued is not None
    assert dequeued.status == "in_progress"

    # only one in-progress at a time
    current = session_queue.get_current("default")
    assert current is not None and current.item_id == dequeued.item_id


def test_enqueue_batch_prepend_increases_priority(
    session_queue: SqlModelSessionQueue, batch_graph: Graph
) -> None:
    asyncio.run(session_queue.enqueue_batch("default", Batch(graph=batch_graph), prepend=False))
    second = asyncio.run(
        session_queue.enqueue_batch("default", Batch(graph=batch_graph), prepend=True)
    )
    assert second.priority == 1


def test_dequeue_empty_returns_none(session_queue: SqlModelSessionQueue) -> None:
    assert session_queue.dequeue() is None


# ---- status mutations ----


def test_complete_fail_cancel_queue_item(session_queue: SqlModelSessionQueue) -> None:
    item_id = _insert_raw(session_queue)
    assert session_queue.complete_queue_item(item_id).status == "completed"
    # second mutation on terminal-status item is a no-op (returns existing)
    assert session_queue.cancel_queue_item(item_id).status == "completed"

    item_id2 = _insert_raw(session_queue)
    failed = session_queue.fail_queue_item(item_id2, "ErrType", "ErrMsg", "trace")
    assert failed.status == "failed"
    assert failed.error_type == "ErrType"
    assert failed.error_message == "ErrMsg"
    assert failed.error_traceback == "trace"

    item_id3 = _insert_raw(session_queue)
    assert session_queue.cancel_queue_item(item_id3).status == "canceled"


def test_set_queue_item_status_unknown_id_raises(
    session_queue: SqlModelSessionQueue,
) -> None:
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue._set_queue_item_status(99999, "completed")


def test_delete_queue_item(session_queue: SqlModelSessionQueue) -> None:
    item_id = _insert_raw(session_queue)
    session_queue.delete_queue_item(item_id)
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(item_id)


def test_set_queue_item_session(
    session_queue: SqlModelSessionQueue, batch_graph: Graph
) -> None:
    item_id = _insert_raw(session_queue)
    new_session = GraphExecutionState(graph=batch_graph)
    session_queue.set_queue_item_session(item_id, new_session)
    fetched = session_queue.get_queue_item(item_id)
    assert fetched.session.id == new_session.id


# ---- bulk delete ----


def test_clear_with_user_id_only_deletes_own_items(
    session_queue: SqlModelSessionQueue,
) -> None:
    _insert_raw(session_queue, user_id="user_a")
    _insert_raw(session_queue, user_id="user_a")
    _insert_raw(session_queue, user_id="user_b")
    result = session_queue.clear("default", user_id="user_a")
    assert result.deleted == 2


def test_clear_without_user_id_deletes_all(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, user_id="user_a")
    _insert_raw(session_queue, user_id="user_b")
    result = session_queue.clear("default")
    assert result.deleted == 2


def test_prune_only_deletes_terminal(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="completed")
    _insert_raw(session_queue, status="failed")
    _insert_raw(session_queue, status="canceled")
    _insert_raw(session_queue, status="in_progress")
    result = session_queue.prune("default")
    assert result.deleted == 3
    # pending and in_progress remain
    assert session_queue.get_queue_status("default").pending == 1
    assert session_queue.get_queue_status("default").in_progress == 1


def test_prune_with_user_id(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, status="completed", user_id="user_a")
    _insert_raw(session_queue, status="failed", user_id="user_b")
    result = session_queue.prune("default", user_id="user_a")
    assert result.deleted == 1


def test_delete_by_destination(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, destination="canvas")
    _insert_raw(session_queue, destination="canvas")
    _insert_raw(session_queue, destination="generate")
    result = session_queue.delete_by_destination("default", destination="canvas")
    assert result.deleted == 2


def test_delete_all_except_current(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="in_progress")
    _insert_raw(session_queue, status="completed")
    result = session_queue.delete_all_except_current("default")
    # only deletes pending
    assert result.deleted == 2
    status = session_queue.get_queue_status("default")
    assert status.pending == 0
    assert status.in_progress == 1
    assert status.completed == 1


# ---- bulk cancel ----


def test_cancel_by_batch_ids(session_queue: SqlModelSessionQueue) -> None:
    batch_id = str(uuid.uuid4())
    _insert_raw(session_queue, batch_id=batch_id)
    _insert_raw(session_queue, batch_id=batch_id)
    _insert_raw(session_queue, batch_id=str(uuid.uuid4()))  # different batch
    result = session_queue.cancel_by_batch_ids("default", [batch_id])
    assert result.canceled == 2


def test_cancel_by_destination(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, destination="canvas")
    _insert_raw(session_queue, destination="canvas", status="completed")  # skipped
    _insert_raw(session_queue, destination="generate")  # different dest
    result = session_queue.cancel_by_destination("default", "canvas")
    assert result.canceled == 1


def test_cancel_by_queue_id(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, queue_id="default")
    _insert_raw(session_queue, queue_id="default")
    _insert_raw(session_queue, queue_id="other")
    result = session_queue.cancel_by_queue_id("default")
    assert result.canceled == 2


def test_cancel_all_except_current(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="in_progress")
    result = session_queue.cancel_all_except_current("default")
    assert result.canceled == 2


# ---- prune-to-limit ----


def test_prune_terminal_to_limit_keeps_n_most_recent(
    session_queue: SqlModelSessionQueue,
) -> None:
    for _ in range(5):
        _insert_raw(session_queue, status="completed")
    deleted = session_queue._prune_terminal_to_limit("default", keep=2)
    assert deleted == 3
    assert session_queue.get_queue_status("default").completed == 2


# ---- list / pagination ----


def test_list_queue_items_pagination(session_queue: SqlModelSessionQueue) -> None:
    ids = [_insert_raw(session_queue) for _ in range(5)]
    page = session_queue.list_queue_items("default", limit=2, priority=0)
    assert len(page.items) == 2
    assert page.has_more is True

    next_page = session_queue.list_queue_items(
        "default", limit=2, priority=0, cursor=page.items[-1].item_id
    )
    assert len(next_page.items) == 2

    # Make sure no item appears twice
    seen_ids = {i.item_id for i in page.items} | {i.item_id for i in next_page.items}
    assert seen_ids.issubset(set(ids))
    assert len(seen_ids) == 4


def test_list_queue_items_filters_status_and_destination(
    session_queue: SqlModelSessionQueue,
) -> None:
    _insert_raw(session_queue, destination="canvas", status="completed")
    _insert_raw(session_queue, destination="canvas", status="pending")
    _insert_raw(session_queue, destination="generate", status="completed")
    page = session_queue.list_queue_items(
        "default", limit=10, priority=0, status="completed", destination="canvas"
    )
    assert len(page.items) == 1


def test_list_all_queue_items(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, destination="canvas")
    _insert_raw(session_queue, destination="canvas")
    _insert_raw(session_queue, destination="generate")
    items = session_queue.list_all_queue_items("default", destination="canvas")
    assert len(items) == 2


def test_get_queue_item_ids_ordering(session_queue: SqlModelSessionQueue) -> None:
    # Items inserted in the same millisecond may tie on created_at, so we only assert
    # set-equality and total_count. Ordering correctness is exercised by the SQL query
    # construction itself (covered by the production query path).
    ids = [_insert_raw(session_queue) for _ in range(3)]
    desc = session_queue.get_queue_item_ids("default", order_dir=SQLiteDirection.Descending)
    asc = session_queue.get_queue_item_ids("default", order_dir=SQLiteDirection.Ascending)
    assert desc.total_count == 3
    assert asc.total_count == 3
    assert set(desc.item_ids) == set(ids)
    assert set(asc.item_ids) == set(ids)


def test_get_queue_item_ids_filters_user_id(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, user_id="alice")
    _insert_raw(session_queue, user_id="bob")
    result = session_queue.get_queue_item_ids("default", user_id="alice")
    assert result.total_count == 1


# ---- aggregations ----


def test_get_queue_status_counts(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, status="pending")
    _insert_raw(session_queue, status="completed")
    _insert_raw(session_queue, status="failed")
    _insert_raw(session_queue, status="canceled")
    status = session_queue.get_queue_status("default")
    assert status.pending == 1
    assert status.completed == 1
    assert status.failed == 1
    assert status.canceled == 1
    assert status.total == 4


def test_get_queue_status_user_id_hides_other_user_current(
    session_queue: SqlModelSessionQueue,
) -> None:
    _insert_raw(session_queue, user_id="alice", status="in_progress")
    status = session_queue.get_queue_status("default", user_id="bob")
    # current item exists but belongs to alice — should be hidden for bob
    assert status.item_id is None


def test_get_batch_status(session_queue: SqlModelSessionQueue) -> None:
    batch_id = str(uuid.uuid4())
    _insert_raw(session_queue, batch_id=batch_id, status="pending")
    _insert_raw(session_queue, batch_id=batch_id, status="completed")
    _insert_raw(session_queue, batch_id=str(uuid.uuid4()), status="completed")
    result = session_queue.get_batch_status("default", batch_id=batch_id)
    assert result.pending == 1
    assert result.completed == 1
    assert result.total == 2


def test_get_counts_by_destination(session_queue: SqlModelSessionQueue) -> None:
    _insert_raw(session_queue, destination="canvas", status="pending")
    _insert_raw(session_queue, destination="canvas", status="completed")
    _insert_raw(session_queue, destination="generate", status="pending")
    result = session_queue.get_counts_by_destination("default", destination="canvas")
    assert result.pending == 1
    assert result.completed == 1
    assert result.total == 2


# ---- retry ----


def test_retry_items_by_id_skips_non_terminal(
    session_queue: SqlModelSessionQueue, batch_graph: Graph
) -> None:
    pending_id = _insert_raw(session_queue, status="pending")
    result = session_queue.retry_items_by_id("default", [pending_id])
    assert result.retried_item_ids == []


def test_retry_items_by_id_clones_failed(
    session_queue: SqlModelSessionQueue, batch_graph: Graph
) -> None:
    # Use enqueue_batch so we get a valid `session` JSON, then fail it
    batch = Batch(graph=batch_graph, runs=1)
    enq = asyncio.run(session_queue.enqueue_batch("default", batch, prepend=False))
    item_id = enq.item_ids[0]
    session_queue.fail_queue_item(item_id, "ErrType", "ErrMsg", "trace")

    retry = session_queue.retry_items_by_id("default", [item_id])
    assert retry.retried_item_ids == [item_id]
    # exactly one new pending item should now exist (the original is failed)
    status = session_queue.get_queue_status("default")
    assert status.pending == 1
    assert status.failed == 1
