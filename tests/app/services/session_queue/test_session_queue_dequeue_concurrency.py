"""Tests that concurrent dequeue() calls (multi-GPU session workers) never claim the same item twice."""

import threading
import uuid

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import PromptTestInvocation


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(session_queue: SqliteSessionQueue, user_id: str = "system") -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values, priority,
                workflow, origin, destination, retried_from_item_id, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("default", session_json, session.id, batch_id, None, 0, None, None, None, None, user_id),
        )
        return cursor.lastrowid


def test_concurrent_dequeue_never_claims_same_item_twice(session_queue: SqliteSessionQueue) -> None:
    item_count = 50
    worker_count = 8
    for _ in range(item_count):
        _insert_queue_item(session_queue)

    claimed_ids: list[int] = []
    claimed_lock = threading.Lock()
    start_barrier = threading.Barrier(worker_count)

    def worker() -> None:
        # Release all workers at once to maximize contention on the dequeue path.
        start_barrier.wait()
        while True:
            item = session_queue.dequeue()
            if item is None:
                break
            with claimed_lock:
                claimed_ids.append(item.item_id)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every item is claimed exactly once: no duplicates, none lost.
    assert len(claimed_ids) == item_count
    assert len(set(claimed_ids)) == item_count
