"""Tests for workflow-call relationship metadata on session_queue items."""

import uuid

import pytest

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def test_get_queue_item_round_trips_workflow_call_metadata(session_queue: SqliteSessionQueue) -> None:
    session = GraphExecutionState(graph=Graph())
    session_json = session.model_dump_json(warnings=False)

    with session_queue._db.transaction() as cursor:
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
                workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                session_json,
                session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                "user-1",
                "workflow-call-1",
                11,
                "parent-session-1",
                7,
                3,
            ),
        )
        item_id = cursor.lastrowid

    queue_item = session_queue.get_queue_item(item_id)

    assert queue_item.workflow_call_id == "workflow-call-1"
    assert queue_item.parent_item_id == 11
    assert queue_item.parent_session_id == "parent-session-1"
    assert queue_item.root_item_id == 7
    assert queue_item.workflow_call_depth == 3


def test_enqueue_workflow_call_child_persists_pending_child_queue_item(session_queue: SqliteSessionQueue) -> None:
    parent_graph = Graph()
    parent_graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    parent_session = GraphExecutionState(graph=parent_graph)
    invocation = parent_session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)

    frame = parent_session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
    child_session = parent_session.create_child_workflow_execution_state(Graph(), frame)
    parent_session.begin_waiting_on_workflow_call(frame)
    parent_session.attach_waiting_workflow_call_child_session(child_session)

    with session_queue._db.transaction() as cursor:
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
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                parent_session.model_dump_json(warnings=False),
                parent_session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                "user-1",
                "in_progress",
            ),
        )
        parent_item_id = cursor.lastrowid

    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    child_queue_item = session_queue.enqueue_workflow_call_child(parent_queue_item, child_session)

    assert child_queue_item.status == "pending"
    assert child_queue_item.workflow_call_id == parent_session.waiting_workflow_call_execution.id
    assert child_queue_item.parent_item_id == parent_item_id
    assert child_queue_item.parent_session_id == parent_session.id
    assert child_queue_item.root_item_id == parent_item_id
    assert child_queue_item.workflow_call_depth == 1
    assert child_queue_item.session_id == child_session.id


def test_get_queue_status_counts_waiting_items(session_queue: SqliteSessionQueue) -> None:
    session = GraphExecutionState(graph=Graph())
    session_json = session.model_dump_json(warnings=False)

    with session_queue._db.transaction() as cursor:
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
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                session_json,
                session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                "user-1",
                "waiting",
            ),
        )

    queue_status = session_queue.get_queue_status("default")

    assert queue_status.waiting == 1
    assert queue_status.total == 1


def test_cancel_queue_item_cascades_from_waiting_parent_to_child_chain(session_queue: SqliteSessionQueue) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())
    grandchild_session = GraphExecutionState(graph=Graph())

    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                parent_session.model_dump_json(warnings=False),
                parent_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "waiting",
            ),
        )
        parent_item_id = cursor.lastrowid
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status,
                workflow_call_id, parent_item_id, parent_session_id, root_item_id, workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                child_session.model_dump_json(warnings=False),
                child_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "waiting",
                "workflow-call-1",
                parent_item_id,
                parent_session.id,
                parent_item_id,
                1,
            ),
        )
        child_item_id = cursor.lastrowid
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status,
                workflow_call_id, parent_item_id, parent_session_id, root_item_id, workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                grandchild_session.model_dump_json(warnings=False),
                grandchild_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "pending",
                "workflow-call-2",
                child_item_id,
                child_session.id,
                parent_item_id,
                2,
            ),
        )
        grandchild_item_id = cursor.lastrowid

    session_queue.cancel_queue_item(parent_item_id)

    assert session_queue.get_queue_item(parent_item_id).status == "canceled"
    assert session_queue.get_queue_item(child_item_id).status == "canceled"
    assert session_queue.get_queue_item(grandchild_item_id).status == "canceled"


def test_cancel_queue_item_cascades_from_child_to_waiting_parents(session_queue: SqliteSessionQueue) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                parent_session.model_dump_json(warnings=False),
                parent_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "waiting",
            ),
        )
        parent_item_id = cursor.lastrowid
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status,
                workflow_call_id, parent_item_id, parent_session_id, root_item_id, workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                child_session.model_dump_json(warnings=False),
                child_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "pending",
                "workflow-call-1",
                parent_item_id,
                parent_session.id,
                parent_item_id,
                1,
            ),
        )
        child_item_id = cursor.lastrowid

    session_queue.cancel_queue_item(child_item_id)

    assert session_queue.get_queue_item(child_item_id).status == "canceled"
    assert session_queue.get_queue_item(parent_item_id).status == "canceled"


def test_retry_items_by_id_retries_root_once_for_child_chain_item(session_queue: SqliteSessionQueue) -> None:
    root_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                root_session.model_dump_json(warnings=False),
                root_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "failed",
            ),
        )
        root_item_id = cursor.lastrowid
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, priority, user_id, status,
                workflow_call_id, parent_item_id, parent_session_id, root_item_id, workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                child_session.model_dump_json(warnings=False),
                child_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "failed",
                "workflow-call-1",
                root_item_id,
                root_session.id,
                root_item_id,
                1,
            ),
        )
        child_item_id = cursor.lastrowid

    retry_result = session_queue.retry_items_by_id("default", [child_item_id, root_item_id])

    assert retry_result.retried_item_ids == [root_item_id]

    all_items = session_queue.list_all_queue_items("default")
    retried_items = [item for item in all_items if item.retried_from_item_id == root_item_id]
    assert len(retried_items) == 1
    assert retried_items[0].status == "pending"
    assert retried_items[0].workflow_call_id is None
    assert retried_items[0].parent_item_id is None
    assert retried_items[0].root_item_id is None
