"""Tests for workflow-call relationship metadata on session_queue items."""

import uuid
from pathlib import Path
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.events.events_common import (
    InvocationStartedEvent,
    QueueItemsRetriedEvent,
    QueueItemStatusChangedEvent,
)
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.session_queue.session_queue_common import (
    NodeFieldValue,
    SessionQueueItemNotFoundError,
    TooManySessionsError,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import ImagesInterface, InvocationContextData
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowMeta, WorkflowWithoutID
from tests.test_nodes import TestEventService


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def event_bus(mock_invoker: Invoker) -> TestEventService:
    assert isinstance(mock_invoker.services.events, TestEventService)
    return mock_invoker.services.events


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    *,
    session: GraphExecutionState,
    status: str,
    queue_id: str = "default",
    batch_id: str | None = None,
    destination: str | None = None,
    user_id: str = "user-1",
    project_id: str | None = None,
    workflow_call_id: str | None = None,
    parent_item_id: int | None = None,
    parent_session_id: str | None = None,
    root_item_id: int | None = None,
    workflow_call_depth: int | None = None,
    workflow: WorkflowWithoutID | None = None,
) -> int:
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
                project_id,
                workflow_call_id,
                parent_item_id,
                parent_session_id,
                root_item_id,
                workflow_call_depth,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                session.model_dump_json(warnings=False),
                session.id,
                batch_id or str(uuid.uuid4()),
                None,
                0,
                workflow.model_dump_json() if workflow else None,
                None,
                destination,
                None,
                user_id,
                project_id,
                workflow_call_id,
                parent_item_id,
                parent_session_id,
                root_item_id,
                workflow_call_depth,
                status,
            ),
        )
        return cursor.lastrowid


def _workflow_without_id() -> WorkflowWithoutID:
    return WorkflowWithoutID(
        name="Parent workflow",
        author="Tester",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="1.0.0", category="user"),
        nodes=[],
        edges=[],
        form=None,
    )


def _build_waiting_workflow_call_parent(
    session_queue: SqliteSessionQueue, child_count: int, project_id: str | None = None
) -> tuple[int, GraphExecutionState, list[GraphExecutionState]]:
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    parent_session = GraphExecutionState(graph=graph)
    invocation = parent_session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    frame = parent_session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
    parent_session.begin_waiting_on_workflow_call(frame)
    child_sessions = [parent_session.create_child_workflow_execution_state(Graph(), frame) for _ in range(child_count)]
    parent_session.attach_waiting_workflow_call_child_sessions(child_sessions)
    parent_item_id = _insert_queue_item(
        session_queue, session=parent_session, status="in_progress", project_id=project_id
    )
    return parent_item_id, parent_session, child_sessions


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


def test_save_queue_item_session_does_not_reload_full_queue_item(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = GraphExecutionState(graph=Graph())
    item_id = _insert_queue_item(session_queue, session=session, status="pending")
    session.errors["node"] = "updated"

    def fail_get_queue_item(item_id: int):
        raise AssertionError(f"Unexpected full queue item reload for {item_id}")

    monkeypatch.setattr(session_queue, "get_queue_item", fail_get_queue_item)

    session_queue.save_queue_item_session(item_id, session)

    with session_queue._db.transaction() as cursor:
        cursor.execute("SELECT session FROM session_queue WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()
    assert row is not None
    persisted = GraphExecutionState.model_validate_json(row[0])
    assert persisted.errors == {"node": "updated"}


def test_active_save_queue_item_session_does_not_overwrite_terminal_item(session_queue: SqliteSessionQueue) -> None:
    session = GraphExecutionState(graph=Graph())
    item_id = _insert_queue_item(session_queue, session=session, status="pending")
    stale_session = session_queue.get_queue_item(item_id).session
    stale_session.errors["stale"] = "parent completion"

    session_queue.cancel_queue_item(item_id)

    assert session_queue._save_queue_item_session_if_active(item_id, stale_session) is False
    persisted = session_queue.get_queue_item(item_id)
    assert persisted.status == "canceled"
    assert persisted.session.errors == {}


def test_failed_child_transitions_sqlite_parent_to_failed(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    parent_item_id, _parent_session, _child_sessions = _build_waiting_workflow_call_parent(session_queue, child_count=1)
    mock_invoker.services.session_queue = session_queue
    runner = DefaultSessionRunner()
    runner.start(mock_invoker.services, Event())
    child_queue_item = SimpleNamespace(item_id=999, parent_item_id=parent_item_id, error_message="child failed")

    runner.workflow_call_queue_lifecycle._fail_parent_from_failed_child(child_queue_item)

    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    assert parent_queue_item.status == "failed"
    assert parent_queue_item.error_message == "child failed"
    assert list(parent_queue_item.session.errors.values()) == ["ValueError: child failed"]


def test_enqueue_workflow_call_children_rejects_stale_parent_session(
    session_queue: SqliteSessionQueue,
) -> None:
    parent_item_id, _parent_session, child_sessions = _build_waiting_workflow_call_parent(session_queue, child_count=2)
    stale_parent = session_queue.get_queue_item(parent_item_id)
    current_parent = session_queue.get_queue_item(parent_item_id)
    current_parent.session.errors["sibling"] = "newer parent state"
    session_queue.save_queue_item_session(parent_item_id, current_parent.session)

    with pytest.raises(ValueError, match="changed while enqueuing"):
        session_queue.enqueue_workflow_call_children(
            parent_queue_item=stale_parent,
            child_sessions=[(child_session, None) for child_session in child_sessions],
        )

    persisted_parent = session_queue.get_queue_item(parent_item_id)
    assert persisted_parent.status == "in_progress"
    assert persisted_parent.session.errors == {"sibling": "newer parent state"}
    assert [item.item_id for item in session_queue.list_all_queue_items("default")] == [parent_item_id]
    with session_queue._db.transaction() as cursor:
        cursor.execute("SELECT COUNT(*) FROM session_queue")
        assert cursor.fetchone()[0] == 1


def test_status_transition_reuses_loaded_queue_item(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = GraphExecutionState(graph=Graph())
    item_id = _insert_queue_item(session_queue, session=session, status="in_progress")
    queue_item = session_queue.get_queue_item(item_id)

    def fail_get_queue_item(item_id: int):
        raise AssertionError(f"Unexpected full queue item reload for {item_id}")

    monkeypatch.setattr(session_queue, "get_queue_item", fail_get_queue_item)

    updated = session_queue.suspend_queue_item(item_id, queue_item=queue_item)

    assert updated is queue_item
    assert updated.status == "waiting"
    assert updated.status_sequence == 1


def test_enqueue_workflow_call_child_inherits_workflow_for_image_metadata(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
                project_id,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                parent_session.model_dump_json(warnings=False),
                parent_session.id,
                str(uuid.uuid4()),
                None,
                0,
                _workflow_without_id().model_dump_json(),
                None,
                None,
                None,
                "user-1",
                "project-1",
                "in_progress",
            ),
        )
        parent_item_id = cursor.lastrowid

    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    child_queue_item = session_queue.enqueue_workflow_call_child(parent_queue_item, child_session)

    assert child_queue_item.status == "pending"
    # Outputs of a child workflow belong to the project that enqueued the parent.
    assert child_queue_item.project_id == "project-1"
    assert child_queue_item.workflow_call_id == parent_session.waiting_workflow_call_execution.id
    assert child_queue_item.parent_item_id == parent_item_id
    assert child_queue_item.parent_session_id == parent_session.id
    assert child_queue_item.root_item_id == parent_item_id
    assert child_queue_item.workflow_call_depth == 1
    assert child_queue_item.session_id == child_session.id
    assert child_queue_item.workflow is None

    image_files = DiskImageFileStorage(tmp_path / "images")
    mock_invoker.services.session_queue = session_queue
    mock_invoker.services.image_files = image_files
    mock_invoker.services.names = SimpleNameService()
    mock_invoker.services.urls = LocalUrlService()
    image_files.start(mock_invoker)
    mock_invoker.services.images.start(mock_invoker)

    workflow_json = _workflow_without_id().model_dump_json()
    workflow_lookup_count = 0
    original_workflow_lookup = session_queue.get_queue_item_workflow_json

    def count_workflow_lookups(item_id: int) -> str | None:
        nonlocal workflow_lookup_count
        workflow_lookup_count += 1
        return original_workflow_lookup(item_id)

    monkeypatch.setattr(session_queue, "get_queue_item_workflow_json", count_workflow_lookups)

    image_names = []
    for index in range(3):
        images = ImagesInterface(
            mock_invoker.services,
            InvocationContextData(
                queue_item=child_queue_item,
                invocation=MagicMock(is_intermediate=False, id=f"image-node-{index}"),
                source_invocation_id=f"image-node-{index}",
            ),
            MagicMock(),
        )
        dto = images.save(Image.new("RGB", (4, 4)))
        image_names.append(dto.image_name)

    assert workflow_lookup_count == 1
    for image_name in image_names:
        with Image.open(image_files.get_path(image_name)) as saved:
            assert saved.info["invokeai_workflow"] == workflow_json
        assert mock_invoker.services.images.get_workflow(image_name) == workflow_json


def test_enqueue_workflow_call_child_rejects_canceled_stale_parent(
    session_queue: SqliteSessionQueue,
) -> None:
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
                queue_id, session, session_id, batch_id, field_values, priority,
                workflow, origin, destination, retried_from_item_id, user_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    stale_parent = session_queue.get_queue_item(parent_item_id)
    session_queue.cancel_queue_item(parent_item_id)

    with pytest.raises(ValueError, match="terminal parent"):
        session_queue.enqueue_workflow_call_child(stale_parent, child_session)

    with session_queue._db.transaction() as cursor:
        cursor.execute("SELECT COUNT(*) FROM session_queue WHERE parent_item_id = ?", (parent_item_id,))
        assert cursor.fetchone()[0] == 0


def test_enqueue_workflow_call_children_publishes_parent_state_before_children(
    session_queue: SqliteSessionQueue,
) -> None:
    parent_item_id, _parent_session, child_sessions = _build_waiting_workflow_call_parent(
        session_queue, child_count=2, project_id="project-1"
    )

    child_queue_items = session_queue.enqueue_workflow_call_children(
        parent_queue_item=session_queue.get_queue_item(parent_item_id),
        child_sessions=[(child_session, None) for child_session in child_sessions],
    )

    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    assert parent_queue_item.status == "waiting"
    assert [child_queue_item.project_id for child_queue_item in child_queue_items] == ["project-1", "project-1"]
    assert parent_queue_item.session.waiting_workflow_call_execution is not None
    assert parent_queue_item.session.waiting_workflow_call_execution.child_item_ids == [
        child_queue_item.item_id for child_queue_item in child_queue_items
    ]
    assert [
        session_queue.get_queue_item(child_queue_item.item_id).status for child_queue_item in child_queue_items
    ] == [
        "pending",
        "pending",
    ]

    assert [session_queue.dequeue().item_id for _ in child_queue_items] == [
        child_queue_item.item_id for child_queue_item in child_queue_items
    ]


def test_concurrent_workflow_call_child_completions_preserve_both_siblings(
    session_queue: SqliteSessionQueue,
) -> None:
    parent_item_id, _parent_session, child_sessions = _build_waiting_workflow_call_parent(session_queue, child_count=2)
    child_queue_items = session_queue.enqueue_workflow_call_children(
        parent_queue_item=session_queue.get_queue_item(parent_item_id),
        child_sessions=[(child_session, None) for child_session in child_sessions],
    )
    for child_queue_item in child_queue_items:
        session_queue.complete_queue_item(child_queue_item.item_id)

    barrier = Barrier(len(child_queue_items))
    completions = []
    errors = []

    def record_completion(child_item_id: int, value: int) -> None:
        try:
            barrier.wait()
            completions.append(
                session_queue.record_workflow_call_child_completion(
                    parent_item_id=parent_item_id,
                    child_item_id=child_item_id,
                    output_values={"result": value},
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [
        Thread(target=record_completion, args=(child_queue_item.item_id, index))
        for index, child_queue_item in enumerate(child_queue_items, start=1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(completions) == len(child_queue_items)
    assert sum(completion is not None and completion.should_resume for completion in completions) == 1

    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    execution = parent_queue_item.session.waiting_workflow_call_execution
    assert execution is not None
    assert set(execution.completed_child_item_ids) == {
        child_queue_item.item_id for child_queue_item in child_queue_items
    }
    assert execution.child_outputs == {
        child_queue_items[0].item_id: {"result": 1},
        child_queue_items[1].item_id: {"result": 2},
    }


@pytest.mark.parametrize("limited", [False, True])
def test_history_pruning_retains_children_until_their_root_ends(session_queue, limited):
    root, _, child_sessions = _build_waiting_workflow_call_parent(session_queue, child_count=2)
    children = session_queue.enqueue_workflow_call_children(
        session_queue.get_queue_item(root), [(session, None) for session in child_sessions]
    )
    session_queue.complete_queue_item(children[0].item_id)
    old = _insert_queue_item(session_queue, session=GraphExecutionState(graph=Graph()), status="completed")

    def prune():
        if limited:
            return session_queue._prune_terminal_to_limit("default", keep=0)
        return session_queue.prune("default", user_id="user-1").deleted

    assert prune() == 1
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(old)
    assert session_queue.get_queue_item(children[0].item_id).status == "completed"

    session_queue.cancel_queue_item(root)
    assert prune() == 3


def test_enqueue_workflow_call_child_rejects_full_pending_queue(session_queue: SqliteSessionQueue) -> None:
    parent_graph = Graph()
    parent_graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    parent_session = GraphExecutionState(graph=parent_graph)
    invocation = parent_session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)

    frame = parent_session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
    child_session = parent_session.create_child_workflow_execution_state(Graph(), frame)
    parent_session.begin_waiting_on_workflow_call(frame)
    parent_session.attach_waiting_workflow_call_child_session(child_session)

    parent_item_id = _insert_queue_item(session_queue, session=parent_session, status="waiting")
    _insert_queue_item(session_queue, session=GraphExecutionState(graph=Graph()), status="pending")
    parent_queue_item = session_queue.get_queue_item(parent_item_id)
    session_queue._SqliteSessionQueue__invoker.services.configuration.max_queue_size = 1

    with pytest.raises(TooManySessionsError, match="remaining queue capacity"):
        session_queue.enqueue_workflow_call_child(parent_queue_item, child_session)

    assert session_queue.get_queue_status("default").pending == 1


def test_enqueue_workflow_call_child_preserves_workflow_through_nested_calls(
    session_queue: SqliteSessionQueue,
) -> None:
    workflow = _workflow_without_id()

    root_graph = Graph()
    root_graph.add_node(CallSavedWorkflowInvocation(id="root-call", workflow_id="workflow-a"))
    root_session = GraphExecutionState(graph=root_graph)
    root_invocation = root_session.next()
    assert isinstance(root_invocation, CallSavedWorkflowInvocation)
    root_frame = root_session.build_workflow_call_frame(root_invocation.id, root_invocation.workflow_id)

    child_graph = Graph()
    child_graph.add_node(CallSavedWorkflowInvocation(id="nested-call", workflow_id="workflow-b"))
    child_session = root_session.create_child_workflow_execution_state(child_graph, root_frame)
    root_session.begin_waiting_on_workflow_call(root_frame)
    root_session.attach_waiting_workflow_call_child_session(child_session)
    root_item_id = _insert_queue_item(session_queue, session=root_session, status="in_progress", workflow=workflow)

    root_queue_item = session_queue.get_queue_item(root_item_id)
    child_queue_item = session_queue.enqueue_workflow_call_child(root_queue_item, child_session)
    nested_invocation = child_session.next()
    assert isinstance(nested_invocation, CallSavedWorkflowInvocation)
    nested_frame = child_session.build_workflow_call_frame(nested_invocation.id, nested_invocation.workflow_id)
    grandchild_graph = Graph()
    grandchild_graph.add_node(CallSavedWorkflowInvocation(id="leaf-call", workflow_id="workflow-c"))
    grandchild_session = child_session.create_child_workflow_execution_state(grandchild_graph, nested_frame)
    child_session.begin_waiting_on_workflow_call(nested_frame)
    child_session.attach_waiting_workflow_call_child_session(grandchild_session)
    child_queue_item.session = child_session

    child_event = InvocationStartedEvent.build(child_queue_item, nested_invocation)
    assert child_event.root_item_id == root_item_id
    assert child_event.parent_item_id == root_item_id
    assert child_event.workflow_call_parent_source_id == "root-call"

    grandchild_queue_item = session_queue.enqueue_workflow_call_child(child_queue_item, grandchild_session)
    grandchild_invocation = grandchild_queue_item.session.next()
    assert grandchild_invocation is not None
    grandchild_event = InvocationStartedEvent.build(grandchild_queue_item, grandchild_invocation)
    assert grandchild_event.root_item_id == root_item_id
    assert grandchild_event.parent_item_id == child_queue_item.item_id
    assert grandchild_event.workflow_call_parent_source_id == "root-call"

    assert child_queue_item.workflow is None
    assert grandchild_queue_item.workflow is None
    assert session_queue.get_queue_item_workflow_json(root_item_id) == workflow.model_dump_json()


def test_enqueue_workflow_call_child_persists_batch_field_values(session_queue: SqliteSessionQueue) -> None:
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

    child_queue_item = session_queue.enqueue_workflow_call_child(
        parent_queue_item=session_queue.get_queue_item(parent_item_id),
        child_session=child_session,
        field_values=[NodeFieldValue(node_path="target", field_name="value", value=2)],
    )

    assert child_queue_item.field_values == [NodeFieldValue(node_path="target", field_name="value", value=2)]


def test_suspend_and_enqueue_child_emit_waiting_then_pending_status_events(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
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

    parent_queue_item = session_queue.suspend_queue_item(parent_item_id)
    child_queue_item = session_queue.enqueue_workflow_call_child(parent_queue_item, child_session)

    queue_events = [event for event in event_bus.events if isinstance(event, QueueItemStatusChangedEvent)]
    assert len(queue_events) == 2

    waiting_event = queue_events[0]
    assert waiting_event.item_id == parent_item_id
    assert waiting_event.status == "waiting"
    assert waiting_event.queue_status.waiting == 1
    assert waiting_event.queue_status.pending == 0
    assert waiting_event.queue_status.item_id is None

    child_pending_event = queue_events[1]
    assert child_pending_event.item_id == child_queue_item.item_id
    assert child_pending_event.status == "pending"
    assert child_pending_event.queue_status.waiting == 1
    assert child_pending_event.queue_status.pending == 1
    assert child_pending_event.queue_status.total == 2
    assert child_pending_event.queue_status.item_id is None


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


def test_startup_cancellation_cancels_waiting_workflow_call_chain(session_queue: SqliteSessionQueue) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())
    sibling_session = GraphExecutionState(graph=Graph())
    batch_id = str(uuid.uuid4())

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
                batch_id,
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
        parent_item_id = cursor.lastrowid
        for child_status, session in (("in_progress", child_session), ("pending", sibling_session)):
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
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "default",
                    session.model_dump_json(warnings=False),
                    session.id,
                    batch_id,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    "user-1",
                    "workflow-call-1",
                    parent_item_id,
                    parent_session.id,
                    parent_item_id,
                    1,
                    child_status,
                ),
            )

    session_queue._set_in_progress_to_canceled()

    assert session_queue.get_queue_item(parent_item_id).status == "canceled"
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            SELECT status
            FROM session_queue
            WHERE parent_item_id = ?
            ORDER BY item_id ASC
            """,
            (parent_item_id,),
        )
        child_statuses = [row[0] for row in cursor.fetchall()]
    assert child_statuses == ["canceled", "canceled"]


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


def test_cancel_child_uses_metadata_chain_walk_and_hydrates_each_item_once(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_item_id, parent_session, child_sessions = _build_waiting_workflow_call_parent(session_queue, child_count=1)
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_sessions[0],
        status="pending",
        workflow_call_id=parent_session.waiting_workflow_call_execution.id,  # type: ignore[union-attr]
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )

    original_hydrate = session_queue._hydrate_queue_item
    hydrate_calls = 0

    def count_hydration(raw_queue_item: dict, *, quarantine: bool):
        nonlocal hydrate_calls
        hydrate_calls += 1
        return original_hydrate(raw_queue_item, quarantine=quarantine)

    monkeypatch.setattr(session_queue, "_hydrate_queue_item", count_hydration)

    canceled = session_queue.cancel_queue_item(child_item_id)

    assert canceled.item_id == child_item_id
    assert canceled.status == "canceled"
    assert hydrate_calls == 2


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


def test_delete_queue_item_removes_entire_workflow_call_chain(session_queue: SqliteSessionQueue) -> None:
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
                "failed",
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
                "canceled",
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
                "canceled",
                "workflow-call-2",
                child_item_id,
                child_session.id,
                parent_item_id,
                2,
            ),
        )
        grandchild_item_id = cursor.lastrowid

    session_queue.delete_queue_item(child_item_id)

    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(parent_item_id)
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(child_item_id)
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(grandchild_item_id)


def test_delete_queue_item_cancels_active_workflow_call_chain_before_deleting(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
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
                "in_progress",
                "workflow-call-1",
                parent_item_id,
                parent_session.id,
                parent_item_id,
                1,
            ),
        )
        child_item_id = cursor.lastrowid

    session_queue.delete_queue_item(child_item_id)

    canceled_events = [event for event in event_bus.events if isinstance(event, QueueItemStatusChangedEvent)]
    assert [event.item_id for event in canceled_events if event.status == "canceled"] == [parent_item_id, child_item_id]

    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(parent_item_id)
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(child_item_id)


def test_cancel_queue_item_cascade_emits_canceled_events_for_waiting_parent_and_running_child(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
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
                "in_progress",
                "workflow-call-1",
                parent_item_id,
                parent_session.id,
                parent_item_id,
                1,
            ),
        )
        child_item_id = cursor.lastrowid

    session_queue.cancel_queue_item(child_item_id)

    queue_events = [event for event in event_bus.events if isinstance(event, QueueItemStatusChangedEvent)]
    canceled_events = [event for event in queue_events if event.status == "canceled"]

    assert [event.item_id for event in canceled_events] == [parent_item_id, child_item_id]
    assert canceled_events[-1].queue_status.canceled == 2
    assert canceled_events[-1].queue_status.waiting == 0
    assert canceled_events[-1].queue_status.in_progress == 0


def test_cancel_workflow_call_children_cancels_nested_descendants(session_queue: SqliteSessionQueue) -> None:
    root_session = GraphExecutionState(graph=Graph())
    waiting_child_session = GraphExecutionState(graph=Graph())
    nested_child_session = GraphExecutionState(graph=Graph())
    sibling_session = GraphExecutionState(graph=Graph())

    root_item_id = _insert_queue_item(session_queue, session=root_session, status="waiting")
    waiting_child_item_id = _insert_queue_item(
        session_queue,
        session=waiting_child_session,
        status="waiting",
        workflow_call_id="workflow-call-1",
        parent_item_id=root_item_id,
        parent_session_id=root_session.id,
        root_item_id=root_item_id,
        workflow_call_depth=1,
    )
    nested_child_item_id = _insert_queue_item(
        session_queue,
        session=nested_child_session,
        status="in_progress",
        workflow_call_id="workflow-call-2",
        parent_item_id=waiting_child_item_id,
        parent_session_id=waiting_child_session.id,
        root_item_id=root_item_id,
        workflow_call_depth=2,
    )
    sibling_item_id = _insert_queue_item(
        session_queue,
        session=sibling_session,
        status="pending",
        workflow_call_id="workflow-call-1",
        parent_item_id=root_item_id,
        parent_session_id=root_session.id,
        root_item_id=root_item_id,
        workflow_call_depth=1,
    )

    canceled_item_ids = session_queue.cancel_workflow_call_children("workflow-call-1")

    assert canceled_item_ids == [waiting_child_item_id, nested_child_item_id, sibling_item_id]
    assert session_queue.get_queue_item(waiting_child_item_id).status == "canceled"
    assert session_queue.get_queue_item(nested_child_item_id).status == "canceled"
    assert session_queue.get_queue_item(sibling_item_id).status == "canceled"


def test_cancel_all_except_current_cancels_waiting_chains_outside_current_chain(
    session_queue: SqliteSessionQueue,
) -> None:
    current_parent_session = GraphExecutionState(graph=Graph())
    current_child_session = GraphExecutionState(graph=Graph())
    other_parent_session = GraphExecutionState(graph=Graph())
    other_child_session = GraphExecutionState(graph=Graph())

    current_parent_item_id = _insert_queue_item(session_queue, session=current_parent_session, status="waiting")
    current_child_item_id = _insert_queue_item(
        session_queue,
        session=current_child_session,
        status="in_progress",
        workflow_call_id="workflow-call-current",
        parent_item_id=current_parent_item_id,
        parent_session_id=current_parent_session.id,
        root_item_id=current_parent_item_id,
        workflow_call_depth=1,
    )
    other_parent_item_id = _insert_queue_item(session_queue, session=other_parent_session, status="waiting")
    other_child_item_id = _insert_queue_item(
        session_queue,
        session=other_child_session,
        status="pending",
        workflow_call_id="workflow-call-other",
        parent_item_id=other_parent_item_id,
        parent_session_id=other_parent_session.id,
        root_item_id=other_parent_item_id,
        workflow_call_depth=1,
    )

    result = session_queue.cancel_all_except_current("default")

    assert result.canceled == 2
    assert session_queue.get_queue_item(current_parent_item_id).status == "waiting"
    assert session_queue.get_queue_item(current_child_item_id).status == "in_progress"
    assert session_queue.get_queue_item(other_parent_item_id).status == "canceled"
    assert session_queue.get_queue_item(other_child_item_id).status == "canceled"


def test_delete_all_except_current_deletes_waiting_chains_outside_current_chain(
    session_queue: SqliteSessionQueue,
) -> None:
    current_parent_session = GraphExecutionState(graph=Graph())
    current_child_session = GraphExecutionState(graph=Graph())
    other_parent_session = GraphExecutionState(graph=Graph())
    other_child_session = GraphExecutionState(graph=Graph())

    current_parent_item_id = _insert_queue_item(session_queue, session=current_parent_session, status="waiting")
    current_child_item_id = _insert_queue_item(
        session_queue,
        session=current_child_session,
        status="in_progress",
        workflow_call_id="workflow-call-current",
        parent_item_id=current_parent_item_id,
        parent_session_id=current_parent_session.id,
        root_item_id=current_parent_item_id,
        workflow_call_depth=1,
    )
    other_parent_item_id = _insert_queue_item(session_queue, session=other_parent_session, status="waiting")
    other_child_item_id = _insert_queue_item(
        session_queue,
        session=other_child_session,
        status="pending",
        workflow_call_id="workflow-call-other",
        parent_item_id=other_parent_item_id,
        parent_session_id=other_parent_session.id,
        root_item_id=other_parent_item_id,
        workflow_call_depth=1,
    )

    result = session_queue.delete_all_except_current("default")

    assert result.deleted == 2
    assert session_queue.get_queue_item(current_parent_item_id).status == "waiting"
    assert session_queue.get_queue_item(current_child_item_id).status == "in_progress"
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(other_parent_item_id)
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(other_child_item_id)


def test_cancel_all_except_current_preserves_waiting_chain_during_pending_child_handoff(
    session_queue: SqliteSessionQueue,
) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())
    unrelated_session = GraphExecutionState(graph=Graph())

    parent_item_id = _insert_queue_item(session_queue, session=parent_session, status="waiting")
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        status="pending",
        workflow_call_id="workflow-call-current",
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )
    unrelated_item_id = _insert_queue_item(session_queue, session=unrelated_session, status="pending")

    result = session_queue.cancel_all_except_current("default")

    assert result.canceled == 1
    assert session_queue.get_queue_item(parent_item_id).status == "waiting"
    assert session_queue.get_queue_item(child_item_id).status == "pending"
    assert session_queue.get_queue_item(unrelated_item_id).status == "canceled"


def test_delete_all_except_current_preserves_waiting_chain_during_pending_child_handoff(
    session_queue: SqliteSessionQueue,
) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())
    unrelated_session = GraphExecutionState(graph=Graph())

    parent_item_id = _insert_queue_item(session_queue, session=parent_session, status="waiting")
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        status="pending",
        workflow_call_id="workflow-call-current",
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )
    unrelated_item_id = _insert_queue_item(session_queue, session=unrelated_session, status="pending")

    result = session_queue.delete_all_except_current("default")

    assert result.deleted == 1
    assert session_queue.get_queue_item(parent_item_id).status == "waiting"
    assert session_queue.get_queue_item(child_item_id).status == "pending"
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(unrelated_item_id)


def test_cancel_by_queue_id_cancels_current_workflow_call_descendants(session_queue: SqliteSessionQueue) -> None:
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())
    nested_child_session = GraphExecutionState(graph=Graph())

    parent_item_id = _insert_queue_item(session_queue, session=parent_session, status="waiting")
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        status="in_progress",
        workflow_call_id="workflow-call-1",
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )
    nested_child_item_id = _insert_queue_item(
        session_queue,
        session=nested_child_session,
        status="pending",
        workflow_call_id="workflow-call-2",
        parent_item_id=child_item_id,
        parent_session_id=child_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=2,
    )

    session_queue.cancel_by_queue_id("default")

    assert session_queue.get_queue_item(parent_item_id).status == "canceled"
    assert session_queue.get_queue_item(child_item_id).status == "canceled"
    assert session_queue.get_queue_item(nested_child_item_id).status == "canceled"


def test_cancel_by_batch_ids_cancels_waiting_parent_and_in_progress_child(
    session_queue: SqliteSessionQueue,
) -> None:
    batch_id = str(uuid.uuid4())
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    parent_item_id = _insert_queue_item(
        session_queue,
        session=parent_session,
        status="waiting",
        batch_id=batch_id,
    )
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        status="in_progress",
        batch_id=batch_id,
        workflow_call_id="workflow-call-1",
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )

    session_queue.cancel_by_batch_ids("default", [batch_id])

    assert session_queue.get_queue_item(parent_item_id).status == "canceled"
    assert session_queue.get_queue_item(child_item_id).status == "canceled"


def test_cancel_by_destination_cancels_waiting_parent_and_in_progress_child(
    session_queue: SqliteSessionQueue,
) -> None:
    destination = "gallery"
    parent_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    parent_item_id = _insert_queue_item(
        session_queue,
        session=parent_session,
        status="waiting",
        destination=destination,
    )
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        status="in_progress",
        destination=destination,
        workflow_call_id="workflow-call-1",
        parent_item_id=parent_item_id,
        parent_session_id=parent_session.id,
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )

    session_queue.cancel_by_destination("default", destination)

    assert session_queue.get_queue_item(parent_item_id).status == "canceled"
    assert session_queue.get_queue_item(child_item_id).status == "canceled"


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


def test_retry_items_by_id_emits_root_only_retry_event_for_nested_failure_chain(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
    root_session = GraphExecutionState(graph=Graph())
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
                "canceled",
                "workflow-call-2",
                child_item_id,
                child_session.id,
                root_item_id,
                2,
            ),
        )
        grandchild_item_id = cursor.lastrowid

    retry_result = session_queue.retry_items_by_id("default", [grandchild_item_id, child_item_id, root_item_id])

    assert retry_result.retried_item_ids == [root_item_id]

    retry_events = [event for event in event_bus.events if isinstance(event, QueueItemsRetriedEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].retried_item_ids == [root_item_id]


def test_retry_items_by_id_respects_remaining_queue_capacity(session_queue: SqliteSessionQueue) -> None:
    root_session = GraphExecutionState(graph=Graph())
    pending_session = GraphExecutionState(graph=Graph())

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
                pending_session.model_dump_json(warnings=False),
                pending_session.id,
                str(uuid.uuid4()),
                0,
                "user-1",
                "pending",
            ),
        )
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

    session_queue._SqliteSessionQueue__invoker.services.configuration.max_queue_size = 1
    retry_result = session_queue.retry_items_by_id("default", [root_item_id])

    assert retry_result.retried_item_ids == []
    assert session_queue.get_queue_status("default").pending == 1
