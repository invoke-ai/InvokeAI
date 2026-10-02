import uuid
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.app.invocations.loops import (
    ForInvocation,
    ForReturnInvocation,
    StateGetInvocation,
    StateSetInvocation,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import AnyTypeTestInvocation, create_edge, create_loop_linkage


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    queue.start(mock_invoker)
    return queue


def _execute_next(state: GraphExecutionState) -> str | None:
    node = state.next()
    if node is None:
        return None
    output = node.invoke(Mock(InvocationContext))
    state.complete(node.id, output)
    return state.prepared_source_mapping[node.id]


def _stateful_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["alpha", "beta", "charlie"]))
    graph.add_node(StateSetInvocation(id="state_set", key="last_item"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(AnyTypeTestInvocation(id="after_collection"))
    graph.add_node(StateGetInvocation(id="after_state", key="last_item"))
    graph.add_edge(create_edge("for", "state", "state_set", "state"))
    graph.add_edge(create_edge("for", "item", "state_set", "value"))
    graph.add_edge(create_edge("state_set", "state", "return", "state"))
    graph.add_edge(create_edge("for", "output_collection", "after_collection", "value"))
    graph.add_edge(create_edge("for", "final_state", "after_state", "state"))
    graph.add_edge(create_edge("for", "item", "return", "output"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _nested_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=[["a", "b"], ["c", "d"]]))
    graph.add_node(AnyTypeTestInvocation(id="inner_collection"))
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(AnyTypeTestInvocation(id="inner_body"))
    graph.add_node(ForReturnInvocation(id="inner_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "inner_collection", "value"))
    graph.add_edge(create_edge("inner_collection", "value", "inner_for", "collection"))
    graph.add_edge(create_edge("inner_for", "item", "inner_body", "value"))
    graph.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    graph.add_edge(create_edge("inner_for", "output_collection", "outer_return", "output"))
    graph.add_edge(create_edge("outer_for", "output_collection", "after", "value"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _empty_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "value"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _empty_for_missing_state_graph() -> Graph:
    graph = _empty_for_graph()
    graph.add_node(StateGetInvocation(id="after_state", key="missing"))
    graph.add_edge(create_edge("for", "final_state", "after_state", "state"))
    return graph


def _insert_session(queue: SqliteSessionQueue, state: GraphExecutionState) -> int:
    session_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())
    session_json = state.model_dump_json(warnings=False, exclude_none=True)
    with queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values, priority,
                workflow, origin, destination, retried_from_item_id, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("default", session_json, session_id, batch_id, None, 0, None, None, None, None, "system"),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def test_sqlite_queue_resumes_partial_stateful_for_loop(session_queue: SqliteSessionQueue) -> None:
    item_id = _insert_session(session_queue, GraphExecutionState(graph=_stateful_for_graph()))

    queue_item = session_queue.dequeue()
    assert queue_item is not None
    assert queue_item.item_id == item_id
    state = queue_item.session

    # Finish the first iteration, then persist the in-progress state through SQLite.
    assert [_execute_next(state) for _ in range(3)] == ["for", "state_set", "return"]
    prepared_mapping = state.prepared_source_mapping.copy()
    # Use the GraphExecutionState JSON contract as the oracle for private prepared metadata.
    direct_round_trip = TypeAdapter(GraphExecutionState).validate_json(
        state.model_dump_json(warnings=False, exclude_none=True), strict=False
    )
    direct_iteration_paths = {
        exec_id: direct_round_trip._prepared_registry().get_iteration_path(exec_id) for exec_id in prepared_mapping
    }
    session_queue.save_queue_item_session(queue_item.item_id, state)

    reloaded_item = session_queue.get_queue_item(queue_item.item_id)
    resumed = reloaded_item.session
    assert resumed.prepared_source_mapping == prepared_mapping
    assert {
        exec_id: resumed._prepared_registry().get_iteration_path(exec_id) for exec_id in prepared_mapping
    } == direct_iteration_paths
    assert not resumed.is_complete()
    assert "after_collection" not in resumed.prepared_source_mapping.values()

    remaining_sources: list[str] = []
    while (source_id := _execute_next(resumed)) is not None:
        remaining_sources.append(source_id)

    assert remaining_sources == [
        "for",
        "state_set",
        "return",
        "for",
        "state_set",
        "return",
        "after_collection",
        "after_state",
    ]
    assert resumed.is_complete()

    after_collection_id = next(
        exec_id for exec_id, source_id in resumed.prepared_source_mapping.items() if source_id == "after_collection"
    )
    after_state_id = next(
        exec_id for exec_id, source_id in resumed.prepared_source_mapping.items() if source_id == "after_state"
    )
    assert resumed.results[after_collection_id].value == ["alpha", "beta", "charlie"]
    assert resumed.results[after_state_id].value == "charlie"

    session_queue.set_queue_item_session(queue_item.item_id, resumed)
    final_item = session_queue.complete_queue_item(queue_item.item_id)
    assert final_item.status == "completed"
    assert final_item.session.is_complete()
    assert final_item.session.results[after_collection_id].value == ["alpha", "beta", "charlie"]
    assert final_item.session.results[after_state_id].value == "charlie"


def test_sqlite_queue_round_trips_a_checkpoint_before_each_next_node(
    session_queue: SqliteSessionQueue,
) -> None:
    """Every manually persisted node boundary must survive a process restart before the next node.

    This isolates the SQLite checkpoint contract; processor save cadence is intentionally outside
    this test because this test-only change does not alter the session processor.
    """
    item_id = _insert_session(session_queue, GraphExecutionState(graph=_stateful_for_graph()))

    queue_item = session_queue.dequeue()
    assert queue_item is not None
    state = queue_item.session
    completed_sources: list[str] = []

    while (source_id := _execute_next(state)) is not None:
        completed_sources.append(source_id)
        session_queue.save_queue_item_session(item_id, state)
        persisted = session_queue.get_queue_item(item_id).session

        assert persisted.executed == state.executed
        assert persisted.results.keys() == state.results.keys()
        assert persisted.prepared_source_mapping == state.prepared_source_mapping
        state = persisted

    assert completed_sources == [
        "for",
        "state_set",
        "return",
        "for",
        "state_set",
        "return",
        "for",
        "state_set",
        "return",
        "after_collection",
        "after_state",
    ]
    assert state.is_complete()
    after_collection_id = next(
        exec_id for exec_id, source_id in state.prepared_source_mapping.items() if source_id == "after_collection"
    )
    after_state_id = next(
        exec_id for exec_id, source_id in state.prepared_source_mapping.items() if source_id == "after_state"
    )
    assert state.results[after_collection_id].value == ["alpha", "beta", "charlie"]
    assert state.results[after_state_id].value == "charlie"
    session_queue.set_queue_item_session(item_id, state)
    final_item = session_queue.complete_queue_item(item_id)
    assert final_item.status == "completed"
    assert final_item.session.is_complete()


def test_sqlite_queue_round_trips_empty_for_final_output(session_queue: SqliteSessionQueue) -> None:
    item_id = _insert_session(session_queue, GraphExecutionState(graph=_empty_for_graph()))

    queue_item = session_queue.dequeue()
    assert queue_item is not None
    state = queue_item.session
    after_node = state.next()
    assert isinstance(after_node, AnyTypeTestInvocation)

    session_queue.save_queue_item_session(queue_item.item_id, state)
    resumed = session_queue.get_queue_item(item_id).session

    for_exec_id = next(exec_id for exec_id, source_id in resumed.prepared_source_mapping.items() if source_id == "for")
    for_output = resumed.results[for_exec_id]
    assert for_output.type == "for_output"
    assert for_output.item is None
    assert for_output.index == -1
    assert for_output.total == 0


def test_sqlite_queue_round_trips_missing_loop_state_value_after_empty_for(
    session_queue: SqliteSessionQueue,
) -> None:
    item_id = _insert_session(session_queue, GraphExecutionState(graph=_empty_for_missing_state_graph()))

    queue_item = session_queue.dequeue()
    assert queue_item is not None
    state = queue_item.session
    after_state = state.next()
    while not isinstance(after_state, StateGetInvocation):
        assert isinstance(after_state, AnyTypeTestInvocation)
        state.complete(after_state.id, after_state.invoke(Mock(InvocationContext)))
        after_state = state.next()
    assert isinstance(after_state, StateGetInvocation)
    state.complete(after_state.id, after_state.invoke(Mock(InvocationContext)))

    session_queue.save_queue_item_session(queue_item.item_id, state)
    resumed = session_queue.get_queue_item(item_id).session

    assert resumed.results[after_state.id].value is None


def test_sqlite_queue_resumes_nested_for_after_first_outer_iteration(
    session_queue: SqliteSessionQueue,
) -> None:
    item_id = _insert_session(session_queue, GraphExecutionState(graph=_nested_for_graph()))

    queue_item = session_queue.dequeue()
    assert queue_item is not None
    state = queue_item.session
    completed_sources = [_execute_next(state) for _ in range(9)]
    assert completed_sources == [
        "outer_for",
        "inner_collection",
        "inner_for",
        "inner_body",
        "inner_return",
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
    ]
    assert state.finalized_loop_contexts == {("inner_for", (0,))}
    prepared_mapping = state.prepared_source_mapping.copy()
    prepared_paths = {exec_id: state._get_iteration_path(exec_id) for exec_id in prepared_mapping}

    session_queue.save_queue_item_session(queue_item.item_id, state)
    resumed = session_queue.get_queue_item(item_id).session

    assert resumed.finalized_loop_contexts == {("inner_for", (0,))}
    assert resumed.prepared_source_mapping == prepared_mapping
    assert {exec_id: resumed._get_iteration_path(exec_id) for exec_id in prepared_mapping} == prepared_paths
    remaining_sources: list[str] = []
    while (source_id := _execute_next(resumed)) is not None:
        remaining_sources.append(source_id)

    assert remaining_sources == [
        "outer_for",
        "inner_collection",
        "inner_for",
        "inner_body",
        "inner_return",
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
        "after",
    ]
    after_exec_id = next(
        exec_id for exec_id, source_id in resumed.prepared_source_mapping.items() if source_id == "after"
    )
    assert resumed.results[after_exec_id].value == [["a", "b"], ["c", "d"]]
    assert resumed.is_complete()
    session_queue.set_queue_item_session(item_id, resumed)
    final_item = session_queue.complete_queue_item(item_id)
    assert final_item.status == "completed"
    assert final_item.session.is_complete()
    assert final_item.session.results[after_exec_id].value == [["a", "b"], ["c", "d"]]
