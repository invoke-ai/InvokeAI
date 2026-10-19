from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.loops import ForInvocation, ForInvocationOutput, ForReturnInvocation, LoopState
from invokeai.app.invocations.primitives import BooleanOutput
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder, ExecutionInterface
from invokeai.app.services.shared.graph import CollectInvocation, Graph, GraphExecutionState, IterateInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from tests.app.services.workflow_call_test_utils import (
    _DummyConfig,
    _DummyEvents,
    _DummyLogger,
    _DummySessionQueue,
    _DummyStats,
)
from tests.test_nodes import create_edge, create_loop_linkage


@invocation_output("test_for_runner_value_output")
class ForRunnerValueOutput(BaseInvocationOutput):
    value: int = OutputField(description="The loop body value")


@invocation("test_for_runner_body", version="1.0.0")
class ForRunnerBodyInvocation(BaseInvocation):
    value: int = InputField(default=0, description="The current loop item")
    fail_on: int | None = InputField(default=None, description="The value that raises an exception")

    def invoke(self, context: InvocationContext) -> ForRunnerValueOutput:
        if self.value == self.fail_on:
            raise ValueError(f"Refusing loop value {self.value}")
        return ForRunnerValueOutput(value=self.value)


@invocation_output("test_for_runner_collection_adapter_output")
class ForRunnerCollectionAdapterOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The inner loop collection")


@invocation("test_for_runner_collection_adapter", version="1.0.0")
class ForRunnerCollectionAdapterInvocation(BaseInvocation):
    value: Any = InputField(default=None, description="The outer loop item")

    def invoke(self, context: InvocationContext) -> ForRunnerCollectionAdapterOutput:
        return ForRunnerCollectionAdapterOutput(collection=self.value)


@invocation_output("test_for_runner_collection_output")
class ForRunnerCollectionOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The completed loop collection")


@invocation("test_for_runner_collection", version="1.0.0")
class ForRunnerCollectionInvocation(BaseInvocation):
    collection: list[Any] = InputField(default_factory=list, description="The completed loop collection")

    def invoke(self, context: InvocationContext) -> ForRunnerCollectionOutput:
        return ForRunnerCollectionOutput(collection=self.collection)


@invocation("test_for_runner_condition", version="1.0.0")
class ForRunnerConditionInvocation(BaseInvocation):
    value: Any = InputField(default=None)
    continue_condition: bool = InputField(default=True)

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        return BooleanOutput(value=self.continue_condition)


def _build_graph(*, fail_on: int | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[1, 2, 3]))
    graph.add_node(ForRunnerBodyInvocation(id="body", fail_on=fail_on))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(ForRunnerCollectionInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _build_nested_graph(*, fail_on: int | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[[1, 2], [3, 4]]))
    graph.add_node(ForRunnerCollectionAdapterInvocation(id="adapter"))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(ForRunnerBodyInvocation(id="body", fail_on=fail_on))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(ForRunnerCollectionInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "adapter", "value"))
    graph.add_edge(create_edge("adapter", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "collect", "item"))
    graph.add_edge(create_edge("collect", "collection", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _build_nested_for_graph(
    *,
    fail_on: int | None = None,
    collection: list[list[int]] | None = None,
    break_inner_after_first: bool = False,
    break_outer_after_first: bool = False,
    include_after: bool = True,
) -> Graph:
    graph = Graph()
    graph.add_node(
        ForInvocation(
            id="outer_for",
            collection=[[1, 2], [3, 4]] if collection is None else collection,
            state=LoopState(values={"outer": True}),
        )
    )
    graph.add_node(ForRunnerCollectionAdapterInvocation(id="inner_collection"))
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(ForRunnerBodyInvocation(id="inner_body", fail_on=fail_on))
    graph.add_node(
        ForReturnInvocation(
            id="inner_return",
            continue_condition=False if break_inner_after_first else None,
        )
    )
    graph.add_node(ForRunnerConditionInvocation(id="outer_condition", continue_condition=not break_outer_after_first))
    graph.add_node(ForRunnerCollectionInvocation(id="outer_output"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    if include_after:
        graph.add_node(ForRunnerCollectionInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "inner_collection", "value"))
    graph.add_edge(create_edge("inner_collection", "collection", "inner_for", "collection"))
    graph.add_edge(create_edge("inner_for", "item", "inner_body", "value"))
    graph.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    graph.add_edge(create_edge("inner_for", "output_collection", "outer_output", "collection"))
    graph.add_edge(create_edge("outer_output", "collection", "outer_return", "output"))
    graph.add_edge(create_edge("inner_for", "output_collection", "outer_condition", "value"))
    graph.add_edge(create_edge("outer_condition", "value", "outer_return", "continue_condition"))
    graph.add_edge(create_edge("outer_for", "state", "outer_return", "state"))
    if include_after:
        graph.add_edge(create_edge("outer_for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _build_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_after_run_node=None,
) -> tuple[DefaultSessionRunner, Event, _DummySessionQueue, _DummyEvents]:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: (
            effects := ExecutionEffectsRecorder(
                source_node_id=data.invocation.id,
                frame_path=data.execution_frame,
                state_id=data.execution_state_id,
                frame_id=data.execution_frame_id,
                workflow_call_depth=data.execution_workflow_call_depth,
            )
        )
        and SimpleNamespace(execution_effects=effects, execution=ExecutionInterface(effects)),
    )

    cancel_event = Event()
    session_queue = _DummySessionQueue()
    events = _DummyEvents()
    runner = DefaultSessionRunner(on_after_run_node_callbacks=[] if on_after_run_node is None else [on_after_run_node])
    runner.start(
        services=SimpleNamespace(
            performance_statistics=_DummyStats(),
            progress_previews=MemoryProgressPreviews(),
            events=events,
            logger=_DummyLogger(),
            configuration=_DummyConfig(),
            session_queue=session_queue,
            invocation_cache=MemoryInvocationCache(max_cache_size=0),
        ),
        cancel_event=cancel_event,
    )
    return runner, cancel_event, session_queue, events


def _build_queue_item(session: GraphExecutionState) -> SimpleNamespace:
    return SimpleNamespace(
        item_id=1,
        status="in_progress",
        session=session,
        session_id=session.id,
    )


def _completed_source_ids(events: _DummyEvents, session: GraphExecutionState) -> list[str]:
    # Final For output updates reuse the execution ID; count executions separately from output updates.
    return [session.prepared_source_mapping[exec_id] for exec_id in dict.fromkeys(i.id for i, _, _ in events.completed)]


def test_session_runner_completes_for_loop_and_persists_final_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    session = GraphExecutionState(graph=_build_graph())
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session_queue.completed_item_ids == [queue_item.item_id]
    assert session_queue.session_updates[-1] == (queue_item.item_id, session)
    assert session.is_complete()
    assert _completed_source_ids(events, session).count("for") == 3
    assert _completed_source_ids(events, session).count("body") == 3
    assert _completed_source_ids(events, session).count("return") == 3
    assert _completed_source_ids(events, session)[-1] == "after"

    [after_exec_id] = session.source_prepared_mapping["after"]
    assert session.results[after_exec_id] == ForRunnerCollectionOutput(collection=[1, 2, 3])
    assert any(
        isinstance(output := session.results.get(exec_id), ForInvocationOutput)
        and output.output_collection == [1, 2, 3]
        for exec_id in session.source_prepared_mapping["for"]
    )


def test_session_runner_cancellation_stops_for_loop_without_releasing_final_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_graph())
    callback_state: dict[str, Any] = {}

    def cancel_after_first_return(invocation, queue_item, output) -> None:
        session_queue = callback_state["session_queue"]
        cancel_event = callback_state["cancel_event"]
        if queue_item.session.prepared_source_mapping[invocation.id] == "return":
            session_queue.cancel_queue_item(queue_item.item_id)
            cancel_event.set()

    runner, cancel_event, session_queue, events = _build_runner(
        monkeypatch, on_after_run_node=cancel_after_first_return
    )
    callback_state.update(cancel_event=cancel_event, session_queue=session_queue)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    completed_source_ids = _completed_source_ids(events, session)
    assert queue_item.status == "canceled"
    assert session_queue.canceled_item_ids == [queue_item.item_id]
    assert session_queue.completed_item_ids == []
    assert session_queue.session_updates[-1] == (queue_item.item_id, session)
    assert not session.is_complete()
    assert completed_source_ids.count("for") == 1
    assert completed_source_ids.count("body") == 1
    assert completed_source_ids.count("return") == 1
    assert "after" not in session.source_prepared_mapping
    assert ("for", ()) not in session.finalized_loop_contexts
    assert len(session.source_prepared_mapping["for"]) == 2
    assert sum(exec_id in session.results for exec_id in session.source_prepared_mapping["for"]) == 1
    assert not any(
        isinstance(output := session.results.get(exec_id), ForInvocationOutput) and output.output_collection
        for exec_id in session.source_prepared_mapping["for"]
    )


def test_session_runner_body_exception_fails_and_cleans_up_for_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    session = GraphExecutionState(graph=_build_graph(fail_on=2))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    completed_source_ids = _completed_source_ids(events, session)
    assert queue_item.status == "failed"
    assert session_queue.failed_item_ids == [queue_item.item_id]
    assert session_queue.completed_item_ids == []
    assert session_queue.session_updates[-1] == (queue_item.item_id, session)
    assert session.has_error()
    assert len(session.errors) == 1
    [failed_exec_id] = session.errors
    assert session.prepared_source_mapping[failed_exec_id] == "body"
    assert session.errors[failed_exec_id] == "ValueError: Refusing loop value 2"
    assert failed_exec_id not in session.results
    assert len(events.errors) == 1
    assert events.errors[0][1].id == failed_exec_id
    assert completed_source_ids.count("for") == 2
    assert completed_source_ids.count("body") == 1
    assert completed_source_ids.count("return") == 1
    assert "after" not in session.source_prepared_mapping
    assert ("for", ()) not in session.finalized_loop_contexts
    assert len(session.source_prepared_mapping["for"]) == 2
    assert session.next() is None
    assert not any(
        isinstance(output := session.results.get(exec_id), ForInvocationOutput) and output.output_collection
        for exec_id in session.source_prepared_mapping["for"]
    )


def test_session_runner_nested_iterate_cancellation_stops_outer_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    session = GraphExecutionState(graph=_build_nested_graph())
    callback_state: dict[str, Any] = {}

    def cancel_after_nested_return(invocation, queue_item, output) -> None:
        if queue_item.session.prepared_source_mapping[invocation.id] == "return":
            callback_state["session_queue"].cancel_queue_item(queue_item.item_id)
            callback_state["cancel_event"].set()

    runner, cancel_event, session_queue, events = _build_runner(
        monkeypatch, on_after_run_node=cancel_after_nested_return
    )
    callback_state.update(cancel_event=cancel_event, session_queue=session_queue)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    completed_source_ids = _completed_source_ids(events, session)
    assert queue_item.status == "canceled"
    assert completed_source_ids.count("for") == 1
    assert completed_source_ids.count("iterate") == 2
    assert completed_source_ids.count("body") == 2
    assert completed_source_ids.count("collect") == 1
    assert completed_source_ids.count("return") == 1
    assert "after" not in session.source_prepared_mapping
    assert not session.finalized_loop_contexts


def test_session_runner_nested_for_cancellation_stops_outer_loop_without_final_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph())
    callback_state: dict[str, Any] = {}

    def cancel_after_inner_return(invocation, queue_item, output) -> None:
        if queue_item.session.prepared_source_mapping[invocation.id] == "inner_return":
            callback_state["session_queue"].cancel_queue_item(queue_item.item_id)
            callback_state["cancel_event"].set()

    runner, cancel_event, session_queue, events = _build_runner(
        monkeypatch, on_after_run_node=cancel_after_inner_return
    )
    callback_state.update(cancel_event=cancel_event, session_queue=session_queue)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    completed_source_ids = _completed_source_ids(events, session)
    assert queue_item.status == "canceled"
    assert session_queue.canceled_item_ids == [queue_item.item_id]
    assert session_queue.completed_item_ids == []
    assert session_queue.session_updates[-1] == (queue_item.item_id, session)
    assert not session.is_complete()
    assert completed_source_ids.count("outer_for") == 1
    assert completed_source_ids.count("inner_for") == 1
    assert completed_source_ids.count("inner_body") == 1
    assert completed_source_ids.count("inner_return") == 1
    assert "outer_return" not in completed_source_ids
    assert "after" not in session.source_prepared_mapping
    assert not session.finalized_loop_contexts
    assert len(session.source_prepared_mapping["inner_for"]) == 2
    assert sum(exec_id in session.results for exec_id in session.source_prepared_mapping["inner_for"]) == 1
    assert len(session.source_prepared_mapping["inner_body"]) == 2
    assert sum(exec_id in session.results for exec_id in session.source_prepared_mapping["inner_body"]) == 1
    assert len(session.source_prepared_mapping["inner_return"]) == 2
    assert sum(exec_id in session.results for exec_id in session.source_prepared_mapping["inner_return"]) == 1
    assert not any(
        isinstance(output := session.results.get(exec_id), ForInvocationOutput) and output.output_collection
        for exec_id in session.source_prepared_mapping.get("outer_for", set())
    )


def test_session_runner_nested_iterate_body_exception_fails_outer_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    session = GraphExecutionState(graph=_build_nested_graph(fail_on=2))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    completed_source_ids = _completed_source_ids(events, session)
    assert queue_item.status == "failed"
    assert session.has_error()
    [failed_exec_id] = session.errors
    assert session.prepared_source_mapping[failed_exec_id] == "body"
    assert session.errors[failed_exec_id] == "ValueError: Refusing loop value 2"
    assert completed_source_ids.count("for") == 1
    assert completed_source_ids.count("iterate") == 2
    assert completed_source_ids.count("body") == 1
    assert "after" not in session.source_prepared_mapping
    assert not session.finalized_loop_contexts


def test_session_runner_completes_nested_for_and_releases_outer_final_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph())
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session_queue.completed_item_ids == [queue_item.item_id]
    assert session.is_complete()
    completed_source_ids = _completed_source_ids(events, session)
    assert completed_source_ids.count("outer_for") == 2
    assert completed_source_ids.count("inner_for") == 4
    assert completed_source_ids.count("inner_body") == 4
    assert completed_source_ids.count("inner_return") == 4
    assert completed_source_ids.count("outer_return") == 2

    [after_exec_id] = session.source_prepared_mapping["after"]
    assert session.results[after_exec_id] == ForRunnerCollectionOutput(collection=[[1, 2], [3, 4]])


def test_session_runner_completes_nested_for_without_downstream_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(include_after=False))
    runner, _cancel_event, session_queue, _events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session_queue.completed_item_ids == [queue_item.item_id]
    assert session.is_complete()


def test_session_runner_does_not_duplicate_empty_nested_loop_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(collection=[[], []]))
    runner, _cancel_event, session_queue, _events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert len(session.executed_history) == len(set(session.executed_history))


def test_session_runner_completes_nested_for_with_empty_inner_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(collection=[[], [3, 4]]))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session.is_complete()
    completed_source_ids = _completed_source_ids(events, session)
    assert completed_source_ids.count("outer_return") == 2
    assert completed_source_ids.count("inner_body") == 2
    assert completed_source_ids.count("inner_return") == 2
    [after_exec_id] = session.source_prepared_mapping["after"]
    assert session.results[after_exec_id] == ForRunnerCollectionOutput(collection=[[], [3, 4]])


def test_session_runner_nested_for_early_break_releases_each_inner_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(break_inner_after_first=True))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session.is_complete()
    completed_source_ids = _completed_source_ids(events, session)
    assert completed_source_ids.count("outer_for") == 2
    assert completed_source_ids.count("inner_for") == 2
    assert completed_source_ids.count("inner_body") == 2
    assert completed_source_ids.count("inner_return") == 2
    assert completed_source_ids.count("outer_return") == 2
    [after_exec_id] = session.source_prepared_mapping["after"]
    assert session.results[after_exec_id] == ForRunnerCollectionOutput(collection=[[1], [3]])


def test_session_runner_nested_for_connected_early_break_releases_outer_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(break_outer_after_first=True))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "completed"
    assert session.is_complete()
    completed_source_ids = _completed_source_ids(events, session)
    assert completed_source_ids.count("outer_for") == 1
    assert completed_source_ids.count("inner_for") == 2
    assert completed_source_ids.count("inner_body") == 2
    assert completed_source_ids.count("inner_return") == 2
    assert completed_source_ids.count("outer_condition") == 1
    assert completed_source_ids.count("outer_return") == 1
    [after_exec_id] = session.source_prepared_mapping["after"]
    assert session.results[after_exec_id] == ForRunnerCollectionOutput(collection=[[1, 2]])


def test_session_runner_nested_for_body_exception_fails_without_outer_final_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GraphExecutionState(graph=_build_nested_for_graph(fail_on=3))
    runner, _cancel_event, session_queue, events = _build_runner(monkeypatch)
    queue_item = _build_queue_item(session)
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item)

    assert queue_item.status == "failed"
    assert session.has_error()
    completed_source_ids = _completed_source_ids(events, session)
    assert completed_source_ids.count("outer_return") == 1
    assert completed_source_ids.count("inner_body") == 2
    assert completed_source_ids.count("inner_return") == 2
    assert "after" not in session.source_prepared_mapping
    assert ("outer_for", ()) not in session.finalized_loop_contexts
