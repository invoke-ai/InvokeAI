"""Manual graph-execution performance benchmarks.

These tests are marked slow and are excluded from normal pytest and CI runs. Run this benchmark with:

    pytest -m slow -s tests/app/services/shared/test_graph_execution_performance.py
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.workflow_return import (
    WorkflowReturnInvocation,
    WorkflowReturnOutput,
    WorkflowReturnValueInvocation,
)
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    CollectInvocationOutput,
    Graph,
    GraphExecutionState,
    IterateInvocation,
)
from tests.test_nodes import AnyTypeTestInvocation, create_edge

ITEM_COUNT = 300
CHILD_NODE_COUNT = 20


@dataclass(frozen=True)
class WorkflowCallBenchmarkResult:
    items: int
    child_nodes: int
    calls: int
    final_execution_nodes: int
    final_session_json_bytes: int
    total_seconds: float
    boundary_restore_seconds: float
    serialization_seconds: float
    child_execution_seconds: float
    traced_final_restore_seconds: float
    traced_final_restore_live_bytes: int
    traced_final_restore_peak_bytes: int


def _build_child_graph(node_count: int) -> Graph:
    if node_count < 3:
        raise ValueError("The child workflow needs a value node, a return-value node, and a return node")

    graph = Graph()
    passthrough_node_count = node_count - 2
    for index in range(passthrough_node_count):
        node_id = f"child_{index}"
        graph.add_node(AnyTypeTestInvocation(id=node_id, value=index if index == 0 else None))
        if index:
            graph.add_edge(create_edge(f"child_{index - 1}", "value", node_id, "value"))
    graph.add_node(WorkflowReturnValueInvocation(id="child_return_value", key="value"))
    graph.add_node(WorkflowReturnInvocation(id="child_return"))
    graph.add_edge(create_edge(f"child_{passthrough_node_count - 1}", "value", "child_return_value", "value"))
    graph.add_edge(create_edge("child_return_value", "value", "child_return", "values"))
    return graph


def _execute_child(session: GraphExecutionState) -> WorkflowReturnOutput:
    context = Mock(InvocationContext)
    while (invocation := session.next()) is not None:
        session.complete(invocation.id, invocation.invoke(context))

    prepared_return_id = next(iter(session.source_prepared_mapping["child_return"]))
    output = session.results[prepared_return_id]
    assert isinstance(output, WorkflowReturnOutput)
    return output


def _build_main_graph(item_count: int) -> Graph:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=item_count, step=1))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(AddInvocation(id="before_call", b=1))
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved-workflow"))
    graph.add_node(AnyTypeTestInvocation(id="after_call"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("range", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "before_call", "a"))
    graph.add_edge(create_edge("before_call", "value", "call", "saved_workflow_input::input::value"))
    graph.add_edge(create_edge("call", "values", "after_call", "value"))
    graph.add_edge(create_edge("after_call", "value", "collect", "item"))
    return graph


def _run_workflow_call_benchmark(item_count: int, child_node_count: int) -> WorkflowCallBenchmarkResult:
    state = GraphExecutionState(graph=_build_main_graph(item_count))
    child_graph = _build_child_graph(child_node_count)
    context = Mock(InvocationContext)
    calls = 0
    boundary_restore_seconds = 0.0
    serialization_seconds = 0.0
    child_execution_seconds = 0.0
    started = time.perf_counter()

    while (invocation := state.next()) is not None:
        if isinstance(invocation, CallSavedWorkflowInvocation):
            child_started = time.perf_counter()
            frame = state.build_workflow_call_frame(invocation.id, invocation.workflow_id)
            child = state.create_child_workflow_execution_state(child_graph, frame)
            child_output = _execute_child(child)
            child_execution_seconds += time.perf_counter() - child_started

            state.begin_waiting_on_workflow_call(frame)
            state.attach_waiting_workflow_call_child_session(child)

            serialization_started = time.perf_counter()
            raw = state.model_dump_json(warnings=False, exclude_none=True)
            serialization_seconds += time.perf_counter() - serialization_started

            restore_started = time.perf_counter()
            state = GraphExecutionState.model_validate_json(raw)
            boundary_restore_seconds += time.perf_counter() - restore_started

            state.end_waiting_on_workflow_call(status="completed")
            state.complete(invocation.id, child_output)
            calls += 1
        else:
            state.complete(invocation.id, invocation.invoke(context))

    total_seconds = time.perf_counter() - started
    final_session_json = state.model_dump_json(warnings=False, exclude_none=True)

    prepared_collect_id = next(iter(state.source_prepared_mapping["collect"]))
    collect_output = state.results[prepared_collect_id]
    assert isinstance(collect_output, CollectInvocationOutput)
    assert len(collect_output.collection) == item_count
    assert calls == item_count
    assert state.is_complete()

    gc.collect()
    tracemalloc.start()
    try:
        restore_started = time.perf_counter()
        restored_state = GraphExecutionState.model_validate_json(final_session_json)
        traced_final_restore_seconds = time.perf_counter() - restore_started
        traced_final_restore_live_bytes, traced_final_restore_peak_bytes = tracemalloc.get_traced_memory()
        assert restored_state.is_complete()
    finally:
        tracemalloc.stop()

    return WorkflowCallBenchmarkResult(
        items=item_count,
        child_nodes=child_node_count,
        calls=calls,
        final_execution_nodes=len(state.execution_graph.nodes),
        final_session_json_bytes=len(final_session_json.encode()),
        total_seconds=round(total_seconds, 6),
        boundary_restore_seconds=round(boundary_restore_seconds, 6),
        serialization_seconds=round(serialization_seconds, 6),
        child_execution_seconds=round(child_execution_seconds, 6),
        traced_final_restore_seconds=round(traced_final_restore_seconds, 6),
        traced_final_restore_live_bytes=traced_final_restore_live_bytes,
        traced_final_restore_peak_bytes=traced_final_restore_peak_bytes,
    )


@pytest.mark.slow
def test_iterated_call_saved_workflow_performance() -> None:
    """Benchmark a 20-node saved workflow called once for each of 300 iterated items."""
    result = _run_workflow_call_benchmark(item_count=ITEM_COUNT, child_node_count=CHILD_NODE_COUNT)

    print("\nIterated Call Saved Workflow benchmark:")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
