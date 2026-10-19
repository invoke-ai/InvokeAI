from copy import deepcopy
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.logic import IfInvocation, IfInvocationOutput
from invokeai.app.invocations.loops import (
    ForInvocation,
    ForInvocationOutput,
    ForReturnInvocation,
    ForReturnInvocationOutput,
    LoopState,
)
from invokeai.app.invocations.workflow_return import WorkflowReturnOutput
from invokeai.app.services.shared.execution_effects import (
    AddEdgeEffect,
    AwaitEffect,
    ChildExecutionHandle,
    CloseStreamEffect,
    ContinuationEffect,
    EmitEffect,
    ExecutionEffectsRecorder,
    ExecutionInterface,
    ExecutionRef,
    ExecutionToken,
    FailEffect,
    InvocationRunResult,
    RemoveEdgeEffect,
    SetValueEffect,
    SpawnExecutionEffect,
    UnsupportedExecutionEffectError,
)
from invokeai.app.services.shared.execution_engine.child import ChildExecutionCapability
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState, IterateInvocation
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    InvocationContextData,
    build_invocation_context,
)


@invocation_output("execution_effects_test_output")
class ExecutionEffectsTestOutput(BaseInvocationOutput):
    value: int = OutputField()


@invocation("execution_effects_test", version="1.0.0")
class ExecutionEffectsTestInvocation(BaseInvocation):
    execution_effects_enabled = True
    value: int = InputField(default=1)
    calls: ClassVar[int] = 0

    def invoke(self, context: InvocationContext) -> ExecutionEffectsTestOutput:
        type(self).calls += 1
        context.effects.record(
            EmitEffect(
                token=ExecutionToken(node_id=self.id, field="value", value=self.value),
                value=self.value,
            )
        )
        return ExecutionEffectsTestOutput(value=self.value)


@invocation("execution_effects_override_test", version="1.0.0")
class ExecutionEffectsOverrideInvocation(BaseInvocation):
    execution_effects_enabled = True
    value: int = InputField(default=1)
    calls: ClassVar[int] = 0

    def invoke(self, context: InvocationContext) -> ExecutionEffectsTestOutput:
        type(self).calls += 1
        context.effects.record(
            EmitEffect(
                token=ExecutionToken(node_id=self.id, field="value", value=self.value),
                value=self.value,
            )
        )
        return ExecutionEffectsTestOutput(value=self.value)

    def invoke_internal(self, context: InvocationContext, services: MagicMock) -> ExecutionEffectsTestOutput:
        cached_value = services.invocation_cache.get("override-cache-key")
        if cached_value is not None:
            return cached_value
        return self.invoke(context)


def _context() -> InvocationContext:
    context = object.__new__(InvocationContext)
    context.execution_effects = ExecutionEffectsRecorder()
    context.effects = context.execution_effects
    return context


def _services(cache_size: int = 1) -> MagicMock:
    services = MagicMock()
    services.configuration.node_cache_size = cache_size
    return services


def test_context_default_recorder_preserves_execution_frame() -> None:
    context = build_invocation_context(
        services=MagicMock(),
        data=InvocationContextData(
            queue_item=None,  # type: ignore[arg-type]
            invocation=ExecutionEffectsTestInvocation(id="node"),
            source_invocation_id="source",
            execution_frame=(2, 1),
            execution_state_id="state",
            execution_frame_id="frame",
            execution_workflow_call_depth=2,
        ),
        is_canceled=lambda: False,
    )

    assert context.execution_effects.source_node_id == "node"
    assert context.execution_effects.frame_path == (2, 1)
    assert context.execution_effects.execution_ref.state_id == "state"
    assert context.execution_effects.execution_ref.frame_id == "frame"
    assert context.execution_effects.execution_ref.workflow_call_depth == 2


def test_direct_and_builder_contexts_preserve_lifecycle_recorder_authority() -> None:
    services = MagicMock()
    capability = ChildExecutionCapability(
        parent_execution_id="node",
        parent_frame=(2, 1),
        authorization_context={"user_id": "user"},
    )
    data = InvocationContextData(
        queue_item=SimpleNamespace(user_id="user"),
        invocation=ExecutionEffectsTestInvocation(id="node"),
        source_invocation_id="source",
        execution_frame=(2, 1),
        execution_state_id="state",
        execution_frame_id="frame",
        execution_workflow_call_depth=3,
        execution_child_capability=capability,
    )
    built = build_invocation_context(services, data, lambda: False)
    direct = InvocationContext(
        images=built.images,
        videos=built.videos,
        tensors=built.tensors,
        conditioning=built.conditioning,
        models=built.models,
        logger=built.logger,
        config=built.config,
        util=built.util,
        boards=built.boards,
        wildcards=built.wildcards,
        data=data,
        services=services,
    )

    for context in (built, direct):
        recorder = context.execution_effects
        assert recorder.allow_lifecycle_effects is True
        assert recorder.child_capability is capability
        assert recorder.execution_ref.state_id == "state"
        assert recorder.execution_ref.frame_id == "frame"
        assert recorder.execution_ref.workflow_call_depth == 3
        context.execution.spawn(graph={"nodes": {}}, inputs={})
        context.execution.await_dependency(ExecutionRef(execution_node_id="dependency"))
        context.execution.fail("failed")
        assert [effect.kind for effect in recorder.snapshot()] == ["spawn_execution", "await", "fail"]


@pytest.mark.parametrize(
    ("condition", "selected_field"),
    [(True, "true_input"), (False, "false_input")],
)
def test_if_invocation_declares_selected_branch_activation_effect(condition: bool, selected_field: str) -> None:
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(source_node_id="if")
    context.execution = ExecutionInterface(context.execution_effects)
    invocation = IfInvocation(id="if", condition=condition, true_input="true", false_input="false")

    result = invocation.invoke_internal_with_effects(context, _services())

    assert result.output.value == ("true" if condition else "false")
    assert len(result.effects) == 1
    effect = result.effects[0]
    assert isinstance(effect, EmitEffect)
    assert effect.token.node_id == "if"
    assert effect.token.field == selected_field
    assert effect.token.value == selected_field
    assert effect.token.token_kind == "activation"


@pytest.mark.parametrize(
    ("index", "expected_effect_kinds"),
    [(0, ["emit"]), (1, ["emit", "close_stream"])],
)
def test_iterate_invocation_declares_ordered_item_stream_effects(index: int, expected_effect_kinds: list[str]) -> None:
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(source_node_id="iterate")
    context.execution = ExecutionInterface(context.execution_effects)
    invocation = IterateInvocation(id="iterate", collection=["first", "last"], index=index)

    result = invocation.invoke_internal_with_effects(context, _services())

    assert result.output.item == invocation.collection[index]
    assert result.output.index == index
    assert result.output.total == len(invocation.collection)
    assert [effect.kind for effect in result.effects] == expected_effect_kinds
    emit = result.effects[0]
    assert isinstance(emit, EmitEffect)
    assert emit.token.node_id == "iterate"
    assert emit.token.field == "item"
    assert emit.token.value == invocation.collection[index]
    assert emit.token.sequence == index
    assert emit.value == invocation.collection[index]
    if index == 1:
        close = result.effects[1]
        assert isinstance(close, CloseStreamEffect)
        assert close.token.node_id == "iterate"
        assert close.token.field == "item"
        assert close.token.token_kind == "stream_end"


def test_for_invocation_declares_frame_scoped_continuation_start_effect() -> None:
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(source_node_id="for", frame_path=(4, 2))
    context.execution = ExecutionInterface(context.execution_effects)
    state = LoopState(values={"count": 3})
    invocation = ForInvocation(id="for", collection=["item"], index=0, state=state)

    result = invocation.invoke_internal_with_effects(context, _services())

    assert result.output.item == "item"
    assert len(result.effects) == 1
    effect = result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    assert effect.operation == "start"
    assert effect.continuation_kind == "for"
    assert effect.execution_ref is not None
    assert effect.execution_ref.node_id == "for"
    assert effect.execution_ref.frame == (4, 2)
    assert effect.payload == {
        "index": 0,
        "total": 1,
        "state": {"values": {"count": 3}},
    }


def test_for_return_invocation_declares_continuation_completion_effect() -> None:
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(source_node_id="return", frame_path=(4, 2))
    context.execution = ExecutionInterface(context.execution_effects)
    state = LoopState(values={"count": 4})
    invocation = ForReturnInvocation(
        id="return",
        output="item",
        state=state,
        continue_condition=False,
    )

    result = invocation.invoke_internal_with_effects(context, _services())

    assert result.output.output == "item"
    assert result.output.state == state
    assert len(result.effects) == 1
    effect = result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    assert effect.operation == "complete"
    assert effect.continuation_kind == "for"
    assert effect.execution_ref is not None
    assert effect.execution_ref.node_id == "return"
    assert effect.execution_ref.frame == (4, 2)
    assert effect.payload == {
        "output": "item",
        "state": {"values": {"count": 4}},
        "continue_condition": False,
    }


def test_graph_state_accepts_for_continuation_effects_without_linkage_tokens() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)

    for_invocation = state.next()
    assert isinstance(for_invocation, ForInvocation)
    for_ref = state.get_execution_ref(for_invocation.id)
    for_context = _context()
    for_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=for_invocation.id,
        frame_path=for_ref.frame.iteration_path,
        state_id=for_ref.state_id,
        frame_id=for_ref.frame.frame_id,
        workflow_call_depth=for_ref.frame.workflow_call_depth,
    )
    for_context.execution = ExecutionInterface(for_context.execution_effects)
    for_result = for_invocation.invoke_internal_with_effects(for_context, _services())
    state.apply(state.get_execution_ref(for_invocation.id, effect_count=len(for_result.effects)), for_result)

    return_invocation = state.next()
    assert return_invocation is not None
    return_ref = state.get_execution_ref(return_invocation.id)
    return_context = _context()
    return_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=return_invocation.id,
        frame_path=return_ref.frame.iteration_path,
        state_id=return_ref.state_id,
        frame_id=return_ref.frame.frame_id,
        workflow_call_depth=return_ref.frame.workflow_call_depth,
    )
    return_context.execution = ExecutionInterface(return_context.execution_effects)
    return_result = return_invocation.invoke_internal_with_effects(return_context, _services())
    state.apply(state.get_execution_ref(return_invocation.id, effect_count=len(return_result.effects)), return_result)

    continuation = next(iter(state._generic_runtime().continuations.values()))
    assert continuation.status == "completed"
    assert all(token.port != "loop_linkage" for token in state.execution_tokens.values())
    assert [effect.kind for effects in state.execution_effects.values() for effect in effects] == [
        "continuation",
        "continuation",
    ]
    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)
    assert dump_execution_state(restored)["execution_effects"] == snapshot["execution_effects"]


def _apply_flat_for_pair(
    collection: list[object], *, apply_return: bool = True
) -> tuple[GraphExecutionState, object, object]:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=collection))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)

    for_invocation = state.next()
    assert isinstance(for_invocation, ForInvocation)
    for_ref = state.get_execution_ref(for_invocation.id)
    for_context = _context()
    for_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=for_invocation.id,
        frame_path=for_ref.frame.iteration_path,
        state_id=for_ref.state_id,
        frame_id=for_ref.frame.frame_id,
        workflow_call_depth=for_ref.frame.workflow_call_depth,
    )
    for_context.execution = ExecutionInterface(for_context.execution_effects)
    for_result = for_invocation.invoke_internal_with_effects(for_context, _services())
    state.apply(state.get_execution_ref(for_invocation.id, effect_count=len(for_result.effects)), for_result)

    return_invocation = state.next()
    assert isinstance(return_invocation, ForReturnInvocation)
    return_ref = state.get_execution_ref(return_invocation.id)
    return_context = _context()
    return_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=return_invocation.id,
        frame_path=return_ref.frame.iteration_path,
        state_id=return_ref.state_id,
        frame_id=return_ref.frame.frame_id,
        workflow_call_depth=return_ref.frame.workflow_call_depth,
    )
    return_context.execution = ExecutionInterface(return_context.execution_effects)
    return_result = return_invocation.invoke_internal_with_effects(return_context, _services())
    if apply_return:
        state.apply(
            state.get_execution_ref(return_invocation.id, effect_count=len(return_result.effects)), return_result
        )
    return state, return_ref, return_result


def test_graph_state_normalizes_for_continuation_result_before_rehydration() -> None:
    state, _return_ref, _return_result = _apply_flat_for_pair([("tuple",)])
    continuation = next(iter(state._generic_runtime().continuations.values()))
    expected_result = {"output": ["tuple"], "state": None, "continue_condition": True}

    assert continuation.result == expected_result
    restored = load_execution_state(dump_execution_state(state))
    restored_continuation = next(iter(restored._generic_runtime().continuations.values()))
    assert restored_continuation.result == expected_result


def test_graph_state_rejects_type_distinct_for_continuation_replay() -> None:
    state, return_ref, return_result = _apply_flat_for_pair([1], apply_return=False)
    continuation_effect = return_result.effects[0]
    assert isinstance(continuation_effect, ContinuationEffect)
    replay_effect = continuation_effect.model_copy(
        deep=True,
        update={"payload": {**continuation_effect.payload, "output": True}},
    )
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="completed continuation has conflicting result"):
        state.apply(return_ref, return_result, [continuation_effect, replay_effect], effect_count=2)

    assert dump_execution_state(state) == before


def test_graph_state_rejects_malformed_for_start_payload_transactionally() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)
    for_invocation = state.next()
    assert isinstance(for_invocation, ForInvocation)
    for_ref = state.get_execution_ref(for_invocation.id)
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=for_invocation.id,
        frame_path=for_ref.frame.iteration_path,
        state_id=for_ref.state_id,
        frame_id=for_ref.frame.frame_id,
        workflow_call_depth=for_ref.frame.workflow_call_depth,
    )
    context.execution = ExecutionInterface(context.execution_effects)
    result = for_invocation.invoke_internal_with_effects(context, _services())
    effect = result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    malformed = effect.model_copy(deep=True, update={"payload": {**effect.payload, "index": 1}})
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="continuation start payload"):
        state.apply(for_ref, result, [malformed], effect_count=1)

    assert dump_execution_state(state) == before


def test_graph_state_rejects_forged_for_output_with_matching_start_effect() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert isinstance(invocation, ForInvocation)
    execution_ref = state.get_execution_ref(invocation.id)
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=invocation.id,
        frame_path=execution_ref.frame.iteration_path,
        state_id=execution_ref.state_id,
        frame_id=execution_ref.frame.frame_id,
        workflow_call_depth=execution_ref.frame.workflow_call_depth,
    )
    context.execution = ExecutionInterface(context.execution_effects)
    result = invocation.invoke_internal_with_effects(context, _services())
    effect = result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    forged_state = LoopState(values={"forged": True})
    forged_output = result.output.model_copy(update={"index": 99, "total": 99, "state": forged_state})
    assert isinstance(forged_output, ForInvocationOutput)
    forged_effect = effect.model_copy(
        deep=True,
        update={
            "payload": {
                "index": 99,
                "total": 99,
                "state": forged_state.model_dump(mode="json"),
            }
        },
    )
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="For output does not match prepared For"):
        state.apply(
            execution_ref,
            InvocationRunResult(output=forged_output, effects=result.effects),
            [forged_effect],
            effect_count=1,
        )

    assert dump_execution_state(state) == before


def test_graph_state_rejects_forged_for_return_output_with_matching_completion_effect() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)
    effect = return_result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    forged_state = LoopState(values={"forged": True})
    forged_output = ForReturnInvocationOutput(output="forged", state=forged_state)
    forged_effect = effect.model_copy(
        deep=True,
        update={
            "payload": {
                "output": "forged",
                "state": forged_state.model_dump(mode="json"),
                "continue_condition": True,
            }
        },
    )
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="ForReturn output does not match prepared ForReturn"):
        state.apply(
            return_ref,
            InvocationRunResult(output=forged_output, effects=return_result.effects),
            [forged_effect],
            effect_count=1,
        )

    assert dump_execution_state(state) == before


def test_graph_state_rejects_forged_for_return_state_with_matching_completion_effect() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)
    effect = return_result.effects[0]
    assert isinstance(effect, ContinuationEffect)
    forged_state = LoopState(values={"forged": True})
    forged_output = return_result.output.model_copy(update={"state": forged_state})
    forged_effect = effect.model_copy(
        deep=True,
        update={"payload": {**effect.payload, "state": forged_state.model_dump(mode="json")}},
    )
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="ForReturn output does not match prepared ForReturn"):
        state.apply(
            return_ref,
            InvocationRunResult(output=forged_output, effects=return_result.effects),
            [forged_effect],
            effect_count=1,
        )

    assert dump_execution_state(state) == before


def test_graph_state_rejects_forged_for_item_with_valid_continuation_fields() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert isinstance(invocation, ForInvocation)
    execution_ref = state.get_execution_ref(invocation.id)
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=invocation.id,
        frame_path=execution_ref.frame.iteration_path,
        state_id=execution_ref.state_id,
        frame_id=execution_ref.frame.frame_id,
        workflow_call_depth=execution_ref.frame.workflow_call_depth,
    )
    context.execution = ExecutionInterface(context.execution_effects)
    result = invocation.invoke_internal_with_effects(context, _services())
    forged_output = result.output.model_copy(update={"item": "forged"})
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="For output item does not match"):
        state.apply(
            execution_ref,
            InvocationRunResult(output=forged_output, effects=result.effects),
            result.effects,
            effect_count=len(result.effects),
        )

    assert dump_execution_state(state) == before


def test_graph_state_rejects_persisted_for_result_tampering() -> None:
    state, _return_ref, _return_result = _apply_flat_for_pair(["item"])
    for_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "for"
    )
    start_ref = state.get_execution_ref(for_exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["results"][for_exec_id]["index"] = 99
    snapshot["results"][for_exec_id]["total"] = 99
    snapshot["results"][for_exec_id]["state"] = {"values": {"forged": True}}
    start_effect = snapshot["execution_effects"][start_ref.reference_id][0]
    start_effect["payload"] = {
        "index": 99,
        "total": 99,
        "state": {"values": {"forged": True}},
    }

    with pytest.raises(ValueError, match="continuation start payload"):
        load_execution_state(snapshot)


def test_graph_state_rejects_persisted_for_result_only_tampering_with_valid_effect() -> None:
    state, _return_ref, _return_result = _apply_flat_for_pair(["item"])
    for_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "for"
    )
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["results"][for_exec_id]["item"] = "forged"

    with pytest.raises(ValueError, match="For output item does not match prepared For"):
        load_execution_state(snapshot)


def test_graph_state_rejects_effect_for_pending_for_return() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["execution_effects"][return_ref.reference_id] = [return_result.effects[0].model_dump(mode="json")]

    with pytest.raises(ValueError, match="pending execution node"):
        load_execution_state(snapshot)


def test_graph_state_rejects_for_effect_in_wrong_persisted_bucket() -> None:
    state, return_ref, _return_result = _apply_flat_for_pair(["item"], apply_return=False)
    for_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "for"
    )
    for_ref = state.get_execution_ref(for_exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["execution_effects"][return_ref.reference_id] = snapshot["execution_effects"].pop(for_ref.reference_id)

    with pytest.raises(ValueError, match="pending execution node"):
        load_execution_state(snapshot)


def test_graph_state_rejects_nested_execution_owner_tampering() -> None:
    state, return_ref, _return_result = _apply_flat_for_pair(["item"], apply_return=False)
    for_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "for"
    )
    for_ref = state.get_execution_ref(for_exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    effect = snapshot["execution_effects"][for_ref.reference_id][0]
    effect["execution_ref"]["execution_node_id"] = return_ref.exec_node_id

    with pytest.raises(ValueError, match="Execution effect is not owned by execution reference"):
        load_execution_state(snapshot)


def test_graph_state_rejects_missing_for_start_effect_transactionally() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert isinstance(invocation, ForInvocation)
    execution_ref = state.get_execution_ref(invocation.id)
    context = _context()
    context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=invocation.id,
        frame_path=execution_ref.frame.iteration_path,
        state_id=execution_ref.state_id,
        frame_id=execution_ref.frame.frame_id,
        workflow_call_depth=execution_ref.frame.workflow_call_depth,
    )
    context.execution = ExecutionInterface(context.execution_effects)
    result = invocation.invoke_internal_with_effects(context, _services())
    before = dump_execution_state(state)

    with pytest.raises(ValueError, match="must include a continuation effect"):
        state.apply(execution_ref, result, [], effect_count=0)

    assert dump_execution_state(state) == before


def test_graph_state_requires_for_return_continuation_effect() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)

    with pytest.raises(ValueError, match="must include a continuation effect"):
        state.apply(return_ref, return_result, [], effect_count=0)


def test_graph_state_applies_direct_for_return_output() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)

    state.apply(return_ref, return_result)

    assert state.is_complete()


def test_graph_state_applies_direct_for_return_output_with_effects() -> None:
    state, return_ref, return_result = _apply_flat_for_pair(["item"], apply_return=False)

    state.apply(
        return_ref,
        return_result.output,
        return_result.effects,
        effect_count=len(return_result.effects),
    )

    assert state.is_complete()


@pytest.mark.parametrize("conflicting", [False, True], ids=["exact-replay", "conflicting-payload"])
def test_graph_state_reconciles_duplicate_flat_for_continuation_effects(conflicting: bool) -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)

    for_invocation = state.next()
    assert isinstance(for_invocation, ForInvocation)
    for_ref = state.get_execution_ref(for_invocation.id)
    for_context = _context()
    for_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=for_invocation.id,
        frame_path=for_ref.frame.iteration_path,
        state_id=for_ref.state_id,
        frame_id=for_ref.frame.frame_id,
        workflow_call_depth=for_ref.frame.workflow_call_depth,
    )
    for_context.execution = ExecutionInterface(for_context.execution_effects)
    for_result = for_invocation.invoke_internal_with_effects(for_context, _services())
    state.apply(state.get_execution_ref(for_invocation.id, effect_count=len(for_result.effects)), for_result)

    return_invocation = state.next()
    assert isinstance(return_invocation, ForReturnInvocation)
    return_ref = state.get_execution_ref(return_invocation.id)
    return_context = _context()
    return_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=return_invocation.id,
        frame_path=return_ref.frame.iteration_path,
        state_id=return_ref.state_id,
        frame_id=return_ref.frame.frame_id,
        workflow_call_depth=return_ref.frame.workflow_call_depth,
    )
    return_context.execution = ExecutionInterface(return_context.execution_effects)
    return_result = return_invocation.invoke_internal_with_effects(return_context, _services())
    continuation_effect = return_result.effects[0]
    assert isinstance(continuation_effect, ContinuationEffect)
    replay_effect = continuation_effect
    if conflicting:
        replay_effect = continuation_effect.model_copy(
            deep=True,
            update={"payload": {**continuation_effect.payload, "output": "conflict"}},
        )

    before = dump_execution_state(state)
    if conflicting:
        with pytest.raises(ValueError, match="completed continuation has conflicting result"):
            state.apply(
                return_ref,
                return_result,
                [continuation_effect, replay_effect],
                effect_count=2,
            )
        assert dump_execution_state(state) == before
    else:
        state.apply(
            return_ref,
            return_result,
            [continuation_effect, replay_effect],
            effect_count=2,
        )
        assert len(state._generic_runtime().continuations) == 1
        assert state.execution_effects[return_ref.reference_id] == [continuation_effect]


@pytest.mark.parametrize(
    ("owner_kwargs", "message"),
    [
        ({"frame_path": (99,)}, "another execution frame"),
        ({"state_id": "other"}, "another graph execution state"),
        ({"workflow_call_depth": 1}, "another workflow-call depth"),
    ],
)
def test_graph_state_rejects_continuation_effect_from_another_scope(
    owner_kwargs: dict[str, object], message: str
) -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert isinstance(invocation, ForInvocation)
    execution_ref = state.get_execution_ref(invocation.id)
    owner_identity: dict[str, object] = {
        "state_id": execution_ref.state_id,
        "frame_id": execution_ref.frame.frame_id,
        "frame_path": execution_ref.frame.iteration_path,
        "workflow_call_depth": execution_ref.frame.workflow_call_depth,
    }
    owner_identity.update(owner_kwargs)

    invalid_effect = ContinuationEffect(
        execution_ref=ExecutionRef(execution_node_id=invocation.id, **owner_identity),  # type: ignore[arg-type]
        operation="start",
        continuation_kind="for",
        payload={"index": 0, "total": 1, "state": {"values": {}}},
    )
    with pytest.raises(ValueError, match=message):
        state.apply(execution_ref, invocation.invoke(MagicMock()), [invalid_effect])

    assert not state.execution_effects


@pytest.mark.parametrize("condition", [True, False])
def test_graph_state_applies_if_activation_and_releases_selected_branch(condition: bool) -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=condition, true_input=1, false_input=2))
    graph.add_node(ExecutionEffectsTestInvocation(id="successor"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="if", field="value"),
            destination=EdgeConnection(node_id="successor", field="value"),
        )
    )
    state = GraphExecutionState(graph=graph)
    executed_source_ids: list[str] = []
    while True:
        invocation = state.next()
        if invocation is None:
            break
        executed_source_ids.append(state.prepared_source_mapping[invocation.id])
        execution_ref = state.get_execution_ref(invocation.id)
        if isinstance(invocation, IfInvocation):
            context = _context()
            context.execution_effects = ExecutionEffectsRecorder(
                source_node_id=invocation.id,
                frame_path=execution_ref.frame.iteration_path,
            )
            context.execution = ExecutionInterface(context.execution_effects)
            run_result = invocation.invoke_internal_with_effects(context, _services())
            execution_ref = state.get_execution_ref(invocation.id, effect_count=len(run_result.effects))
            state.apply(execution_ref, run_result)
        else:
            state.complete(invocation.id, invocation.invoke(context=MagicMock()))

    assert executed_source_ids == ["if", "successor"]
    assert "successor" in executed_source_ids
    activation_tokens = [token for token in state.execution_tokens.values() if token.token_kind == "activation"]
    assert len(activation_tokens) == 1
    assert activation_tokens[0].port == ("true_input" if condition else "false_input")
    if_exec_id = next(iter(state.source_prepared_mapping["if"]))
    if_execution_ref = state.get_execution_ref(if_exec_id)
    persisted_effects = state.execution_effects[if_execution_ref.reference_id]
    assert len(persisted_effects) == 1
    assert isinstance(persisted_effects[0], EmitEffect)
    assert persisted_effects[0].token.token_kind == "activation"
    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)
    assert (
        dump_execution_state(restored)["execution_effects"][if_execution_ref.reference_id]
        == snapshot["execution_effects"][if_execution_ref.reference_id]
    )
    assert not any(stream.owner_id == if_exec_id for stream in state._generic_runtime().streams.values())


def test_graph_state_rejects_unknown_if_activation_port() -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=True, true_input="true", false_input="false"))
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert invocation is not None
    execution_ref = state.get_execution_ref(invocation.id)
    tokens_before = state.execution_tokens.copy()
    invalid_effect = EmitEffect(
        token=ExecutionToken(
            node_id=invocation.id,
            field="bogus",
            value="bogus",
            token_kind="activation",
        ),
        value="bogus",
    )

    with pytest.raises(ValueError, match="unknown activation port"):
        state.apply(
            execution_ref,
            IfInvocationOutput(value="true"),
            effects=[invalid_effect],
        )

    assert state.execution_tokens == tokens_before
    assert not state.execution_effects


def test_graph_state_rejects_unselected_if_activation_port() -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=True, true_input="true", false_input="false"))
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert invocation is not None
    execution_ref = state.get_execution_ref(invocation.id)
    tokens_before = state.execution_tokens.copy()
    invalid_effect = EmitEffect(
        token=ExecutionToken(
            node_id=invocation.id,
            field="false_input",
            value="false_input",
            token_kind="activation",
        ),
        value="false_input",
    )

    with pytest.raises(ValueError, match="resolved If branch"):
        state.apply(
            execution_ref,
            IfInvocationOutput(value="true"),
            effects=[invalid_effect],
        )

    assert state.execution_tokens == tokens_before
    assert not state.execution_effects


def test_graph_state_rejects_if_activation_value_mismatch() -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=True, true_input="true", false_input="false"))
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert invocation is not None
    execution_ref = state.get_execution_ref(invocation.id)
    tokens_before = state.execution_tokens.copy()
    invalid_effect = EmitEffect(
        token=ExecutionToken(
            node_id=invocation.id,
            field="true_input",
            value="false_input",
            token_kind="activation",
        ),
        value="false_input",
    )

    with pytest.raises(ValueError, match="activation value"):
        state.apply(
            execution_ref,
            IfInvocationOutput(value="true"),
            effects=[invalid_effect],
        )

    assert state.execution_tokens == tokens_before
    assert not state.execution_effects


def test_execution_token_and_ref_are_frame_aware() -> None:
    token = ExecutionToken(invocation_id="node", port="value", iteration_path=(2, 4))
    ref = ExecutionRef(token=token, scope="iteration")

    assert token.node_id == "node"
    assert token.field == "value"
    assert token.frame == (2, 4)
    assert ref.invocation_id == "node"
    assert ref.iteration_path == (2, 4)

    with pytest.raises(ValidationError, match="conflicts"):
        ExecutionRef(token=token, execution_node_id="other")
    with pytest.raises(ValidationError, match="conflicts"):
        ExecutionRef(token=token, frame_path=(9,))


def test_execution_ref_identity_aliases_are_safe_without_token() -> None:
    ref = ExecutionRef(execution_node_id="node")

    assert ref.invocation_id == "node"
    assert ref.output_name == ""


@pytest.mark.parametrize(
    "alias, value",
    [
        ("node_id", "other"),
        ("invocation_id", "other"),
        ("field", "other"),
        ("port", "other"),
        ("output", "other"),
        ("output_name", "other"),
        ("frame", (9,)),
        ("iteration_path", (9,)),
    ],
)
def test_execution_ref_rejects_conflicting_legacy_alias_with_token(alias: str, value: object) -> None:
    token = ExecutionToken(node_id="node", field="value", frame=(2,))

    with pytest.raises(ValidationError, match="conflicts"):
        ExecutionRef(token=token, **{alias: value})


def test_execution_interface_rejects_unsupported_lifecycle_effects() -> None:
    recorder = ExecutionEffectsRecorder(source_node_id="parent")
    execution = ExecutionInterface(recorder)

    with pytest.raises(UnsupportedExecutionEffectError, match="not supported"):
        execution.spawn(graph={"nodes": {}}, inputs={})
    with pytest.raises(UnsupportedExecutionEffectError, match="not supported"):
        execution.await_dependency(ExecutionRef(execution_node_id="child"))
    with pytest.raises(UnsupportedExecutionEffectError, match="not supported"):
        execution.fail("failed")

    assert recorder.snapshot() == ()


def test_execution_interface_spawn_is_guarded() -> None:
    recorder = ExecutionEffectsRecorder(source_node_id="parent")
    execution = ExecutionInterface(recorder)

    with pytest.raises(UnsupportedExecutionEffectError):
        execution.spawn(
            graph={"nodes": {}},
            inputs={"value": {"items": [1, True, None]}},
            authorization_context={"user_id": "user"},
        )


def test_capability_enabled_execution_interface_records_child_lifecycle_effects() -> None:
    capability = ChildExecutionCapability(
        parent_execution_id="parent",
        parent_frame=(),
        authorization_context={"user_id": "user"},
    )
    recorder = ExecutionEffectsRecorder(
        source_node_id="parent",
        allow_lifecycle_effects=True,
        child_capability=capability,
    )
    execution = ExecutionInterface(recorder)

    handle = execution.spawn(graph={"nodes": {}}, inputs={})
    execution.await_dependency(ExecutionRef(execution_node_id="dependency"))
    execution.fail("failed")

    assert handle.child_execution_id
    assert handle.parent_execution_id == "parent"
    assert [effect.kind for effect in recorder.snapshot()] == ["spawn_execution", "await", "fail"]
    spawn = recorder.snapshot()[0]
    assert isinstance(spawn, SpawnExecutionEffect)
    assert spawn.child_execution_id == handle.child_execution_id
    assert spawn.authorization_context == {"user_id": "user"}


def test_call_saved_workflow_declares_authorized_spawn_and_await_effects() -> None:
    invocation = CallSavedWorkflowInvocation(id="call", workflow_id="workflow")
    capability = ChildExecutionCapability(parent_execution_id="call", parent_frame=())
    workflow_record = SimpleNamespace(workflow=SimpleNamespace(model_dump=lambda **_: {"nodes": {}}))
    data = InvocationContextData(
        queue_item=SimpleNamespace(user_id="user"),
        invocation=invocation,
        source_invocation_id="call",
        execution_state_id="state",
        execution_frame_id="frame",
        execution_child_capability=capability,
        execution_workflow_authorizer=lambda _workflow_id: workflow_record,
        execution_workflow_inputs={"value": 3},
    )
    context = build_invocation_context(SimpleNamespace(), data, lambda: False)

    invocation.invoke(context)

    effects = context.execution_effects.snapshot()
    assert [effect.kind for effect in effects] == ["spawn_execution", "await"]
    assert effects[0].inputs == {"value": 3}
    assert effects[1].dependency.execution_node_id == effects[0].child_execution_id


def test_call_saved_workflow_failure_effect_preserves_empty_message_exception() -> None:
    class EmptyMessageError(Exception):
        pass

    def authorize_workflow(_workflow_id: str) -> object:
        raise EmptyMessageError()

    invocation = CallSavedWorkflowInvocation(id="call", workflow_id="workflow")
    capability = ChildExecutionCapability(parent_execution_id="call", parent_frame=())
    data = InvocationContextData(
        queue_item=SimpleNamespace(user_id="user"),
        invocation=invocation,
        source_invocation_id="call",
        execution_state_id="state",
        execution_frame_id="frame",
        execution_child_capability=capability,
        execution_workflow_authorizer=authorize_workflow,
    )
    context = build_invocation_context(SimpleNamespace(), data, lambda: False)

    invocation.invoke(context)

    effects = context.execution_effects.snapshot()
    assert len(effects) == 1
    effect = effects[0]
    assert isinstance(effect, FailEffect)
    assert effect.message == ""
    assert effect.error_type == "EmptyMessageError"
    assert "EmptyMessageError" in (effect.error_traceback or "")


def test_capability_enabled_execution_interface_rejects_wrong_parent_scope() -> None:
    capability = ChildExecutionCapability(parent_execution_id="other", parent_frame=())
    recorder = ExecutionEffectsRecorder(
        source_node_id="parent",
        allow_lifecycle_effects=True,
        child_capability=capability,
    )

    with pytest.raises(PermissionError, match="another execution"):
        ExecutionInterface(recorder).spawn(graph={"nodes": {}}, inputs={})


def test_lifecycle_effects_require_a_child_capability() -> None:
    execution = ExecutionInterface(ExecutionEffectsRecorder(allow_lifecycle_effects=True))

    with pytest.raises(PermissionError, match="unavailable"):
        execution.await_dependency(ExecutionRef(execution_node_id="dependency"))
    with pytest.raises(PermissionError, match="unavailable"):
        execution.fail("failed")


@pytest.mark.parametrize(
    "effect",
    [
        SetValueEffect(target=ExecutionRef(node_id="node", field="value"), value=3),
        AddEdgeEffect(
            source=ExecutionRef(node_id="source", field="value"),
            destination=ExecutionRef(node_id="destination", field="input"),
        ),
        RemoveEdgeEffect(
            source=ExecutionRef(node_id="source", field="value"),
            destination=ExecutionRef(node_id="destination", field="input"),
        ),
    ],
)
def test_recorder_rejects_effects_without_graph_dispatch(effect: object) -> None:
    recorder = ExecutionEffectsRecorder(source_node_id="node")

    with pytest.raises(UnsupportedExecutionEffectError, match="not supported"):
        recorder.record(effect)  # type: ignore[arg-type]

    assert recorder.snapshot() == ()


def test_execution_interface_preserves_explicit_empty_frame() -> None:
    recorder = ExecutionEffectsRecorder(source_node_id="node", frame_path=(2,))
    execution = ExecutionInterface(recorder)

    execution.emit("value", 1, frame=())
    execution.close_stream("value", frame=())

    emit_effect, close_effect = recorder.snapshot()
    assert isinstance(emit_effect, EmitEffect)
    assert isinstance(close_effect, CloseStreamEffect)
    assert emit_effect.token.frame == ()
    assert close_effect.token.frame == ()


@pytest.mark.parametrize(
    "value",
    [
        {"child_execution_id": "", "parent_execution_id": "parent"},
        {"child_execution_id": "child", "parent_execution_id": ""},
        {"child_execution_id": "child", "parent_execution_id": "parent", "authorization_context": ""},
    ],
)
def test_child_execution_handle_rejects_invalid_identity_or_authorization(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChildExecutionHandle.model_validate(value)


@pytest.mark.parametrize(
    "graph, inputs",
    [
        (None, {}),
        ({"nodes": {}}, []),
        ({"nodes": {}}, {"": 1}),
    ],
)
def test_spawn_rejects_invalid_inputs(graph: object, inputs: object) -> None:
    with pytest.raises((TypeError, ValidationError)):
        SpawnExecutionEffect(
            parent=ExecutionRef(execution_node_id="parent"),
            graph=graph,
            inputs=inputs,  # type: ignore[arg-type]
            child_execution_id="child",
        )


def test_recorder_drain_returns_and_clears_effects() -> None:
    recorder = ExecutionEffectsRecorder()
    effect = EmitEffect(token=ExecutionToken(node_id="node", field="value", value=3), value=3)
    recorder.record(effect)

    assert recorder.drain() == (effect,)
    assert recorder.snapshot() == ()


def test_effect_values_accept_json_serializable_values_and_reject_runtime_values() -> None:
    target = ExecutionRef(node_id="node", field="value")
    nested_value = {"items": [1, True, None, {"name": "value"}]}

    assert SetValueEffect(target=target, value=nested_value).value == nested_value
    with pytest.raises(ValidationError, match="JSON-serializable"):
        SetValueEffect(target=target, value=object())
    with pytest.raises(ValidationError, match="JSON-serializable"):
        EmitEffect(token=ExecutionToken(node_id="node", field="value"), value=object())
    with pytest.raises(ValidationError, match="JSON-serializable"):
        ExecutionToken(node_id="node", field="value", value=object())
    with pytest.raises(ValidationError, match="JSON-serializable"):
        SpawnExecutionEffect(parent=ExecutionRef(execution_node_id="node"), graph=object(), child_execution_id="child")
    with pytest.raises(ValidationError, match="JSON-serializable"):
        SpawnExecutionEffect(
            parent=ExecutionRef(execution_node_id="node"),
            graph={},
            inputs={"value": object()},
            child_execution_id="child",
        )
    with pytest.raises(ValidationError, match="JSON-serializable"):
        SpawnExecutionEffect(
            parent=ExecutionRef(execution_node_id="node"),
            graph={},
            authorization_context={"user": object()},
            child_execution_id="child",
        )


def test_effect_values_survive_execution_state_json_dump() -> None:
    effect = SetValueEffect(
        target=ExecutionRef(node_id="node", field="value"),
        value={"items": [1, True, None, {"name": "value"}]},
    )
    state = GraphExecutionState(graph=Graph(), execution_effects={"reference": [effect]})

    snapshot = dump_execution_state(state)

    assert snapshot["execution_effects"]["reference"][0]["value"] == effect.value


def test_saved_workflow_lifecycle_effects_survive_json_round_trip() -> None:
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="saved-workflow", workflow_id="workflow"))
    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert node is not None
    parent = ExecutionRef(execution_node_id=node.id)
    effects = [
        SpawnExecutionEffect(
            execution_ref=parent,
            parent=parent,
            graph={"nodes": {"output": {"type": "workflow_return"}}, "saved_workflow_id": "workflow"},
            inputs={"prompt": "value"},
            child_execution_id="child",
            authorization_context={"user_id": "user"},
        ),
        AwaitEffect(execution_ref=parent, dependency=ExecutionRef(execution_node_id="child")),
    ]
    ref = state.get_execution_ref(node.id, effect_count=len(effects))
    state.results[node.id] = WorkflowReturnOutput(values={})
    state.executed.add(node.id)
    state.execution_effects[ref.reference_id] = effects
    state.execution_refs[node.id] = ref

    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)

    assert snapshot["execution_effects"][ref.reference_id] == [effect.model_dump(mode="json") for effect in effects]
    assert dump_execution_state(restored)["execution_effects"] == snapshot["execution_effects"]


def test_failed_pending_workflow_call_is_terminal_after_json_round_trip() -> None:
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="saved-workflow", workflow_id="workflow"))
    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert node is not None
    parent = ExecutionRef(execution_node_id=node.id)
    effects = [
        SpawnExecutionEffect(
            execution_ref=parent,
            parent=parent,
            graph={"nodes": {}},
            inputs={},
            child_execution_id="child",
        ),
        AwaitEffect(execution_ref=parent, dependency=ExecutionRef(execution_node_id="child")),
    ]
    ref = state.get_execution_ref(node.id, effect_count=len(effects))
    state.execution_refs[node.id] = ref
    state.execution_effects[ref.reference_id] = effects
    state.errors[node.id] = "child failed"

    restored = load_execution_state(dump_execution_state(state))

    assert restored.is_complete()
    assert restored.next() is None


def test_set_value_effect_requires_a_value() -> None:
    with pytest.raises(ValidationError):
        SetValueEffect(target=ExecutionRef(node_id="node", field="value"))


def test_lifecycle_effect_models_validate_owner_refs() -> None:
    await_effect = AwaitEffect(
        execution_ref=ExecutionRef(execution_node_id="parent"),
        dependency=ExecutionRef(execution_node_id="child"),
    )
    fail_effect = FailEffect(execution_ref=ExecutionRef(execution_node_id="parent"), message="failed")
    assert isinstance(await_effect, AwaitEffect)
    assert isinstance(fail_effect, FailEffect)


@pytest.mark.parametrize(
    "effect",
    [
        SpawnExecutionEffect(
            parent=ExecutionRef(execution_node_id="parent"),
            graph={},
            child_execution_id="child",
        ),
        AwaitEffect(
            execution_ref=ExecutionRef(execution_node_id="parent"),
            dependency=ExecutionRef(execution_node_id="child"),
        ),
        FailEffect(execution_ref=ExecutionRef(execution_node_id="parent"), message="failed"),
    ],
)
def test_recorder_rejects_unsupported_lifecycle_effects(effect: object) -> None:
    recorder = ExecutionEffectsRecorder(source_node_id="parent")

    with pytest.raises(UnsupportedExecutionEffectError, match="not supported"):
        recorder.record(effect)  # type: ignore[arg-type]

    assert recorder.snapshot() == ()


@pytest.mark.parametrize(
    "value",
    [
        {"node_id": "", "field": "value"},
        {"node_id": "node", "field": ""},
        {"node_id": "node", "field": "value", "frame": (-1,)},
        {"node_id": "node", "field": "value", "frame": (1,), "iteration_path": (2,)},
    ],
)
def test_execution_token_rejects_malformed_identity(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExecutionToken.model_validate(value)


def test_effect_models_validate_typed_refs() -> None:
    source = ExecutionRef(node_id="source", field="value")
    destination = ExecutionRef(node_id="destination", field="input")

    effect = AddEdgeEffect(source=source, destination=destination)
    assert effect.kind == "add_edge"

    with pytest.raises(ValidationError):
        AddEdgeEffect(source=source, destination={"node_id": "destination"})


def test_recorder_rejects_untyped_values_and_returns_copy() -> None:
    recorder = ExecutionEffectsRecorder()
    effect = EmitEffect(
        token=ExecutionToken(node_id="node", field="value", value=3),
        value=3,
    )

    recorder.record(effect)
    effects = recorder.snapshot()

    assert effects == (effect,)
    with pytest.raises(TypeError, match="Expected ExecutionEffect"):
        recorder.record("not an effect")  # type: ignore[arg-type]

    invalid_effect = effect.model_copy(update={"value": object()})
    with pytest.raises(ValueError, match="JSON-serializable"):
        recorder.record(invalid_effect)
    assert recorder.snapshot() == (effect,)


def test_effectful_invoke_bypasses_output_only_cache() -> None:
    ExecutionEffectsTestInvocation.calls = 0
    services = _services()
    cached_output = ExecutionEffectsTestOutput(value=99)
    services.invocation_cache.get.return_value = cached_output
    invocation_instance = ExecutionEffectsTestInvocation(id="node", value=7, use_cache=True)

    ordinary_output = invocation_instance.invoke_internal(_context(), services)
    assert ordinary_output == cached_output
    assert ExecutionEffectsTestInvocation.calls == 0
    services.invocation_cache.get.assert_called_once()

    services.invocation_cache.reset_mock()
    result = invocation_instance.invoke_internal_with_effects(_context(), services)

    assert result.output == ExecutionEffectsTestOutput(value=7)
    assert len(result.effects) == 1
    assert result.effects[0].kind == "emit"
    assert ExecutionEffectsTestInvocation.calls == 1
    services.invocation_cache.get.assert_not_called()
    services.invocation_cache.save.assert_not_called()


def test_effectful_invoke_clears_stale_effects() -> None:
    context = _context()
    context.effects.record(EmitEffect(token=ExecutionToken(node_id="stale", field="value", value=0), value=0))
    result = ExecutionEffectsTestInvocation(id="node", value=2).invoke_internal_with_effects(context, _services())

    assert [effect.value for effect in result.effects if isinstance(effect, EmitEffect)] == [2]


def test_effectful_invoke_disables_cache_in_invoke_internal_override() -> None:
    ExecutionEffectsOverrideInvocation.calls = 0
    services = _services()
    services.invocation_cache.get.return_value = ExecutionEffectsTestOutput(value=99)
    invocation_instance = ExecutionEffectsOverrideInvocation(id="node", value=7, use_cache=True)

    result = invocation_instance.invoke_internal_with_effects(_context(), services)

    assert result.output == ExecutionEffectsTestOutput(value=7)
    assert len(result.effects) == 1
    assert ExecutionEffectsOverrideInvocation.calls == 1
    assert invocation_instance.use_cache is True
    services.invocation_cache.get.assert_not_called()
