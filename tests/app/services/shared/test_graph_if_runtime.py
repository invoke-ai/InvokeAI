import pytest

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from invokeai.app.services.shared.graph_runtime_records import _ApplyTransaction
from tests.app.services.shared.import_test_utils import assert_module_imports_without_graph


@pytest.fixture
def state() -> GraphExecutionState:
    state = GraphExecutionState(graph=Graph(nodes={"if": IfInvocation(id="if")}))
    state.execution_graph.add_node(IfInvocation(id="if-exec"))
    state._prepared_registry().register("if-exec", "if")
    state._prepared_registry().set_iteration_path("if-exec", (2,))
    return state


def test_runtime_import_without_graph() -> None:
    assert_module_imports_without_graph("invokeai.app.services.shared.graph_if_runtime")


@pytest.mark.parametrize(
    "field,value",
    [("state_id", "foreign"), ("frame_id", "foreign"), ("iteration_path", (3,)), ("workflow_call_depth", 1)],
)
@pytest.mark.parametrize("target", ["gate", "token"])
def test_live_activation_frame_mismatch(state: GraphExecutionState, field: str, value: object, target: str) -> None:
    dependency = ActivationDependency(owner_id="if", branch="true_input", frame=(2,))
    state._resolve_activation_gate("if-exec", "true_input")
    state._record_compatibility_activation_token("if-exec", "true_input")
    assert state._is_activation_dependency_satisfied(dependency)

    if target == "gate":
        gate = state._activation_gate("if-exec")
        state._generic_runtime().replace_gate(
            gate.model_copy(update={"frame": gate.frame.model_copy(update={field: value})})
        )
    else:
        token_id, token = next(iter(state.execution_tokens.items()))
        state.execution_tokens[token_id] = token.model_copy(
            update={"frame": token.frame.model_copy(update={field: value})}
        )

    if target == "gate":
        with pytest.raises(ValueError, match="activation gate identity conflict"):
            state._is_activation_dependency_satisfied(dependency)
        with pytest.raises(ValueError, match="activation gate identity conflict"):
            state._is_activation_dependency_rejected(
                ActivationDependency(owner_id="if", branch="false_input", frame=(2,))
            )
    else:
        assert not state._is_activation_dependency_satisfied(dependency)


def test_unresolved_dependency_is_deferred_but_rejected_is_not(state: GraphExecutionState) -> None:
    dependency = ActivationDependency(owner_id="if", branch="false_input", frame=(2,))
    state._if_activation_dependencies_by_exec["consumer"] = (dependency,)
    assert state._is_deferred_by_unresolved_if("consumer")
    assert not state._has_rejected_activation_dependency("consumer")

    state._resolve_activation_gate("if-exec", "true_input")
    assert not state._is_deferred_by_unresolved_if("consumer")
    assert state._has_rejected_activation_dependency("consumer")
    assert not state._is_activation_dependency_satisfied(dependency)


def test_activation_resolution_and_token_are_idempotent_and_rollback(state: GraphExecutionState) -> None:
    gate_before = state._activation_gate("if-exec").model_dump()
    state._completed_source_ids_cache = {"cached"}
    transaction = _ApplyTransaction()
    state._apply_transaction = transaction
    assert state._resolve_activation_gate("if-exec", "true_input")
    assert state._completed_source_ids_cache is None
    state._record_compatibility_activation_token("if-exec", "true_input")
    tokens_before = dict(state.execution_tokens)
    state._completed_source_ids_cache = {"cached-again"}
    assert not state._resolve_activation_gate("if-exec", "true_input")
    assert state._completed_source_ids_cache == {"cached-again"}
    state._record_compatibility_activation_token("if-exec", "true_input")
    assert state.execution_tokens == tokens_before
    assert state._record_activation_dependencies("if-exec") == ()
    assert "if-exec" in state._if_activation_dependencies_by_exec

    transaction.rollback()
    state._apply_transaction = None
    assert state._activation_gate("if-exec").model_dump() == gate_before
    assert state.execution_tokens == {}
    assert "if-exec" not in state._if_activation_dependencies_by_exec


def test_apply_rollback_reconstructs_pending_if_nodes(state: GraphExecutionState) -> None:
    state.graph.add_node(AddInvocation(id="source", a=1, b=2))
    state.graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    state._resolved_if_exec_branches["if-exec"] = "true_input"

    state._reset_apply_derived_caches()

    assert state._pending_if_exec_nodes == {"if-exec"}
