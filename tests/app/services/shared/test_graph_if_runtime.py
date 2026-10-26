from types import SimpleNamespace

import pytest

from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.logic import BooleanLogicInvocation, IfInvocation
from invokeai.app.invocations.loops import ForInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import BooleanOutput
from invokeai.app.services.shared import graph_if_runtime
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from invokeai.app.services.shared.graph_iterator_ancestry import _IteratorAncestry
from invokeai.app.services.shared.graph_runtime_records import _ApplyTransaction
from invokeai.app.services.shared.graph_validation import CollectInvocation, IterateInvocation
from tests.app.services.shared.import_test_utils import assert_module_imports_without_graph
from tests.test_nodes import AnyTypeTestInvocation


@pytest.fixture
def state() -> GraphExecutionState:
    state = GraphExecutionState(graph=Graph(nodes={"if": IfInvocation(id="if")}))
    state.execution_graph.add_node(IfInvocation(id="if-exec"))
    state._prepared_registry().register("if-exec", "if")
    state._prepared_registry().set_iteration_path("if-exec", (2,))
    return state


@pytest.mark.platform_independent
def test_runtime_import_without_graph() -> None:
    assert_module_imports_without_graph("invokeai.app.services.shared.graph_if_runtime")


@pytest.mark.parametrize(
    ("source_kind", "source_field", "axes", "expected"),
    [
        ("regular", "value", ("outer", "inner"), ("outer",)),
        ("regular", "value", (), ()),
        ("for_final", "output_collection", ("outer",), ("outer",)),
    ],
)
def test_collect_item_axis_closure_handles_empty_axes_and_for_final_output(
    source_kind: str, source_field: str, axes: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    source = IfInvocation(id="source", condition=True) if source_kind == "regular" else ForInvocation(id="source")
    graph = Graph()
    graph.nodes.update({"source": source, "collect": CollectInvocation(id="collect")})
    state = SimpleNamespace(graph=graph)
    edge = Edge(
        source=EdgeConnection(node_id="source", field=source_field),
        destination=EdgeConnection(node_id="collect", field="item"),
    )

    assert graph_if_runtime._without_closed_axis(state, edge, axes) == expected


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
    state._index_activation_dependencies("consumer", (dependency,))
    assert state._if_activation_dependents_by_dependency[dependency] == ("consumer",)
    assert "consumer" not in state._if_activation_active_dependents_by_dependency.get(dependency, {})

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


def test_nested_if_iteration_paths_use_supplied_source_graph_recursively() -> None:
    state = GraphExecutionState(
        graph=Graph(
            nodes={
                "if": IfInvocation(id="if", condition=True),
                "inner": IfInvocation(id="inner", condition=True),
                "guarded": IfInvocation(id="guarded"),
                "condition": BooleanLogicInvocation(id="condition", a=True, b=True),
                "bridge": AddInvocation(id="bridge"),
            }
        )
    )
    state.execution_graph.add_node(BooleanLogicInvocation(id="condition-exec", a=True, b=True))
    state._prepared_registry().register("condition-exec", "condition")
    state.results["condition-exec"] = BooleanOutput(value=True)
    state.graph.add_node(IterateInvocation(id="iterator"))
    state.graph.edges.extend(
        [
            Edge(
                source=EdgeConnection(node_id="iterator", field="item"),
                destination=EdgeConnection(node_id="guarded", field="condition"),
            ),
            Edge(
                source=EdgeConnection(node_id="guarded", field="value"),
                destination=EdgeConnection(node_id="inner", field="true_input"),
            ),
            Edge(
                source=EdgeConnection(node_id="inner", field="value"),
                destination=EdgeConnection(node_id="bridge", field="a"),
            ),
            Edge(
                source=EdgeConnection(node_id="bridge", field="value"),
                destination=EdgeConnection(node_id="if", field="true_input"),
            ),
            Edge(
                source=EdgeConnection(node_id="condition", field="value"),
                destination=EdgeConnection(node_id="if", field="condition"),
            ),
        ]
    )
    state.graph._invalidate_edge_indexes()
    source_graph = state._get_source_graph_flat()
    constrained_graph = source_graph.copy()
    constrained_graph.remove_edge("iterator", "guarded")

    assert (
        graph_if_runtime._get_if_iteration_paths_with_ancestry(
            state,
            "if",
            set(),
            lambda source_id: {"condition-exec"} if source_id == "condition" else set(),
            lambda _edge, _prepared_id: (),
            _IteratorAncestry(state, source_graph),
        )
        == []
    )
    assert graph_if_runtime._get_if_iteration_paths_with_ancestry(
        state,
        "if",
        set(),
        lambda source_id: {"condition-exec"} if source_id == "condition" else set(),
        lambda _edge, _prepared_id: (),
        _IteratorAncestry(state, constrained_graph),
    ) == [()]


def test_unresolved_nested_literal_if_chain_has_no_condition_path() -> None:
    nodes = {"iterator": IterateInvocation(id="iterator"), "guarded": IfInvocation(id="guarded")}
    nodes.update({f"if-{i}": IfInvocation(id=f"if-{i}", condition=True) for i in range(24)})
    edges = [
        Edge(
            source=EdgeConnection(node_id="iterator", field="item"),
            destination=EdgeConnection(node_id="guarded", field="condition"),
        )
    ]
    previous = "guarded"
    for i in range(24):
        edges.append(
            Edge(
                source=EdgeConnection(node_id=previous, field="value"),
                destination=EdgeConnection(node_id=f"if-{i}", field="true_input"),
            )
        )
        previous = f"if-{i}"
    state = GraphExecutionState(graph=Graph())
    state.graph = Graph(nodes=nodes, edges=edges)
    assert (
        graph_if_runtime._get_if_iteration_paths_with_ancestry(
            state, previous, set(), lambda _: set(), lambda *_: (), _IteratorAncestry(state)
        )
        == []
    )


def test_resolved_literal_if_uses_root_frame_path() -> None:
    state = GraphExecutionState(graph=Graph(nodes={"if": IfInvocation(id="if", condition=True, true_input=42)}))
    assert graph_if_runtime._get_if_iteration_paths_with_ancestry(
        state, "if", set(), lambda _: set(), lambda *_: (), _IteratorAncestry(state)
    ) == [()]


def test_if_branch_frame_axes_deduplicates_stably_with_linear_comparisons(monkeypatch: pytest.MonkeyPatch) -> None:
    comparisons = 0

    class Axis(str):
        def __eq__(self, other: object) -> bool:
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

        __hash__ = str.__hash__

    axes = [Axis(f"axis-{index}") for index in range(1000)]
    graph = Graph(nodes={"if": IfInvocation(id="if")})
    for index in range(2):
        source_id = f"source{index}"
        graph.add_node(AddInvocation(id=source_id))
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source_id, field="value"),
                destination=EdgeConnection(node_id="if", field="true_input" if index == 0 else "false_input"),
            )
        )
    state = SimpleNamespace(graph=graph)
    axis_tuple = tuple(axes)
    monkeypatch.setattr(graph_if_runtime, "_get_edge_frame_axes", lambda _state, _ancestry, edge, _cache: axis_tuple)

    result = graph_if_runtime._get_if_branch_frame_axes(
        state, _IteratorAncestry(GraphExecutionState(graph=Graph())), "if", {}
    )

    assert tuple(result["true_input"]) == tuple(f"axis-{index}" for index in range(1000))
    assert result["true_input"] == result["false_input"]
    assert comparisons < 10_000


def test_edge_frame_axis_traversal_visits_long_passthrough_chain_linearly(monkeypatch: pytest.MonkeyPatch) -> None:
    chain_length = 1000
    graph = Graph()
    graph.nodes["source"] = CollectionConcatInvocation(id="source", first=["x"])
    graph.nodes["iterate"] = IterateInvocation(id="iterate")
    graph.nodes["if"] = IfInvocation(id="if")
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="collection"),
            destination=EdgeConnection(node_id="iterate", field="collection"),
        )
    )
    previous = ("iterate", "item")
    for index in range(chain_length):
        node_id = f"pass{index}"
        graph.nodes[node_id] = AnyTypeTestInvocation(id=node_id)
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=previous[0], field=previous[1]),
                destination=EdgeConnection(node_id=node_id, field="value"),
            )
        )
        previous = (node_id, "value")
    edge = Edge(
        source=EdgeConnection(node_id=previous[0], field=previous[1]),
        destination=EdgeConnection(node_id="if", field="true_input"),
    )
    graph.add_edge(edge)
    state = GraphExecutionState(graph=graph)
    get_input_edges = graph._get_input_edges
    visits = 0

    def counted_get_input_edges(node_id: str, *args: object, **kwargs: object) -> list[Edge]:
        nonlocal visits
        visits += 1
        return get_input_edges(node_id, *args, **kwargs)

    monkeypatch.setattr(graph, "_get_input_edges", counted_get_input_edges)

    axes = graph_if_runtime._get_edge_frame_axes(state, _IteratorAncestry(state), edge, {})

    assert axes == ("iterate",)
    assert visits < chain_length * 3


@pytest.mark.parametrize("executed", [False, True])
def test_prepared_selected_source_preserves_paths_without_ancestry(executed: bool) -> None:
    state = GraphExecutionState(
        graph=Graph(nodes={"if": IfInvocation(id="if", condition=True), "source": AddInvocation(id="source")})
    )
    state.graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    state.source_prepared_mapping["source"] = set() if executed else {"source-exec"}
    if executed:
        state.executed.add("source")

    assert graph_if_runtime._get_if_iteration_paths_with_ancestry(
        state,
        "if",
        set(),
        lambda source_id: state.source_prepared_mapping[source_id],
        lambda *_: (2,),
        _IteratorAncestry(state),
    ) == ([] if executed else [(2,)])


def test_collect_empty_path_fallback_reuses_input_edge_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph(
        nodes={
            "outer": AddInvocation(id="outer"),
            "source": AddInvocation(id="source"),
            "collect": CollectInvocation(id="collect"),
        }
    )
    graph.edges.append(
        Edge(
            source=EdgeConnection(node_id="outer", field="item"),
            destination=EdgeConnection(node_id="source", field="a"),
        )
    )
    input_edge = Edge(
        source=EdgeConnection(node_id="source", field="output_collection"),
        destination=EdgeConnection(node_id="collect", field="item"),
    )
    graph.edges.append(input_edge)
    graph.edges.append(
        Edge(
            source=EdgeConnection(node_id="outer", field="item"),
            destination=EdgeConnection(node_id="consumer", field="value"),
        )
    )
    graph._invalidate_edge_indexes()
    state = SimpleNamespace(graph=graph, source_prepared_mapping={"outer"})
    axis_lookups = 0

    def get_edge_frame_axes(_state: object, _ancestry: object, edge: Edge, _cache: dict) -> tuple[str, ...]:
        nonlocal axis_lookups
        assert edge == input_edge
        axis_lookups += 1
        return ("outer", "inner")

    monkeypatch.setattr(graph_if_runtime, "_get_edge_frame_axes", get_edge_frame_axes)

    assert graph_if_runtime._get_effective_iteration_paths_for_edge(
        state,
        Edge(
            source=EdgeConnection(node_id="collect", field="collection"),
            destination=EdgeConnection(node_id="consumer", field="value"),
        ),
        {"source"},
        lambda _source_id: {"outer-exec"},
        lambda _edge, _prepared_id: (),
        _IteratorAncestry(GraphExecutionState(graph=Graph())),
    ) == {()}
    assert axis_lookups == 1
