"""Collect grouping must follow the iteration frame of its inputs, whatever depth or wrapper each input arrives through.

Each Collect execution is read separately: several executions at one iteration path is itself the failure, and a dict
keyed by path would hide it. Order across different sources in one Collect is unspecified, so items are compared sorted.
"""

import json
import re
from typing import Any as AnyValue
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.collections import CollectionConcatInvocation as Concat
from invokeai.app.invocations.logic import IfInvocation as If
from invokeai.app.invocations.loops import ForInvocation as For
from invokeai.app.invocations.loops import ForReturnInvocation as ForReturn
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import CollectInvocation as Collect
from invokeai.app.services.shared.graph import (
    Graph,
    GraphExecutionState,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from invokeai.app.services.shared.graph import IterateInvocation as Iterate
from invokeai.app.services.shared.graph_materializer import _ExecutionMaterializer
from tests.test_nodes import AnyTypeTestInvocation as Value
from tests.test_nodes import FixedCollectionTestInvocation as Truthy
from tests.test_nodes import MarkedAnyTypeTestInvocation as Marked
from tests.test_nodes import create_edge, create_loop_linkage

pytestmark = pytest.mark.platform_independent

Runs = list[tuple[tuple[int, ...], list[AnyValue]]]


@pytest.fixture(params=[False, True], ids=["generic-adapter", "compatibility-scheduler"])
def compatibility_scheduler(request: pytest.FixtureRequest) -> bool:
    return request.param


def _graph(src: list[AnyValue], levels: int, nodes: list[BaseInvocation], edges: str) -> Graph:
    """Build `src -> it0 -> cc0 -> it1 -> ... -> it{levels - 1}` plus `nodes` and "a.field -> b.field" `edges` lines."""
    graph = Graph()
    spine: list[BaseInvocation] = [Concat(id="src", first=src), Iterate(id="it0")]
    links = ["src.collection -> it0.collection"]
    for level in range(1, levels):
        spine += [Concat(id=f"cc{level - 1}"), Iterate(id=f"it{level}")]
        links += [f"it{level - 1}.item -> cc{level - 1}.first", f"cc{level - 1}.collection -> it{level}.collection"]
    for node in spine + nodes:
        graph.add_node(node)
    # Edges out of a Collect validate against its `item` inputs, so those go in first.
    parsed = [re.split(r"\s*->\s*|\.", line.strip()) for line in links + edges.splitlines() if line.strip()]
    for source, source_field, target, target_field in sorted(parsed, key=lambda edge: edge[3] != "item"):
        if source_field == "loop_linkage":
            graph.add_edge(create_loop_linkage(source, target))
        else:
            graph.add_edge(create_edge(source, source_field, target, target_field))
    graph.validate_self()
    return graph


def _sorted_items(items: list[AnyValue]) -> list[AnyValue]:
    return sorted(items, key=repr)


def _collect_runs(graph: Graph, compatibility: bool, *sinks: str) -> dict[str, Runs]:
    """Execute `graph` to completion and return each sink's executions, normalized."""
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = _ExecutionScheduler(state) if compatibility else _GenericGraphSchedulerAdapter(state)
    expected_scheduler = _ExecutionScheduler if compatibility else _GenericGraphSchedulerAdapter
    assert isinstance(state._execution_scheduler, expected_scheduler)
    for _ in range(5000):
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))
    assert state.is_complete() and not state.has_error()
    return {
        name: _sorted_runs(
            [
                (state._get_iteration_path(execution), state.results[execution].collection)
                for execution in state._prepared_registry().get_prepared_ids(name)
                if execution in state.results
            ]
        )
        for name in sinks or ("out",)
    }


def _sorted_runs(runs: Runs) -> Runs:
    return sorted(((path, _sorted_items(items)) for path, items in runs), key=repr)


def test_plain_iterate_collect_prepared_path_lookups_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph()
    graph.add_node(Concat(id="src", first=list(range(1000))))
    graph.add_node(Iterate(id="it0"))
    graph.add_node(Collect(id="out"))
    graph.add_edge(create_edge("src", "collection", "it0", "collection"))
    graph.add_edge(create_edge("it0", "item", "out", "item"))
    graph.validate_self()
    path_lookups = 0
    get_path = _ExecutionMaterializer._get_prepared_edge_iteration_path

    def counted_get_path(self: _ExecutionMaterializer, edge: AnyValue, prepared_id: str) -> tuple[int, ...]:
        nonlocal path_lookups
        path_lookups += 1
        return get_path(self, edge, prepared_id)

    monkeypatch.setattr(_ExecutionMaterializer, "_get_prepared_edge_iteration_path", counted_get_path)
    state = GraphExecutionState(graph=graph)
    for _ in range(5000):
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete() and not state.has_error()
    assert path_lookups <= 9_000


# Flaw: collapsing an all-empty frame must keep the parent frames, not index past the known iterator axes.
@pytest.mark.parametrize(
    "src, levels, expected",
    [
        pytest.param([[[], []]], 3, [((0, 0), []), ((0, 1), [])], id="one-outer-two-empty"),
        pytest.param([[[]], [[]]], 3, [((0, 0), []), ((1, 0), [])], id="two-outer-one-empty-each"),
        pytest.param([[], [[]]], 3, [((0,), []), ((1, 0), [])], id="empty-at-two-depths"),
        pytest.param([[[[], []]]], 4, [((0, 0, 0), []), ((0, 0, 1), [])], id="one-level-deeper"),
    ],
)
def test_all_empty_frames_produce_one_empty_run_per_frame(
    compatibility_scheduler: bool, src: list[AnyValue], levels: int, expected: Runs
) -> None:
    graph = _graph(
        src,
        levels,
        [
            Marked(id="m", marker="m"),
            Collect(id="col"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Collect(id="out"),
        ],
        f"""
        it{levels - 1}.item -> m.value
        m.value -> col.item
        col.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> out.item
        """,
    )

    assert _collect_runs(graph, compatibility_scheduler)["out"] == _sorted_runs(expected)


_MIXED_DEPTH_SRC = [[[["c", "a"]], [["c"]]], [[["b"], ["a"]]], []]
_SHALLOW_COLLECT_RUNS = [
    ((0,), ["T:[\"x[['c', 'a']]\", \"x[['c']]\"]", "S:[\"x['c', 'a']\"]", "S:[\"x['c']\"]"]),
    ((1,), ["T:[\"x[['b'], ['a']]\"]", "S:[\"x['b']\", \"x['a']\"]"]),
    ((2,), ["T:[]", "S:[]"]),
]


def _mixed_depth(shallow_nodes: list[BaseInvocation], shallow_edges: str) -> Graph:
    """A shallow T input and a Collect-per-it1-frame S input join `out`; `shallow_edges` must reach out.item."""
    return _graph(
        _MIXED_DEPTH_SRC,
        3,
        [
            *shallow_nodes,
            Marked(id="cm2", marker="x"),
            Collect(id="c2"),
            Marked(id="S", marker="S:"),
            Collect(id="out"),
        ],
        f"""
        {shallow_edges}
        it2.item -> cm2.value
        cm2.value -> c2.item
        c2.collection -> S.value
        S.value -> out.item
        """,
    )


def _collected_shallow_input(literal_ifs: int) -> Graph:
    nodes: list[BaseInvocation] = [Marked(id="cm1", marker="x"), Collect(id="c1"), Marked(id="T", marker="T:")]
    edges = ["it1.item -> cm1.value", "cm1.value -> c1.item", "c1.collection -> T.value"]
    previous = "T"
    for index in range(literal_ifs):
        nodes.append(If(id=f"if{index}", condition=True))
        edges.append(f"{previous}.value -> if{index}.true_input")
        previous = f"if{index}"
    edges.append(f"{previous}.value -> out.item")
    return _mixed_depth(nodes, "\n".join(edges))


def _iterated_shallow_input() -> Graph:
    return _mixed_depth([Marked(id="T", marker="T:")], "it0.item -> T.value\nT.value -> out.item")


def _collects_through_literal_if() -> Graph:  # fuzz-1259
    return _graph(
        [[[["c", "a", "c"]], [["c"]], [["c", ""], ["a", "b"]]], [[["", "b"], ["a", "a"]]], []],
        3,
        [
            Marked(id="cm1", marker="x"),
            Collect(id="col2"),
            Marked(id="p3", marker="T:"),
            Marked(id="cm4", marker="x"),
            Collect(id="col5"),
            Marked(id="p6", marker="F:"),
            If(id="if", condition=True),
            Collect(id="out"),
            Marked(id="cm7", marker="x"),
            Collect(id="col8"),
            Marked(id="p9", marker="S:"),
        ],
        """
        cm1.value -> col2.item
        cm4.value -> col5.item
        if.value -> out.item
        cm7.value -> col8.item
        p9.value -> out.item
        it1.item -> cm1.value
        col2.collection -> p3.value
        it1.item -> cm4.value
        col5.collection -> p6.value
        p3.value -> if.true_input
        p6.value -> if.false_input
        it2.item -> cm7.value
        col8.collection -> p9.value
        """,
    )


def _shallow_input_behind_two_literal_ifs_with_outer_collect() -> Graph:  # fuzz-2338
    return _graph(
        [[[["b"], ["a", "", "b"]], [["b", "b", "b"], []], []], [[["c"], []], [["c"], ["b"]]]],
        3,
        [
            Value(id="root2", value="aT:"),
            Marked(id="cm3", marker="x"),
            Collect(id="col4"),
            Marked(id="p5", marker="bT:"),
            If(id="iif1", condition=False),
            Marked(id="p6", marker="T:"),
            Marked(id="m7", marker="F:"),
            If(id="if", condition=False),
            Collect(id="out"),
            Marked(id="cm8", marker="x"),
            Collect(id="col9"),
            Marked(id="p10", marker="S:"),
            Collect(id="out2"),
        ],
        """
        cm3.value -> col4.item
        if.value -> out.item
        cm8.value -> col9.item
        p10.value -> out.item
        out.collection -> out2.item
        it2.item -> cm3.value
        col4.collection -> p5.value
        root2.value -> iif1.true_input
        p5.value -> iif1.false_input
        iif1.value -> p6.value
        it0.item -> m7.value
        p6.value -> if.true_input
        m7.value -> if.false_input
        it2.item -> cm8.value
        col9.collection -> p10.value
        """,
    )


# Flaw: inputs reaching one Collect from different depths must still land in one run per shared outer frame.
@pytest.mark.parametrize(
    "build, expected",
    [
        pytest.param(lambda: _collected_shallow_input(0), {"out": _SHALLOW_COLLECT_RUNS}, id="shallow-collect"),
        pytest.param(lambda: _collected_shallow_input(1), {"out": _SHALLOW_COLLECT_RUNS}, id="shallow-collect-one-if"),
        pytest.param(lambda: _collected_shallow_input(2), {"out": _SHALLOW_COLLECT_RUNS}, id="shallow-collect-two-ifs"),
        pytest.param(
            _iterated_shallow_input,
            {
                "out": [
                    ((0,), ["T:[[['c', 'a']], [['c']]]", "S:[\"x['c', 'a']\"]", "S:[\"x['c']\"]"]),
                    ((1,), ["T:[[['b'], ['a']]]", "S:[\"x['b']\", \"x['a']\"]"]),
                    ((2,), ["T:[]", "S:[]"]),
                ]
            },
            id="shallow-item",
        ),
        pytest.param(
            _collects_through_literal_if,
            {
                "out": [
                    (
                        (0,),
                        [
                            "T:[\"x[['c', 'a', 'c']]\", \"x[['c']]\", \"x[['c', ''], ['a', 'b']]\"]",
                            "S:[\"x['c', 'a', 'c']\"]",
                            "S:[\"x['c']\"]",
                            "S:[\"x['c', '']\", \"x['a', 'b']\"]",
                        ],
                    ),
                    ((1,), ["T:[\"x[['', 'b'], ['a', 'a']]\"]", "S:[\"x['', 'b']\", \"x['a', 'a']\"]"]),
                    ((2,), ["T:[]", "S:[]"]),
                ]
            },
            id="fuzz-1259-collects-through-if",
        ),
        pytest.param(
            _shallow_input_behind_two_literal_ifs_with_outer_collect,
            {
                "out": [
                    (
                        (0,),
                        [
                            "F:[[['b'], ['a', '', 'b']], [['b', 'b', 'b'], []], []]",
                            "S:[\"x['b']\", \"x['a', '', 'b']\"]",
                            "S:[\"x['b', 'b', 'b']\", 'x[]']",
                            "S:[]",
                        ],
                    ),
                    ((1,), ["F:[[['c'], []], [['c'], ['b']]]", "S:[\"x['c']\", 'x[]']", "S:[\"x['c']\", \"x['b']\"]"]),
                ],
                "out2": [
                    (
                        (),
                        [
                            [
                                "F:[[['b'], ['a', '', 'b']], [['b', 'b', 'b'], []], []]",
                                "S:[\"x['b']\", \"x['a', '', 'b']\"]",
                                "S:[\"x['b', 'b', 'b']\", 'x[]']",
                                "S:[]",
                            ],
                            ["F:[[['c'], []], [['c'], ['b']]]", "S:[\"x['c']\", 'x[]']", "S:[\"x['c']\", \"x['b']\"]"],
                        ],
                    )
                ],
            },
            id="fuzz-2338-nested-ifs-outer-collect",
        ),
    ],
)
def test_inputs_at_different_depths_share_one_run_per_outer_frame(
    compatibility_scheduler: bool, build, expected: dict[str, Runs]
) -> None:
    runs = _collect_runs(build(), compatibility_scheduler, *expected)

    assert runs == {sink: _sorted_runs(sink_runs) for sink, sink_runs in expected.items()}


_FOR_SRC = [[], [["a", "b"]]]
_FOR_RUNS = {
    "out": [((0,), []), ((1,), ["S:['Fa', 'Fb']", "T:['a', 'b']"])],
    "out2": [((0,), []), ((1,), ["rS:['Fa', 'Fb']", "rT:['a', 'b']"])],
}


def _for_final_with_sibling(sibling_nodes: list[BaseInvocation], sibling_edges: str) -> Graph:
    """`sibling_edges` carry `it1.item` to out.item beside a For final; out.collection is re-iterated into out2."""
    return _graph(
        _FOR_SRC,
        2,
        [
            Marked(id="mT", marker="T:"),
            *sibling_nodes,
            For(id="for"),
            Marked(id="fb", marker="F"),
            ForReturn(id="ret"),
            Marked(id="Sx", marker="S:"),
            Collect(id="out"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Collect(id="out2"),
        ],
        f"""
        it1.item -> mT.value
        {sibling_edges}
        it1.item -> for.collection
        for.item -> fb.value
        fb.value -> ret.output
        for.loop_linkage -> ret.loop_linkage
        for.output_collection -> Sx.value
        Sx.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> out2.item
        """,
    )


def _for_sibling_direct() -> Graph:
    return _for_final_with_sibling([], "mT.value -> out.item")


def _for_sibling_through_item_if() -> Graph:
    return _for_final_with_sibling(
        [If(id="if"), Truthy(id="fc"), Value(id="R", value="R")],
        """
        it1.item -> fc.value
        fc.condition -> if.condition
        mT.value -> if.true_input
        R.value -> if.false_input
        if.value -> out.item
        """,
    )


def _for_sibling_through_literal_if() -> Graph:
    return _for_final_with_sibling(
        [If(id="if", condition=True)],
        """
        mT.value -> if.true_input
        if.value -> out.item
        """,
    )


def _unselected_for_behind_item_if_with_reiterate() -> Graph:  # fuzz-1681
    return _graph(
        [[], [["a", "b"]]],
        2,
        [
            Marked(id="m1", marker="T:"),
            For(id="for2", collection=["u"]),
            Marked(id="fb3", marker="F"),
            ForReturn(id="ret4"),
            If(id="if"),
            Truthy(id="cond"),
            Collect(id="out"),
            For(id="for5"),
            Marked(id="fb6", marker="F"),
            ForReturn(id="ret7"),
            Marked(id="p8", marker="S:"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Value(id="rr", value="RR"),
            If(id="rif"),
            Collect(id="out2"),
            Truthy(id="rc"),
        ],
        """
        if.value -> out.item
        p8.value -> out.item
        rif.value -> out2.item
        it1.item -> m1.value
        for2.item -> fb3.value
        fb3.value -> ret4.output
        for2.loop_linkage -> ret4.loop_linkage
        m1.value -> if.true_input
        for2.output_collection -> if.false_input
        it1.item -> cond.value
        cond.condition -> if.condition
        it1.item -> for5.collection
        for5.item -> fb6.value
        fb6.value -> ret7.output
        for5.loop_linkage -> ret7.loop_linkage
        for5.output_collection -> p8.value
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        rit.item -> rc.value
        rc.condition -> rif.condition
        """,
    )


def _root_for_through_literal_if_beside_empty_iterator() -> Graph:  # fuzz-2236
    return _graph(
        [[["", "b", "a"]], [[], ["b", "b"]]],
        2,
        [
            For(id="for1", collection=["u", "v"]),
            Marked(id="fb2", marker="F"),
            ForReturn(id="ret3"),
            Marked(id="m4", marker="F:"),
            If(id="if", condition=True),
            Collect(id="out"),
            Concat(id="sbs"),
            Iterate(id="itb"),
            Marked(id="sx", marker="n"),
            Collect(id="out2"),
        ],
        """
        if.value -> out.item
        sx.value -> out.item
        out.collection -> out2.item
        for1.item -> fb2.value
        fb2.value -> ret3.output
        for1.loop_linkage -> ret3.loop_linkage
        it1.item -> m4.value
        for1.output_collection -> if.true_input
        m4.value -> if.false_input
        sbs.collection -> itb.collection
        itb.item -> sx.value
        """,
    )


def _nested_for_ifs_beside_per_item_for_with_reiterate() -> Graph:  # fuzz-2600
    return _graph(
        [[[]], [[[], ["b"], [""]], [["a", ""]], [["a"], ["", "a"]]]],
        3,
        [
            Value(id="root2", value="aT:"),
            Value(id="root3", value="bT:"),
            If(id="iif1", condition=False),
            Marked(id="p4", marker="T:"),
            For(id="for6", collection=["u", "v"]),
            Marked(id="fb7", marker="F"),
            ForReturn(id="ret8"),
            For(id="for9", collection=["u"]),
            Marked(id="fb10", marker="F"),
            ForReturn(id="ret11"),
            If(id="iif5"),
            BooleanInvocation(id="ic12", value=False),
            Marked(id="p13", marker="F:"),
            If(id="if", condition=False),
            Collect(id="out"),
            For(id="for14"),
            Marked(id="fb15", marker="F"),
            ForReturn(id="ret16"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Value(id="rr", value="RR"),
            If(id="rif", condition=True),
            Collect(id="out2"),
        ],
        """
        if.value -> out.item
        for14.output_collection -> out.item
        rif.value -> out2.item
        root2.value -> iif1.true_input
        root3.value -> iif1.false_input
        iif1.value -> p4.value
        for6.item -> fb7.value
        fb7.value -> ret8.output
        for6.loop_linkage -> ret8.loop_linkage
        for9.item -> fb10.value
        fb10.value -> ret11.output
        for9.loop_linkage -> ret11.loop_linkage
        for6.output_collection -> iif5.true_input
        for9.output_collection -> iif5.false_input
        ic12.value -> iif5.condition
        iif5.value -> p13.value
        p4.value -> if.true_input
        p13.value -> if.false_input
        it2.item -> for14.collection
        for14.item -> fb15.value
        fb15.value -> ret16.output
        for14.loop_linkage -> ret16.loop_linkage
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        """,
    )


# Flaw: a For final beside a passthrough must join the passthrough's run, keeping both values in one frame.
@pytest.mark.parametrize(
    "build, expected",
    [
        pytest.param(_for_sibling_direct, _FOR_RUNS, id="direct"),
        pytest.param(_for_sibling_through_item_if, _FOR_RUNS, id="through-per-item-if"),
        pytest.param(_for_sibling_through_literal_if, _FOR_RUNS, id="through-literal-if"),
        pytest.param(
            _unselected_for_behind_item_if_with_reiterate,
            {
                "out": [((0,), []), ((1,), ["T:['a', 'b']", "S:['Fa', 'Fb']"])],
                "out2": [((0,), []), ((1,), ["rT:['a', 'b']", "rS:['Fa', 'Fb']"])],
            },
            id="fuzz-1681-for-behind-per-item-if",
        ),
        pytest.param(
            _root_for_through_literal_if_beside_empty_iterator,
            {
                "out": [((0,), [["Fu", "Fv"]]), ((1,), [["Fu", "Fv"], ["Fu", "Fv"]])],
                "out2": [((), [[["Fu", "Fv"]], [["Fu", "Fv"], ["Fu", "Fv"]]])],
            },
            id="fuzz-2236-root-for-beside-empty-iterator",
        ),
        pytest.param(
            _nested_for_ifs_beside_per_item_for_with_reiterate,
            {
                "out": [
                    ((0, 0), ["F:['Fu']"]),
                    ((1, 0, 0), ["F:['Fu']", []]),
                    ((1, 0, 1), ["F:['Fu']", ["Fb"]]),
                    ((1, 0, 2), ["F:['Fu']", ["F"]]),
                    ((1, 1, 0), ["F:['Fu']", ["Fa", "F"]]),
                    ((1, 2, 0), ["F:['Fu']", ["Fa"]]),
                    ((1, 2, 1), ["F:['Fu']", ["F", "Fa"]]),
                ],
                "out2": [
                    ((0, 0, 0), ["rF:['Fu']"]),
                    ((1, 0, 0), ["rF:['Fu']", "r[]"]),
                    ((1, 0, 1), ["rF:['Fu']", "r['Fb']"]),
                    ((1, 0, 2), ["rF:['Fu']", "r['F']"]),
                    ((1, 1, 0), ["rF:['Fu']", "r['Fa', 'F']"]),
                    ((1, 2, 0), ["rF:['Fu']", "r['Fa']"]),
                    ((1, 2, 1), ["rF:['Fu']", "r['F', 'Fa']"]),
                ],
            },
            id="fuzz-2600-nested-for-ifs",
        ),
    ],
)
def test_for_final_beside_passthrough_stays_in_one_run_per_frame(
    compatibility_scheduler: bool, build, expected: dict[str, Runs]
) -> None:
    runs = _collect_runs(build(), compatibility_scheduler, *expected)

    assert runs == {sink: _sorted_runs(sink_runs) for sink, sink_runs in expected.items()}


# Flaw: a per-item If that is never selected contributes no items, so it must not add runs of its own.
# Which runs the O: values group into is undecided, so only the absence of empty runs and of lost values is pinned.
@pytest.mark.parametrize(
    "src, expected_values",
    [
        pytest.param(
            [[[["c"]], [[""]]], [[["b"], ["d"]]]],
            {"O:[[['c']], [['']]]", "O:[[['b'], ['d']]]"},
            id="reduced",
        ),
        pytest.param(
            [[[["c", "a", "a"]], [[""]]], [[["b"], ["c", "c"], ["c"]]], [[["a", "c", "b"], ["a", "a"]]]],
            {
                "O:[[['c', 'a', 'a']], [['']]]",
                "O:[[['b'], ['c', 'c'], ['c']]]",
                "O:[[['a', 'c', 'b'], ['a', 'a']]]",
            },
            id="fuzz-1292",
        ),
    ],
)
def test_unselected_per_item_if_adds_no_empty_runs(
    compatibility_scheduler: bool, src: list[AnyValue], expected_values: set[str]
) -> None:
    graph = _graph(
        src,
        3,
        [
            Marked(id="cm1", marker="x"),
            Collect(id="col"),
            Marked(id="p3", marker="T:"),
            Marked(id="p4", marker="F:"),
            If(id="if"),
            Truthy(id="fc"),
            Marked(id="m5", marker="O:"),
            If(id="if2", condition=False),
            Collect(id="out"),
        ],
        """
        it0.item -> cm1.value
        cm1.value -> col.item
        col.collection -> p3.value
        col.collection -> p4.value
        p3.value -> if.true_input
        p4.value -> if.false_input
        it2.item -> fc.value
        fc.condition -> if.condition
        it0.item -> m5.value
        m5.value -> if2.false_input
        if.value -> if2.true_input
        if2.value -> out.item
        """,
    )

    runs = _collect_runs(graph, compatibility_scheduler)["out"]

    assert all(items for _, items in runs), runs
    assert expected_values <= {item for _, items in runs for item in items}, runs


def test_root_conditioned_if_waits_for_a_nested_for_final(compatibility_scheduler: bool) -> None:
    graph = _graph(
        [[[["a"]]]],
        1,
        [
            Marked(id="m2", marker="aT:"),
            For(id="for3", collection=["u", "v"]),
            Marked(id="fb4", marker="F"),
            ForReturn(id="ret5"),
            If(id="iif1", condition=False),
            BooleanInvocation(id="ic6", value=True),
            For(id="for8"),
            Marked(id="fb9", marker="F"),
            ForReturn(id="ret10"),
            Marked(id="p11", marker="aF:"),
            If(id="iif7", condition=True),
            Marked(id="p13", marker="F:"),
            If(id="if", condition=False),
            BooleanInvocation(id="cond", value=True),
        ],
        """
        it0.item -> m2.value
        for3.item -> fb4.value
        fb4.value -> ret5.output
        for3.loop_linkage -> ret5.loop_linkage
        m2.value -> iif1.true_input
        ic6.value -> iif1.condition
        it0.item -> for8.collection
        for8.item -> fb9.value
        fb9.value -> ret10.output
        for8.loop_linkage -> ret10.loop_linkage
        for8.output_collection -> p11.value
        p11.value -> iif7.true_input
        iif7.value -> p13.value
        iif1.value -> if.true_input
        p13.value -> if.false_input
        cond.value -> if.condition
        """,
    )
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    trace = []
    for _ in range(200):
        node = state.next()
        if node is None:
            break
        source_id = state.prepared_source_mapping[node.id]
        trace.append((source_id, state._get_iteration_path(node.id)))
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete(), f"execution stalled; trace={trace}"
    assert not state.has_error(), trace
    observed = {
        node_id: [
            (state._get_iteration_path(exec_id), getattr(state.results[exec_id], "value", None))
            for exec_id in state._prepared_registry().get_prepared_ids(node_id)
            if exec_id in state.results
        ]
        for node_id in ("m2", "iif1", "if")
    }
    assert observed == {
        "m2": [((0,), "aT:[[['a']]]")],
        "iif1": [((0,), "aT:[[['a']]]")],
        "if": [((0,), "aT:[[['a']]]")],
    }


@pytest.mark.parametrize(
    "true_branch_edges",
    [
        pytest.param(
            """
            it1.item -> t.value
            t.value -> c1.item
            c1.collection -> c2.item
            c2.collection -> if.true_input
            """,
            id="collect-of-collect-fuzz-2037",
        ),
        pytest.param(
            """
            it0.item -> t.value
            t.value -> c2.item
            c2.collection -> if.true_input
            """,
            id="collect-closing-outer-axis-fuzz-460",
        ),
    ],
)
@pytest.mark.parametrize("computed_condition", [False, True], ids=["literal-condition", "computed-condition"])
def test_root_condition_selecting_per_frame_collect_closes_downstream_collect(
    true_branch_edges: str, computed_condition: bool, compatibility_scheduler: bool
) -> None:
    nodes: list[BaseInvocation] = [
        Marked(id="x", marker="x"),
        Collect(id="col"),
        Marked(id="f", marker="F:"),
        Marked(id="t", marker="x"),
        Collect(id="c2"),
        If(id="if", condition=False),
        Collect(id="out"),
    ]
    if "c1" in true_branch_edges:
        nodes.append(Collect(id="c1"))
    edges = f"""
        it1.item -> x.value
        x.value -> col.item
        col.collection -> f.value
        f.value -> if.false_input
        {true_branch_edges}
        if.value -> out.item
    """
    if computed_condition:
        nodes.append(BooleanInvocation(id="cond", value=False))
        edges += "cond.value -> if.condition\n"
    graph = _graph([["a", "b"], ["c"]], 2, nodes, edges)

    assert _collect_runs(graph, compatibility_scheduler)["out"] == [((), ["F:['xa', 'xb']", "F:['xc']"])]


@pytest.mark.parametrize("computed_condition", [False, True], ids=["literal-condition", "computed-condition"])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume-after-parent-collect"])
def test_reiterating_collect_keeps_root_if_value_in_parent_frames(
    compatibility_scheduler: bool, computed_condition: bool, resume: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes: list[BaseInvocation] = [
        Marked(id="m", marker="T:"),
        Collect(id="out"),
        Iterate(id="rit"),
        Marked(id="rm", marker="r"),
        If(id="rif") if computed_condition else If(id="rif", condition=False),
        Value(id="rr", value="RR"),
        Collect(id="out2"),
    ]
    edges = """
        it1.item -> m.value
        m.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        rif.value -> out2.item
    """
    if computed_condition:
        nodes.append(BooleanInvocation(id="cond", value=False))
        edges += "cond.value -> rif.condition\n"
    graph = _graph([[], [["a"]], [["b"], ["c"]]], 2, nodes, edges)

    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    checkpointed = not resume
    for _ in range(5000):
        out_exec_ids = state.source_prepared_mapping.get("out", set())
        rit_exec_ids = state.source_prepared_mapping.get("rit", set())
        if (
            resume
            and not checkpointed
            and out_exec_ids
            and all(exec_id in state.results for exec_id in out_exec_ids)
            and not any(exec_id in state.results for exec_id in rit_exec_ids)
        ):
            state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
            state._execution_scheduler = (
                _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
            )
            checkpointed = True
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert checkpointed
    assert state.is_complete() and not state.has_error()
    materializer = _ExecutionMaterializer(state)
    registry = state._prepared_registry()
    root_value_edge = graph._get_input_edges("out2", "item")[0]
    from invokeai.app.services.shared.graph_if_runtime import _get_if_iteration_paths_with_ancestry

    iterator_ancestry = materializer._get_iterator_ancestry(materializer.iterator_graph())
    assert _get_if_iteration_paths_with_ancestry(
        state,
        "rif",
        set(),
        materializer._get_prepared_nodes_for_source,
        materializer._get_prepared_edge_iteration_path,
        iterator_ancestry,
    ) == [(1, 0), (2, 0), (2, 1)]
    static_axis_derivations = 0
    get_static_collect_group_axes = materializer._get_static_collect_group_axes

    def count_static_axis_derivations(edge: AnyValue) -> tuple[str, ...]:
        nonlocal static_axis_derivations
        static_axis_derivations += 1
        return get_static_collect_group_axes(edge)

    monkeypatch.setattr(materializer, "_get_static_collect_group_axes", count_static_axis_derivations)
    rif_exec_ids = registry.get_prepared_ids("rif")
    for exec_id in rif_exec_ids:
        assert materializer._get_prepared_edge_iteration_axes(root_value_edge, exec_id) == ("it0", "rit")
    parent_item_edges = graph._get_input_edges("out", "item")
    assert static_axis_derivations == len(parent_item_edges)

    empty_out_exec_id = registry.get_prepared_id_at_path("out", (0,))
    assert empty_out_exec_id is not None
    collection_edge = graph._get_input_edges("rit", "collection")[0]
    assert materializer._get_prepared_edge_iteration_axes(collection_edge, empty_out_exec_id) == ("it0",)

    execution_graph_reads = 0
    get_execution_graph_flat = state._get_execution_graph_flat

    def count_execution_graph_reads() -> AnyValue:
        nonlocal execution_graph_reads
        execution_graph_reads += 1
        return get_execution_graph_flat()

    monkeypatch.setattr(state, "_get_execution_graph_flat", count_execution_graph_reads)
    for exec_id in rif_exec_ids:
        assert materializer._get_prepared_edge_iteration_axes(root_value_edge, exec_id) == ("it0", "rit")
    assert execution_graph_reads == 0

    observed = _sorted_runs(
        [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in state.source_prepared_mapping.get("out2", ())
            if exec_id in state.results
        ]
    )
    assert observed == [((0,), []), ((1,), ["RR"]), ((2,), ["RR", "RR"])]

    state.add_node(Marked(id="unrelated", marker="x"))
    assert materializer._get_prepared_edge_iteration_axes(root_value_edge, next(iter(rif_exec_ids))) == (
        "it0",
        "rit",
    )
    assert static_axis_derivations == len(parent_item_edges) * 2


def test_reiterated_collect_parent_path_index_is_stable_while_children_expand(
    compatibility_scheduler: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_count = 24
    nodes: list[BaseInvocation] = [
        Marked(id="m", marker="T:"),
        Collect(id="out"),
        Iterate(id="rit"),
        Marked(id="rm", marker="r"),
        If(id="rif", condition=False),
        Value(id="rr", value="RR"),
        Collect(id="out2"),
    ]
    graph = _graph(
        [[[f"v{i}"]] for i in range(child_count)],
        2,
        nodes,
        """
        it1.item -> m.value
        m.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        rif.value -> out2.item
        """,
    )
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    registry = state._prepared_registry()
    original_lookup = registry.get_prepared_id_at_path
    original_get_iteration_path = registry.get_iteration_path
    original_register = registry.register
    out_index_builds = 0
    out_index_source_visits = 0
    out_lookups = 0
    late_out_registrations = 0
    building_out_index = False

    def track_iteration_path(exec_id: str) -> tuple[int, ...] | None:
        nonlocal out_index_source_visits
        if building_out_index:
            out_index_source_visits += 1
        return original_get_iteration_path(exec_id)

    def track_lookup(source_node_id: str, iteration_path: tuple[int, ...]) -> str | None:
        nonlocal building_out_index, out_index_builds, out_lookups
        if source_node_id != "out":
            return original_lookup(source_node_id, iteration_path)
        out_lookups += 1
        building_out_index = source_node_id not in registry._prepared_ids_by_source_and_path
        if building_out_index:
            out_index_builds += 1
        try:
            return original_lookup(source_node_id, iteration_path)
        finally:
            building_out_index = False

    def track_register(exec_id: str, source_node_id: str) -> None:
        nonlocal late_out_registrations
        if source_node_id == "out" and out_index_builds:
            late_out_registrations += 1
        original_register(exec_id, source_node_id)

    monkeypatch.setattr(registry, "get_prepared_id_at_path", track_lookup)
    monkeypatch.setattr(registry, "get_iteration_path", track_iteration_path)
    monkeypatch.setattr(registry, "register", track_register)

    for _ in range(10000):
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete() and not state.has_error()
    assert len(registry.get_prepared_ids("out")) == child_count
    assert len(registry.get_prepared_ids("rif")) == child_count
    assert _sorted_runs(
        [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in registry.get_prepared_ids("out2")
        ]
    ) == _sorted_runs([((index,), ["RR"]) for index in range(child_count)])
    assert out_index_builds == 1
    assert out_index_source_visits == child_count
    assert late_out_registrations == 0
    assert out_lookups >= child_count


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "checkpoint-after-parent-collect"])
@pytest.mark.parametrize("empty_parent_run", [False, True], ids=["non-empty-parent-runs", "empty-parent-run"])
def test_reiterating_collect_consumes_each_sibling_group_independently(
    compatibility_scheduler: bool, resume: bool, empty_parent_run: bool
) -> None:
    graph = _graph(
        ([[], ["a"], ["b", "c"]] if empty_parent_run else [["a"], ["b", "c"]]),
        2,
        [
            Marked(id="m", marker="T:"),
            Concat(id="sib_src", first=["q", "p", "r"]),
            Iterate(id="itb"),
            Marked(id="n", marker="n"),
            Collect(id="out"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Collect(id="out2"),
        ],
        """
        it1.item -> m.value
        m.value -> out.item
        sib_src.collection -> itb.collection
        itb.item -> n.value
        n.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> out2.item
        """,
    )
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    checkpointed = not resume
    for _ in range(5000):
        out_exec_ids = state.source_prepared_mapping.get("out", set())
        rit_exec_ids = state.source_prepared_mapping.get("rit", set())
        if (
            resume
            and not checkpointed
            and out_exec_ids
            and all(exec_id in state.results for exec_id in out_exec_ids)
            and not any(exec_id in state.results for exec_id in rit_exec_ids)
        ):
            state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
            state._execution_scheduler = (
                _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
            )
            checkpointed = True
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert checkpointed
    assert state.is_complete() and not state.has_error()
    root_sibling_runs = [
        state.results[exec_id].collection
        for exec_id in state.source_prepared_mapping.get("out2", ())
        if exec_id in state.results and state._get_iteration_path(exec_id) == ()
    ]
    assert root_sibling_runs == [["rnq", "rnp", "rnr"]]
    expected = (
        [
            ((), ["rnp", "rnq", "rnr"]),
            ((0,), []),
            ((1,), ["rT:a"]),
            ((2,), ["rT:b", "rT:c"]),
        ]
        if empty_parent_run
        else [
            ((), ["rnp", "rnq", "rnr"]),
            ((0,), ["rT:a"]),
            ((1,), ["rT:b", "rT:c"]),
        ]
    )
    assert (
        _sorted_runs(
            [
                (state._get_iteration_path(exec_id), state.results[exec_id].collection)
                for exec_id in state.source_prepared_mapping.get("out2", ())
                if exec_id in state.results
            ]
        )
        == expected
    )


@pytest.mark.parametrize("checkpoint_after_out", [False, True], ids=["fresh", "checkpoint-after-out"])
def test_downstream_if_keeps_sibling_collect_parent_frames(
    compatibility_scheduler: bool, checkpoint_after_out: bool
) -> None:
    graph = _graph(
        [["a"], ["b", "c"]],
        2,
        [
            Marked(id="m", marker="T:"),
            Truthy(id="cond"),
            If(id="if"),
            Concat(id="sib_src", first=["p", "q", "r"]),
            Iterate(id="itb"),
            Marked(id="n", marker="n"),
            Collect(id="out"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Truthy(id="rc"),
            Value(id="rr", value="RR"),
            If(id="rif"),
            Collect(id="out2"),
        ],
        """
        it1.item -> m.value
        m.value -> if.true_input
        it1.item -> cond.value
        cond.condition -> if.condition
        if.value -> out.item
        sib_src.collection -> itb.collection
        itb.item -> n.value
        n.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        rit.item -> rc.value
        rc.condition -> rif.condition
        rif.value -> out2.item
        """,
    )
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    checkpointed = not checkpoint_after_out
    for _ in range(5000):
        out_ids = state._prepared_registry().get_prepared_ids("out")
        rit_ids = state._prepared_registry().get_prepared_ids("rit")
        if (
            checkpoint_after_out
            and not checkpointed
            and len(out_ids) == 3
            and all(exec_id in state.results for exec_id in out_ids)
            and not any(exec_id in state.results for exec_id in rit_ids)
        ):
            state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
            state._execution_scheduler = (
                _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
            )
            checkpointed = True
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert checkpointed
    assert state.is_complete() and not state.has_error()
    observed_out = _sorted_runs(
        [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in state._prepared_registry().get_prepared_ids("out")
            if exec_id in state.results
        ]
    )
    observed_out2 = _sorted_runs(
        [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in state._prepared_registry().get_prepared_ids("out2")
            if exec_id in state.results
        ]
    )
    assert observed_out == [((), ["np", "nq", "nr"]), ((0,), ["T:a"]), ((1,), ["T:b", "T:c"])]
    assert observed_out2 == [
        ((), ["rnp", "rnq", "rnr"]),
        ((0,), ["rT:a"]),
        ((1,), ["rT:b", "rT:c"]),
    ]


def test_sibling_collect_classification_is_cached_per_source_graph(
    compatibility_scheduler: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream_count = 12
    parent_sources = {"m", "n"}
    graph = _graph(
        ["a", "b"],
        1,
        [
            Marked(id="m", marker="T:"),
            Concat(id="sib_src", first=["p", "q"]),
            Iterate(id="itb"),
            Marked(id="n", marker="n"),
            Collect(id="out"),
            Iterate(id="rit"),
            *[Marked(id=f"rm{index}", marker="r") for index in range(downstream_count)],
            *[Collect(id=f"sink{index}") for index in range(downstream_count)],
        ],
        "\n".join(
            [
                "it0.item -> m.value",
                "m.value -> out.item",
                "sib_src.collection -> itb.collection",
                "itb.item -> n.value",
                "n.value -> out.item",
                "out.collection -> rit.collection",
                *[
                    f"rit.item -> rm{index}.value\nrm{index}.value -> sink{index}.item"
                    for index in range(downstream_count)
                ],
            ]
        ),
    )
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    materializer = _ExecutionMaterializer(state)
    original_get_source_axes = materializer._get_collect_source_iterator_ids
    classifications = dict.fromkeys(parent_sources, 0)

    def count_parent_item_classification(source_node_id: str) -> tuple[str, ...]:
        if source_node_id in parent_sources:
            classifications[source_node_id] += 1
        return original_get_source_axes(source_node_id)

    monkeypatch.setattr(materializer, "_get_collect_source_iterator_ids", count_parent_item_classification)
    for index in range(downstream_count):
        assert materializer._is_sibling_collect_reiteration(f"rm{index}")

    assert classifications == dict.fromkeys(parent_sources, 1)

    state.add_node(Marked(id="unrelated", marker="x"))
    assert materializer._is_sibling_collect_reiteration("rm0")
    assert classifications == dict.fromkeys(parent_sources, 2)


def _run_reiteration_graph(
    graph: Graph, compatibility_scheduler: bool, checkpoint_after_out: bool
) -> GraphExecutionState:
    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = (
        _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
    )
    if checkpoint_after_out:
        for _ in range(5000):
            out_ids = state._prepared_registry().get_prepared_ids("out")
            if len(out_ids) == 2 and all(exec_id in state.results for exec_id in out_ids):
                break
            node = state.next()
            assert node is not None, "graph stalled before upstream Collect completed"
            state.complete(node.id, node.invoke(Mock()))
        else:
            pytest.fail("upstream Collect did not produce both executions")

        assert not any(exec_id in state.results for exec_id in state._prepared_registry().get_prepared_ids("rit"))
        state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
        state._execution_scheduler = (
            _ExecutionScheduler(state) if compatibility_scheduler else _GenericGraphSchedulerAdapter(state)
        )

    for _ in range(5000):
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))
    assert state.is_complete() and not state.has_error()
    return state


def _lstein_reiterate_graph(with_for: bool) -> Graph:
    nodes: list[BaseInvocation] = [
        Marked(id="cm", marker="C:"),
        Truthy(id="condition"),
        If(id="if"),
        Collect(id="out"),
        Iterate(id="rit"),
        Marked(id="rm", marker="r"),
        Value(id="RR", value="RR"),
        Truthy(id="rcondition"),
        If(id="rif"),
        Collect(id="out2"),
    ]
    if with_for:
        nodes.extend([For(id="for", collection=["u"]), Marked(id="fb", marker="F"), ForReturn(id="ret")])
        true_branch = """
        for.item -> fb.value
        fb.value -> ret.output
        for.loop_linkage -> ret.loop_linkage
        for.output_collection -> if.true_input
        """
    else:
        nodes.append(Value(id="for_value", value=["Fu"]))
        true_branch = "for_value.value -> if.true_input"

    return _graph(
        [[["c"], ["c", ""]], []],
        2,
        nodes,
        f"""
        it1.item -> cm.value
        cm.value -> if.false_input
        it1.item -> condition.value
        condition.condition -> if.condition
        {true_branch}
        if.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        RR.value -> rif.false_input
        rit.item -> rcondition.value
        rcondition.condition -> rif.condition
        rif.value -> out2.item
        """,
    )


@pytest.mark.parametrize("with_for", [False, True], ids=["root-value", "for-final"])
@pytest.mark.parametrize("checkpoint_after_out", [False, True], ids=["fresh", "checkpoint-after-out"])
def test_reiterated_root_condition_keeps_lstein_empty_frames(
    compatibility_scheduler: bool, with_for: bool, checkpoint_after_out: bool
) -> None:
    state = _run_reiteration_graph(_lstein_reiterate_graph(with_for), compatibility_scheduler, checkpoint_after_out)
    out_runs = [
        (state._get_iteration_path(exec_id), state.results[exec_id].collection)
        for exec_id in state._prepared_registry().get_prepared_ids("out")
        if exec_id in state.results
    ]
    out2_runs = [
        (state._get_iteration_path(exec_id), state.results[exec_id].collection)
        for exec_id in state._prepared_registry().get_prepared_ids("out2")
        if exec_id in state.results
    ]
    assert _sorted_runs(out_runs) == [((0,), [["Fu"], ["Fu"]]), ((1,), [])]
    assert _sorted_runs(out2_runs) == [((0,), ["r['Fu']", "r['Fu']"]), ((1,), [])]


def _seed_1234_reiteration_graph() -> Graph:
    return _graph(
        [[], [[[], [], ["b"]], [], []]],
        3,
        [
            Marked(id="m1", marker="T:"),
            Marked(id="m2", marker="F:"),
            If(id="if", condition=False),
            Truthy(id="cond"),
            Marked(id="cm3", marker="x"),
            Collect(id="col4"),
            Marked(id="p5", marker="O:"),
            If(id="if2", condition=False),
            Collect(id="out"),
            Iterate(id="rit"),
            Marked(id="rm", marker="r"),
            Value(id="rr", value="RR"),
            If(id="rif"),
            Collect(id="out2"),
            BooleanInvocation(id="rc", value=True),
        ],
        """
        it0.item -> m1.value
        it0.item -> m2.value
        m1.value -> if.true_input
        m2.value -> if.false_input
        it1.item -> cond.value
        cond.condition -> if.condition
        it0.item -> cm3.value
        cm3.value -> col4.item
        col4.collection -> p5.value
        if.value -> if2.true_input
        p5.value -> if2.false_input
        if2.value -> out.item
        out.collection -> rit.collection
        rit.item -> rm.value
        rm.value -> rif.true_input
        rr.value -> rif.false_input
        rc.value -> rif.condition
        rif.value -> out2.item
        """,
    )


@pytest.mark.parametrize("checkpoint_after_out", [False, True], ids=["fresh", "checkpoint-after-out"])
def test_seed_1234_reiteration_keeps_empty_outer_frame(
    compatibility_scheduler: bool, checkpoint_after_out: bool
) -> None:
    state = _run_reiteration_graph(_seed_1234_reiteration_graph(), compatibility_scheduler, checkpoint_after_out)
    collected_value = "O:['x[]', \"x[[[], [], ['b']], [], []]\"]"
    reiterated_value = f"r{collected_value}"
    out_runs = [
        (state._get_iteration_path(exec_id), state.results[exec_id].collection)
        for exec_id in state._prepared_registry().get_prepared_ids("out")
        if exec_id in state.results
    ]
    out2_runs = [
        (state._get_iteration_path(exec_id), state.results[exec_id].collection)
        for exec_id in state._prepared_registry().get_prepared_ids("out2")
        if exec_id in state.results
    ]
    assert _sorted_runs(out_runs) == [((0,), []), ((1,), [collected_value] * 3)]
    assert _sorted_runs(out2_runs) == [((0,), []), ((1,), [reiterated_value] * 3)]
