"""Execution-engine fuzzer.

Run with ``uv run --no-sync pytest tests/app/services/shared/test_execution_engine_fuzz.py -m slow --durations=10``.
"""

from __future__ import annotations

import json
import random
from typing import Any
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from invokeai.app.services.shared.graph_if_runtime import UnsupportedIfIterationAxesError
from tests.test_nodes import (
    AnyTypeTestInvocation,
    FixedCollectionTestInvocation,
    MarkedAnyTypeTestInvocation,
    create_edge,
    create_loop_linkage,
)


def G(nodes, edges):
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in sorted(
        edges, key=lambda edge: 0 if (edge[3] if isinstance(edge, tuple) else edge.destination.field) == "item" else 1
    ):
        graph.add_edge(create_edge(*edge) if isinstance(edge, tuple) else edge)
    graph.validate_self()
    return graph


LEAVES = ["a", "b", "", "c"]


def data(r, depth):
    if depth == 0:
        return r.choice(LEAVES)
    n = r.choice([0, 1, 1, 2, 2, 3])
    return [data(r, depth - 1) for _ in range(n)]


def build(seed):
    r = random.Random(seed)
    L = r.choice([1, 2, 2, 3])
    depth = L + 1  # leaves under deepest iterate are lists of strings
    src = data(r, depth)
    if not src and r.random() < 0.7:

        def nonempty(level):
            return "a" if level == 0 else [nonempty(level - 1)]

        src = data(r, depth) or [nonempty(depth - 1)]
    nodes = [CollectionConcatInvocation(id="src", first=src)]
    edges = []
    prev = ("src", "collection")
    for k in range(L):
        nodes.append(IterateInvocation(id=f"it{k}"))
        edges.append((*prev, f"it{k}", "collection"))
        prev = (f"it{k}", "item")
        if k < L - 1:
            nodes.append(CollectionConcatInvocation(id=f"cc{k}"))
            edges.append((f"it{k}", "item", f"cc{k}", "first"))
            prev = (f"cc{k}", "collection")
    ctr = [0]

    def new(prefix):
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    shared_cache = {}
    used = []

    def branch(kind, marker):
        t = kind[0]
        used.append(kind[0])
        if t == "root":
            n = new("root")
            nodes.append(AnyTypeTestInvocation(id=n, value=marker))
            return (n, "value")
        if t == "iter":
            n = new("m")
            nodes.append(MarkedAnyTypeTestInvocation(id=n, marker=marker))
            edges.append((f"it{kind[1]}", "item", n, "value"))
            return (n, "value")
        if t in ("collect", "shared"):
            k = kind[1]
            key = ("c", k)
            if t == "shared" and key in shared_cache:
                c = shared_cache[key]
            else:
                m = new("cm")
                nodes.append(MarkedAnyTypeTestInvocation(id=m, marker="x"))
                edges.append((f"it{k}", "item", m, "value"))
                c = new("col")
                nodes.append(CollectInvocation(id=c))
                edges.append((m, "value", c, "item"))
                if t == "shared":
                    shared_cache[key] = c
            n = new("p")
            nodes.append(MarkedAnyTypeTestInvocation(id=n, marker=marker))
            edges.append((c, "collection", n, "value"))
            return (n, "value")
        if t == "collect2":
            k = kind[1]
            m = new("cm")
            nodes.append(MarkedAnyTypeTestInvocation(id=m, marker="x"))
            edges.append((f"it{k}", "item", m, "value"))
            c = new("col")
            nodes.append(CollectInvocation(id=c))
            edges.append((m, "value", c, "item"))
            c2 = new("col")
            nodes.append(CollectInvocation(id=c2))
            edges.append((c, "collection", c2, "item"))
            return (c2, "collection")
        if t == "if":
            ik = r.choice(["litT", "litF", "rootT", "rootF", "item"])
            il = r.randrange(L)
            n = new("iif")
            kw = {"condition": ik == "litT"} if ik in ("litT", "litF") else {}
            a = branch(base(), "a" + marker)
            b = branch(base(), "b" + marker)
            nodes.append(IfInvocation(id=n, **kw))
            edges.extend([(*a, n, "true_input"), (*b, n, "false_input")])
            if ik in ("rootT", "rootF"):
                c = new("ic")
                nodes.append(BooleanInvocation(id=c, value=ik == "rootT"))
                edges.append((c, "value", n, "condition"))
            elif ik == "item":
                c = new("ic")
                nodes.append(FixedCollectionTestInvocation(id=c))
                edges.extend([(f"it{il}", "item", c, "value"), (c, "condition", n, "condition")])
            if r.random() < 0.5:
                return (n, "value")
            p = new("p")
            nodes.append(MarkedAnyTypeTestInvocation(id=p, marker=marker))
            edges.append((n, "value", p, "value"))
            return (p, "value")
        if t == "forE":
            f = new("for")
            b = new("fb")
            rt = new("ret")
            ei = new("eit")
            es = new("es")
            nodes.extend(
                [
                    CollectionConcatInvocation(id=es, first=[]),
                    IterateInvocation(id=ei),
                    ForInvocation(id=f),
                    MarkedAnyTypeTestInvocation(id=b, marker="F"),
                    ForReturnInvocation(id=rt),
                ]
            )
            edges.extend(
                [
                    (es, "collection", ei, "collection"),
                    (ei, "item", f, "collection"),
                    (f, "item", b, "value"),
                    (b, "value", rt, "output"),
                    create_loop_linkage(f, rt),
                ]
            )
            return (f, "output_collection")
        if t == "forR":
            f = new("for")
            b = new("fb")
            rt = new("ret")
            nodes.extend(
                [
                    ForInvocation(id=f, collection=r.choice([[], ["u"], ["u", "v"]])),
                    MarkedAnyTypeTestInvocation(id=b, marker="F"),
                    ForReturnInvocation(id=rt),
                ]
            )
            edges.extend([(f, "item", b, "value"), (b, "value", rt, "output"), create_loop_linkage(f, rt)])
            return (f, "output_collection")
        if t == "for":
            k = kind[1]
            direct = kind[2]
            f = new("for")
            b = new("fb")
            rt = new("ret")
            nodes.extend(
                [ForInvocation(id=f), MarkedAnyTypeTestInvocation(id=b, marker="F"), ForReturnInvocation(id=rt)]
            )
            edges.extend(
                [
                    (f"it{k}", "item", f, "collection"),
                    (f, "item", b, "value"),
                    (b, "value", rt, "output"),
                    create_loop_linkage(f, rt),
                ]
            )
            if direct:
                return (f, "output_collection")
            n = new("p")
            nodes.append(MarkedAnyTypeTestInvocation(id=n, marker=marker))
            edges.append((f, "output_collection", n, "value"))
            return (n, "value")

    def base():
        opts = [
            ("root",),
            ("iter", r.randrange(L)),
            ("collect", r.randrange(L)),
            ("for", r.randrange(L), r.random() < 0.5),
            ("forR",),
        ]
        return r.choice(opts)

    def rkind():
        opts = [("root",), ("iter", r.randrange(L)), ("collect", r.randrange(L)), ("shared", r.randrange(L))]
        if L >= 2:
            opts.append(("collect2", r.randrange(1, L)))
        opts.append(("for", r.randrange(L), r.random() < 0.5))
        opts += [("if",), ("if",), ("forE",), ("forR",)]
        return r.choice(opts)

    tk, fk = rkind(), rkind()
    if r.random() < 0.3:
        fk = tk if tk[0] != "for" else fk
    ckind = r.choice(["litT", "litF", "rootT", "rootF", "item", "item"])
    clevel = r.randrange(L)
    tsrc = branch(tk, "T:")
    fsrc = branch(fk, "F:")
    ifkw = {"condition": True} if ckind == "litT" else ({"condition": False} if ckind == "litF" else {})
    nodes.append(IfInvocation(id="if", **ifkw))
    edges.extend([(*tsrc, "if", "true_input"), (*fsrc, "if", "false_input")])
    if ckind in ("rootT", "rootF"):
        nodes.append(BooleanInvocation(id="cond", value=ckind == "rootT"))
        edges.append(("cond", "value", "if", "condition"))
    elif ckind == "item":
        nodes.append(FixedCollectionTestInvocation(id="cond"))
        edges.extend([(f"it{clevel}", "item", "cond", "value"), ("cond", "condition", "if", "condition")])
    outsrc = ("if", "value")
    if r.random() < 0.3:
        o = branch(base(), "O:")
        nodes.append(IfInvocation(id="if2", condition=r.random() < 0.5))
        edges.extend([("if", "value", "if2", "true_input"), (*o, "if2", "false_input")])
        outsrc = ("if2", "value")
    nodes.append(CollectInvocation(id="out"))
    edges.append((*outsrc, "out", "item"))
    extra = r.choice(["none", "sibroot", "sibiter", "sibcoll", "sibchain", "sibsame"])
    if extra == "sibroot":
        nodes.append(AnyTypeTestInvocation(id="sx", value="S"))
        edges.append(("sx", "value", "out", "item"))
    elif extra == "sibcoll":
        nodes.append(CollectionConcatInvocation(id="sx", first=["S"]))
        edges.append(("sx", "collection", "out", "collection"))
    elif extra in ("sibiter", "sibchain"):
        nodes += [
            CollectionConcatInvocation(id="sbs", first=r.choice([[], ["p"], ["p", "q", "r"]])),
            IterateInvocation(id="itb"),
            MarkedAnyTypeTestInvocation(id="sx", marker="n"),
        ]
        edges += [("sbs", "collection", "itb", "collection"), ("itb", "item", "sx", "value")]
        last = ("sx", "value")
        if extra == "sibchain":
            nodes += [
                IfInvocation(id="sj0", condition=True),
                AnyTypeTestInvocation(id="sjr", value="J"),
                IfInvocation(id="sj1", condition=r.random() < 0.5),
            ]
            edges += [
                ("sx", "value", "sj0", "true_input"),
                ("sjr", "value", "sj0", "false_input"),
                ("sj0", "value", "sj1", "true_input"),
                ("sj0", "value", "sj1", "false_input"),
            ]
            last = ("sj1", "value")
        edges.append((*last, "out", "item"))
    elif extra == "sibsame":
        b = branch(base(), "S:")
        edges.append((*b, "out", "item"))
    tail = r.choice(["none", "out2", "reif"])
    if tail == "out2":
        nodes.append(CollectInvocation(id="out2"))
        edges.append(("out", "collection", "out2", "item"))
    elif tail == "reif":
        # If -> Collect -> Iterate -> If -> Collect
        rk = r.choice(["lit", "root", "item"])
        nodes += [
            IterateInvocation(id="rit"),
            MarkedAnyTypeTestInvocation(id="rm", marker="r"),
            AnyTypeTestInvocation(id="rr", value="RR"),
            IfInvocation(id="rif", **({"condition": r.random() < 0.5} if rk == "lit" else {})),
            CollectInvocation(id="out2"),
        ]
        edges += [
            ("out", "collection", "rit", "collection"),
            ("rit", "item", "rm", "value"),
            ("rm", "value", "rif", "true_input"),
            ("rr", "value", "rif", "false_input"),
            ("rif", "value", "out2", "item"),
        ]
        if rk == "root":
            nodes.append(BooleanInvocation(id="rc", value=r.random() < 0.5))
            edges.append(("rc", "value", "rif", "condition"))
        if rk == "item":
            nodes.append(FixedCollectionTestInvocation(id="rc"))
            edges += [("rit", "item", "rc", "value"), ("rc", "condition", "rif", "condition")]
    used.append(extra)
    used.append(tail)
    desc = f"used={used} if2={'if2' in [n.id for n in nodes]} L={L} src={src} T={tk} F={fk} cond={ckind}@{clevel}"
    return desc, (lambda: G(list(nodes), list(edges)))


_KNOWN_VALID_SEEDS = frozenset(
    {
        3,
        52,
        233,
        333,
        573,
        630,
        880,
        887,
        992,
        1300,
        1331,
        1593,
        1728,
        1836,
        1906,
        1940,
        1970,
        2047,
        2057,
        2355,
        2382,
        2394,
        2575,
        2667,
        2775,
        2829,
    }
)


@pytest.mark.parametrize("seed", [52, 887, 1593, 2667], ids=lambda seed: f"seed{seed}")
def test_empty_nested_if_frames_retire_branch_descendants(seed: int) -> None:
    desc, graph_factory = build(seed)
    graph_factory().validate_self()
    results = {
        (forced, checkpoint): _run(seed, forced_compatibility=forced, roundtrip_at=checkpoint)
        for forced in (False, True)
        for checkpoint in (None, 3, 4, 5)
    }
    assert all(result == results[(False, None)] for result in results.values()), f"seed={seed} graph={desc}"


def _canonical_result(graph: Graph, node_id: str, value: Any) -> Any:
    if not isinstance(value, list):
        return value
    node = graph.get_node(node_id)
    if isinstance(node, CollectInvocation):
        edges = graph._get_input_edges(node_id, "item")
        if len(edges) == 1 and isinstance(graph.get_node(edges[0].source.node_id), CollectInvocation):
            value = [_canonical_result(graph, edges[0].source.node_id, item) for item in value]
        # Collect aggregates independent sources, so only its outer item order is unspecified.
        return tuple(sorted(value, key=repr))
    if isinstance(node, IfInvocation):
        branch_edges = graph._get_input_edges(node_id, "true_input") + graph._get_input_edges(node_id, "false_input")
        if any(isinstance(graph.get_node(edge.source.node_id), CollectInvocation) for edge in branch_edges):
            return tuple(sorted(value, key=repr))
    return value


def _normalize_collect_fed_iterate_permutations(
    graph: Graph,
    state: GraphExecutionState,
    outputs: dict[str, list[tuple[tuple[int, ...], Any]]],
    execution_ids: dict[str, list[str]],
) -> dict[str, list[tuple[tuple[int, ...], Any]]]:
    """Normalize only frame-to-value assignments on axes fed by Collect item order."""
    collect_fed_iterators = {
        edge.destination.node_id
        for edge in graph.edges
        if edge.destination.field == "collection"
        and isinstance(graph.get_node(edge.source.node_id), CollectInvocation)
        and isinstance(graph.get_node(edge.destination.node_id), IterateInvocation)
    }
    if not collect_fed_iterators:
        return outputs

    execution_graph = state._get_execution_graph_flat()
    for node_id, executions in outputs.items():
        grouped_positions: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for position, (path, _) in enumerate(executions):
            execution_id = execution_ids[node_id][position]
            matching_axis_indices: set[int] = set()
            for iterator_id in sorted(collect_fed_iterators):
                pending = list(execution_graph.predecessors(execution_id))
                visited: set[str] = set()
                while pending:
                    candidate_id = pending.pop()
                    if candidate_id in visited:
                        continue
                    visited.add(candidate_id)
                    candidate_node = state.execution_graph.get_node(candidate_id)
                    if (
                        state.prepared_source_mapping.get(candidate_id) == iterator_id
                        and isinstance(candidate_node, IterateInvocation)
                        and candidate_node.index >= 0
                    ):
                        iteration_path = state._get_iteration_path(candidate_id)
                        if len(iteration_path) <= len(path) and path[: len(iteration_path)] == iteration_path:
                            matching_axis_indices.add(len(iteration_path) - 1)
                    pending.extend(execution_graph.predecessors(candidate_id))

            for axis_index in matching_axis_indices:
                frame_without_axis = path[:axis_index], path[axis_index + 1 :]
                grouped_positions.setdefault((axis_index, *frame_without_axis), []).append(position)

        if grouped_positions:
            normalized_executions = list(executions)
            for (axis_index, _, _), positions in grouped_positions.items():
                ordered_positions = sorted(
                    positions, key=lambda position: (executions[position][0][axis_index], position)
                )
                ordered_values = sorted((executions[position][1] for position in positions), key=repr)
                for position, value in zip(ordered_positions, ordered_values, strict=True):
                    normalized_executions[position] = (executions[position][0], value)
            executions[:] = normalized_executions

    return outputs


def _run(
    seed: int, *, forced_compatibility: bool, roundtrip_at: int | None
) -> dict[str, list[tuple[tuple[int, ...], Any]]]:
    desc, graph_factory = build(seed)
    graph = graph_factory()
    state = GraphExecutionState(graph=graph)
    if forced_compatibility:
        state._execution_scheduler = _ExecutionScheduler(state)
    selected_route = state._scheduler()
    route_type = type(selected_route)
    trace: list[tuple[str, tuple[int, ...]]] = []
    completed_before_reload: set[str] = set()
    reloaded = False
    try:
        for step in range(2000):
            if roundtrip_at == step:
                completed_before_reload = set(state.results)
                reloaded = True
                state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
                state._execution_scheduler = (
                    _ExecutionScheduler(state)
                    if route_type is _ExecutionScheduler
                    else _GenericGraphSchedulerAdapter(state)
                )
                assert type(state._scheduler()) is route_type
            node = state.next()
            if reloaded:
                assert node is None or node.id not in completed_before_reload, (
                    f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} "
                    f"replayed completed execution {node.id if node else None}"
                )
            if node is None:
                break
            sid = state.prepared_source_mapping[node.id]
            trace.append((sid, state._get_iteration_path(node.id)))
            state.complete(node.id, node.invoke(Mock()))
        else:
            pytest.fail(
                f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} exceeded 2000 steps; trace={trace[-30:]}"
            )
    except UnsupportedIfIterationAxesError:
        if trace:
            pytest.fail(
                f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} axis rejection after invocation; trace={trace[-30:]}"
            )
        raise
    except Exception as error:
        pytest.fail(
            f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} "
            f"raised {type(error).__name__}: {error}; trace={trace[-30:]}"
        )

    assert state.is_complete(), (
        f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} stalled; trace={trace[-30:]}"
    )
    assert not state.has_error(), (
        f"seed={seed} graph={desc} route={route_type.__name__} checkpoint={roundtrip_at} ended with execution errors"
    )
    outputs: dict[str, list[tuple[tuple[int, ...], Any]]] = {}
    output_execution_ids: dict[str, list[str]] = {}
    for node_id, node in graph.nodes.items():
        if not isinstance(node, (CollectInvocation, IfInvocation)):
            continue
        executions = []
        for exec_id in state._prepared_registry().get_prepared_ids(node_id):
            if exec_id in state.results:
                result = state.results[exec_id]
                value = getattr(result, "collection", getattr(result, "value", None))
                executions.append(
                    (tuple(state._get_iteration_path(exec_id)), _canonical_result(graph, node_id, value), exec_id)
                )
        executions.sort(key=lambda item: item[0])
        outputs[node_id] = [(path, value) for path, value, _ in executions]
        output_execution_ids[node_id] = [exec_id for _, _, exec_id in executions]
    return _normalize_collect_fed_iterate_permutations(graph, state, outputs, output_execution_ids)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3000), ids=lambda seed: f"seed{seed}")
def test_execution_engine_fuzz_routes_and_snapshots(seed: int) -> None:
    try:
        desc, graph_factory = build(seed)
        graph_factory().validate_self()
    except UnsupportedIfIterationAxesError as error:
        if seed in _KNOWN_VALID_SEEDS:
            pytest.fail(f"seed={seed} graph={desc}: known-valid seed rejected before invocation: {error}")
        pytest.skip(f"seed={seed} graph={desc}: expected pre-invocation axis rejection: {error}")

    results = {}
    for forced in (False, True):
        for checkpoint in (None, 3, 4, 5):
            route_name = "compat" if forced else "auto"
            try:
                results[(route_name, checkpoint)] = _run(seed, forced_compatibility=forced, roundtrip_at=checkpoint)
            except UnsupportedIfIterationAxesError as error:
                if seed in _KNOWN_VALID_SEEDS:
                    pytest.fail(
                        f"seed={seed} graph={desc} route={route_name} checkpoint={checkpoint}: "
                        f"known-valid seed rejected before invocation: {error}"
                    )
                pytest.skip(
                    f"seed={seed} graph={desc} route={route_name} checkpoint={checkpoint}: "
                    f"expected pre-invocation axis rejection: {error}"
                )

    reference = results[("auto", None)]
    if seed == 1681:
        expected = {
            "out": [((0,), ()), ((1,), ("S:['Fa', 'Fb']", "T:['a', 'b']"))],
            "out2": [((0,), ()), ((1,), ("rS:['Fa', 'Fb']", "rT:['a', 'b']"))],
            "if": [((1, 0), "T:['a', 'b']")],
        }
        canary = {node_id: reference[node_id] for node_id in expected}
        assert canary == expected, (
            f"seed=1681 graph={desc}: expected duplicate-sensitive per-execution result {expected}, got {canary}"
        )

    for key, result in results.items():
        assert result == reference, f"seed={seed} graph={desc} route={key}: expected {reference}, got {result}"


def test_seed_2220_rejects_unsupported_axes_before_invocation() -> None:
    with pytest.raises(UnsupportedIfIterationAxesError):
        _run(2220, forced_compatibility=False, roundtrip_at=None)
