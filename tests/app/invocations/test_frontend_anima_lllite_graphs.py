"""The Anima ControlNet-LLLite canvas graphs webv2 compiles are graphs this backend accepts.

`animaLLLiteCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with
`vitest -u`) from the real canvas compiler: a complete inpainting submission with three adapters. The
frontend routes any number of adapters through one collector, so one graph covers the edge types for
every count. This module checks more than names: pydantic validates every node against its invocation (bounds such as
`anima_lllite.weight` in [-10, 10] included), `Graph.validate_self` checks every edge type through the
collector into `anima_denoise.control_lllite`, and the denoiser's own normalization runs on the
fields the adapter nodes would emit. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.anima.anima_lllite import AnimaLLLiteInvocation
from invokeai.app.services.shared.graph import Graph

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "invokeai"
    / "frontend"
    / "webv2"
    / "src"
    / "features"
    / "generation"
    / "core"
    / "__snapshots__"
    / "animaLLLiteCanvasGraphs.json"
)

GRAPHS: dict[str, Any] = {
    name: graph for name, graph in json.loads(FIXTURE_PATH.read_text(encoding="utf-8")).items() if name != "_comment"
}


@pytest.fixture(params=sorted(GRAPHS), ids=str)
def graph(request: pytest.FixtureRequest) -> Graph:
    graph = Graph.model_validate(GRAPHS[request.param])
    graph.validate_self()
    return graph


def _lllite_nodes(graph: Graph) -> list[AnimaLLLiteInvocation]:
    return [node for node in graph.nodes.values() if isinstance(node, AnimaLLLiteInvocation)]


def test_the_fixture_combines_several_distinct_adapters() -> None:
    (graph,) = GRAPHS.values()
    keys = [node["control_model"]["key"] for node in graph["nodes"].values() if node["type"] == "anima_lllite"]
    assert len(keys) == 3 and len(set(keys)) == 3


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`control_weight` for `weight`) would validate and run
    at its default. Values must also survive validation unchanged rather than being coerced."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"
        if isinstance(node, AnimaLLLiteInvocation):
            assert (node.weight, node.begin_step_percent, node.end_step_percent) == (
                raw["weight"],
                raw["begin_step_percent"],
                raw["end_step_percent"],
            )


def test_every_adapter_reaches_control_lllite_through_one_collector(graph: Graph) -> None:
    denoise = [node for node in graph.nodes.values() if isinstance(node, AnimaDenoiseInvocation)]
    assert len(denoise) == 1

    into_denoise = [e for e in graph.edges if e.destination.node_id == denoise[0].id]
    control_edges = [e for e in into_denoise if e.destination.field == "control_lllite"]
    assert len(control_edges) == 1
    collector_id = control_edges[0].source.node_id
    assert graph.nodes[collector_id].get_type() == "collect"

    collected = {e.source.node_id for e in graph.edges if e.destination.node_id == collector_id}
    assert collected == {node.id for node in _lllite_nodes(graph)}
    assert all(e.source.field == "control" for e in graph.edges if e.destination.node_id == collector_id)


def test_control_layers_send_a_control_image_and_no_inpaint_mask(graph: Graph) -> None:
    for node in _lllite_nodes(graph):
        assert node.image.image_name
        assert node.mask is None
        assert node.control_model.base.value == "anima"
        assert node.control_model.type.value == "controlnet"
    assert not any(isinstance(graph.nodes[e.destination.node_id], AnimaLLLiteInvocation) for e in graph.edges)


def test_the_denoiser_accepts_the_adapters_it_would_receive(graph: Graph) -> None:
    # `invoke` only repackages the node's own fields; it reads nothing from the context.
    fields = [node.invoke(None).control for node in _lllite_nodes(graph)]  # type: ignore[arg-type]

    normalized = AnimaDenoiseInvocation._normalize_control_lllite(fields)

    assert sorted(f.control_model.key for f in normalized) == sorted(f.control_model.key for f in fields)


def test_a_repeated_adapter_model_is_one_the_denoiser_refuses() -> None:
    """The frontend blocks this case; pin that the backend constraint it mirrors still exists."""
    node = _lllite_nodes(Graph.model_validate(GRAPHS["inpaint_three_adapters"]))[0]
    field = node.invoke(None).control  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="used by more than one control input"):
        AnimaDenoiseInvocation._normalize_control_lllite([field, field])
