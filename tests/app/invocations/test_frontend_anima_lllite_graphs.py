"""The Anima ControlNet-LLLite canvas graphs webv2 compiles are graphs this backend accepts.

`animaLLLiteCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with
`vitest -u`) from the real canvas compiler: a complete inpainting submission with three control-layer
adapters, and an outpainting submission that adds the optional 4-channel inpaint adapter beside a control
layer. The frontend routes any number of adapters through one collector, so these graphs cover the edge
types for every count, masked and unmasked. This module checks more than names: pydantic validates every
node against its invocation (bounds such as `anima_lllite.weight` in [-10, 10] included),
`Graph.validate_self` checks every edge type through the collector into `anima_denoise.control_lllite`,
and the denoiser's own normalization runs on the fields the adapter nodes would emit. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.anima.anima_lllite import AnimaLLLiteField, AnimaLLLiteInvocation
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.image import ImageLerpInvocation
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


INPAINT_ADAPTER_NODE_ID = "anima_inpaint_adapter"


def _lllite_nodes(graph: Graph) -> list[AnimaLLLiteInvocation]:
    return [node for node in graph.nodes.values() if isinstance(node, AnimaLLLiteInvocation)]


def _control_layer_nodes(graph: Graph) -> list[AnimaLLLiteInvocation]:
    return [node for node in _lllite_nodes(graph) if node.id != INPAINT_ADAPTER_NODE_ID]


def _control_field(graph: Graph, node: AnimaLLLiteInvocation) -> AnimaLLLiteField:
    """The field the node would emit. `invoke` only repackages the node's own fields, so an image or mask that an
    edge supplies at run time stands in as the image its source node would produce."""
    fed = {e.destination.field for e in graph.edges if e.destination.node_id == node.id}
    runtime = node.model_copy(
        update={field: ImageField(image_name=f"{node.id}.{field}") for field in fed if field in {"image", "mask"}}
    )
    return runtime.invoke(None).control  # type: ignore[arg-type]


def test_the_fixtures_combine_several_distinct_adapters() -> None:
    keys = {
        name: [node["control_model"]["key"] for node in graph["nodes"].values() if node["type"] == "anima_lllite"]
        for name, graph in GRAPHS.items()
    }
    assert len(keys["inpaint_three_adapters"]) == 3 and len(set(keys["inpaint_three_adapters"])) == 3
    assert len(keys["outpaint_inpaint_adapter_and_control_layer"]) == 2
    assert len(set(keys["outpaint_inpaint_adapter_and_control_layer"])) == 2


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`control_weight` for `weight`) would validate and run
    at its default. Values must also survive validation unchanged rather than being coerced."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"
        if isinstance(node, AnimaLLLiteInvocation):
            # The inpaint adapter leaves the step range to the node's defaults, as the legacy graph did.
            written = {
                field: raw[field] for field in ("weight", "begin_step_percent", "end_step_percent") if field in raw
            }
            assert "weight" in written
            assert {field: getattr(node, field) for field in written} == written


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
    control_layers = _control_layer_nodes(graph)
    assert control_layers
    for node in control_layers:
        assert node.image.image_name
        assert node.mask is None
        assert node.control_model.base.value == "anima"
        assert node.control_model.type.value == "controlnet"
    control_layer_ids = {node.id for node in control_layers}
    assert not any(e.destination.node_id in control_layer_ids for e in graph.edges)


def test_the_inpaint_adapter_takes_the_infilled_image_and_an_inverted_mask() -> None:
    """The adapter wants white = inpaint; the canvas mask is white = keep, so an `img_lerp` 255 -> 0 feeds its mask."""
    graph = Graph.model_validate(GRAPHS["outpaint_inpaint_adapter_and_control_layer"])
    graph.validate_self()
    adapter = graph.nodes[INPAINT_ADAPTER_NODE_ID]
    assert isinstance(adapter, AnimaLLLiteInvocation)
    assert adapter.weight == 0.75
    assert adapter.control_model.base.value == "anima"

    inputs = {e.destination.field: e.source for e in graph.edges if e.destination.node_id == adapter.id}
    assert set(inputs) == {"image", "mask"}
    assert graph.nodes[inputs["image"].node_id].get_type() == "infill_lama"
    assert inputs["image"].field == "image"
    invert = graph.nodes[inputs["mask"].node_id]
    assert isinstance(invert, ImageLerpInvocation)
    assert (invert.min, invert.max) == (255, 0)
    assert inputs["mask"].field == "image"
    mask_source = next(e.source for e in graph.edges if e.destination.node_id == invert.id)
    assert graph.nodes[mask_source.node_id].get_type() == "mask_combine"


def test_the_denoiser_accepts_the_adapters_it_would_receive(graph: Graph) -> None:
    fields = [_control_field(graph, node) for node in _lllite_nodes(graph)]

    normalized = AnimaDenoiseInvocation._normalize_control_lllite(fields)

    assert sorted(f.control_model.key for f in normalized) == sorted(f.control_model.key for f in fields)
    assert [f.mask_name is not None for f in fields] == [
        node.id == INPAINT_ADAPTER_NODE_ID for node in _lllite_nodes(graph)
    ]


def test_a_repeated_adapter_model_is_one_the_denoiser_refuses() -> None:
    """The frontend blocks this case; pin that the backend constraint it mirrors still exists."""
    graph = Graph.model_validate(GRAPHS["inpaint_three_adapters"])
    field = _control_field(graph, _lllite_nodes(graph)[0])

    with pytest.raises(ValueError, match="used by more than one control input"):
        AnimaDenoiseInvocation._normalize_control_lllite([field, field])
