"""The FLUX.3 layout canvas graphs webv2 compiles are graphs this backend accepts, and box what was drawn.

`flux3LayoutCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with `vitest -u`) from
the real canvas compiler: a txt2img, an inpaint, and a txt2img edit of a reference image, each on a 768x1024 canvas
with two Regional Guidance layers, one inside the frame with quotes and non-ASCII text in its prompt, and one ending on
the frame's right and bottom edges. pydantic validates every node against its invocation, the regions against
`Flux3LayoutRegion` included; `Graph.validate_self` checks every edge type, the one into `core_metadata.flux3_caption`
included; and the layout node runs on its literal inputs. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.flux3.flux3_layout_prompt import Flux3LayoutPromptInvocation
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
    / "flux3LayoutCanvasGraphs.json"
)

GRAPHS: dict[str, Any] = {
    name: graph for name, graph in json.loads(FIXTURE_PATH.read_text(encoding="utf-8")).items() if name != "_comment"
}

# What each graph's layout node writes for the stand-in prompt "A sunny beach.". 768x1024 in 0..1000,
# [y_min, x_min, y_max, x_max]: the kite spans x 192..576 and y 64..320, the dog x 512..768 and y 640..1024; 62.5 and
# 312.5 round up, 666.7 to 667. The edit has a reference image, so its boxes place new content in the output.
OBJECT_ROWS = (
    'A sunny beach. [{"id": "obj_1", "bbox": [63, 250, 313, 750], "desc": "a red kite reading \\"Éole\\""}, '
    '{"id": "obj_2", "bbox": [625, 667, 1000, 1000], "desc": "a dog digging in the sand"}]'
)
EXPECTED_CAPTIONS = {
    "txt2img": OBJECT_ROWS,
    "inpaint": OBJECT_ROWS,
    "edit": (
        'A sunny beach. [{"id": "obj_1", "from": "target", "src_bbox": null, "tgt_bbox": [63, 250, 313, 750], '
        '"desc": "a red kite reading \\"Éole\\""}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [625, 667, 1000, 1000], '
        '"desc": "a dog digging in the sand"}]'
    ),
}


@pytest.fixture(params=sorted(GRAPHS), ids=str)
def graph(request: pytest.FixtureRequest) -> Graph:
    graph = Graph.model_validate(GRAPHS[request.param])
    graph.validate_self()
    return graph


def _source(graph: Graph, node_id: str, field: str) -> tuple[str, str]:
    (edge,) = [e for e in graph.edges if e.destination.node_id == node_id and e.destination.field == field]
    return edge.source.node_id, edge.source.field


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`prompt` for a region's `desc`) would validate and run at
    its default. Region values must also survive validation unchanged."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"

    layout = graph.nodes["layout_prompt"]
    assert isinstance(layout, Flux3LayoutPromptInvocation)
    assert [region.model_dump() for region in layout.regions] == GRAPHS[name]["nodes"]["layout_prompt"]["regions"]


def test_the_layout_prompt_is_what_is_encoded_and_recorded(graph: Graph) -> None:
    # The prompt that batching varies feeds the layout node, whose output is the only thing the encoder sees.
    assert _source(graph, "layout_prompt", "prompt") == ("positive_prompt", "value")
    assert _source(graph, "pos_cond", "prompt") == ("layout_prompt", "value")
    metadata_id = next(node.id for node in graph.nodes.values() if node.get_type() == "core_metadata")
    assert _source(graph, metadata_id, "flux3_caption") == ("layout_prompt", "value")
    # Recall restores the prompt the user typed, not the caption with its box table.
    assert _source(graph, metadata_id, "positive_prompt") == ("positive_prompt", "value")


@pytest.mark.parametrize("name", sorted(EXPECTED_CAPTIONS))
def test_the_layout_node_places_the_drawn_regions(name: str) -> None:
    layout = Graph.model_validate(GRAPHS[name]).nodes["layout_prompt"]
    assert isinstance(layout, Flux3LayoutPromptInvocation)
    # The positive prompt arrives through the edge at run time; this stands in for the batch value.
    runtime = layout.model_copy(update={"prompt": "A sunny beach."})

    assert runtime.invoke(None).value == EXPECTED_CAPTIONS[name]  # type: ignore[arg-type]
