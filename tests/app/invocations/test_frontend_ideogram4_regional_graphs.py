"""The Ideogram 4 regional canvas graph webv2 compiles is a graph this backend accepts, and captions what was drawn.

`ideogram4RegionalCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with `vitest -u`)
from the real canvas compiler: a txt2img submission on a 768x1024 canvas with two Regional Guidance layers, one inside
the frame and one ending on its right and bottom edges. pydantic validates every node against its invocation, the
regions against `Ideogram4Region` included; `Graph.validate_self` checks every edge type, the one into
`core_metadata.ideogram4_caption` included; and the caption builder runs on its literal inputs. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.ideogram4.ideogram4_caption import Ideogram4CaptionBuilderInvocation
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
    / "ideogram4RegionalCanvasGraphs.json"
)

RAW_GRAPH: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["txt2img"]


@pytest.fixture
def graph() -> Graph:
    graph = Graph.model_validate(RAW_GRAPH)
    graph.validate_self()
    return graph


def _source(graph: Graph, node_id: str, field: str) -> tuple[str, str]:
    (edge,) = [e for e in graph.edges if e.destination.node_id == node_id and e.destination.field == field]
    return edge.source.node_id, edge.source.field


def test_every_field_the_frontend_writes_is_one_the_invocation_declares(graph: Graph) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`desc` for a region's `prompt`) would validate and run
    at its default. Region values must also survive validation unchanged."""
    for node_id, raw in RAW_GRAPH["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"

    builder = graph.nodes["caption_builder"]
    assert isinstance(builder, Ideogram4CaptionBuilderInvocation)
    assert [region.model_dump() for region in builder.regions] == RAW_GRAPH["nodes"]["caption_builder"]["regions"]


def test_the_caption_is_what_is_encoded_and_recorded(graph: Graph) -> None:
    # The prompt that batching varies feeds the caption builder, whose output is the only thing the encoder sees.
    assert _source(graph, "caption_builder", "prompt") == ("positive_prompt", "value")
    assert _source(graph, "pos_cond", "prompt") == ("caption_builder", "value")
    metadata_id = next(node.id for node in graph.nodes.values() if node.get_type() == "core_metadata")
    assert _source(graph, metadata_id, "ideogram4_caption") == ("caption_builder", "value")
    # Recall restores the prompt the user typed, not the compiled caption.
    assert _source(graph, metadata_id, "positive_prompt") == ("positive_prompt", "value")


def test_the_caption_builder_places_the_drawn_regions(graph: Graph) -> None:
    builder = graph.nodes["caption_builder"]
    assert isinstance(builder, Ideogram4CaptionBuilderInvocation)
    # The positive prompt arrives through the edge at run time; this stands in for the batch value.
    runtime = builder.model_copy(update={"prompt": "A sunny beach."})

    caption = runtime.invoke(None).value  # type: ignore[arg-type]

    # 768x1024 in 0..1000, [y_min, x_min, y_max, x_max]: the kite spans x 192..576 and y 64..320, the dog
    # x 512..768 and y 640..1024; 62.5 and 312.5 round up, 666.7 to 667.
    assert caption == (
        '{"high_level_description":"A sunny beach.","compositional_deconstruction":{"background":"","elements":['
        '{"type":"obj","bbox":[63,250,313,750],"desc":"a red kite"},'
        '{"type":"obj","bbox":[625,667,1000,1000],"desc":"a dog digging in the sand"}'
        "]}}"
    )
