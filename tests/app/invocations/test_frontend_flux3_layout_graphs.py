"""The FLUX.3 layout canvas graphs webv2 compiles are graphs this backend accepts, and box what was drawn.

`flux3LayoutCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with `vitest -u`) from
the real canvas compiler: a txt2img, an inpaint, and a txt2img edit of a reference image, each on a 768x1024 canvas
with two Regional Guidance layers, one inside the frame with quotes and non-ASCII text in its prompt, and one ending on
the frame's right and bottom edges. pydantic validates every node against its invocation, the regions against
`Flux3LayoutRegion` included; `Graph.validate_self` checks every edge type, the one into `core_metadata.flux3_caption`
included; and the layout node runs on its literal inputs. No model is loaded.

The `upsampled_*` graphs are the same canvases with the prompt upsampler on: it takes the layout node's place and its
regions, so they must hold no `layout_prompt`, and the rows the upsampler node locks for the stand-in prompt must be the
rows the layout node writes into the caption.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.flux3.flux3_layout_prompt import Flux3LayoutPromptInvocation
from invokeai.app.invocations.flux3.flux3_prompt_upsampler import Flux3PromptUpsamplerInvocation
from invokeai.app.services.shared.graph import Graph
from invokeai.backend.flux3.layout import flux3_locked_rows

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
UPSAMPLED_PREFIX = "upsampled_"
UPSAMPLED_GRAPHS = sorted(name for name in GRAPHS if name.startswith(UPSAMPLED_PREFIX))
PROMPT = "A sunny beach."

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


def _caption_node_id(name: str) -> str:
    return "prompt_upsampler" if name.startswith(UPSAMPLED_PREFIX) else "layout_prompt"


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

    node_id = _caption_node_id(name)
    caption_node = graph.nodes[node_id]
    assert isinstance(caption_node, Flux3LayoutPromptInvocation | Flux3PromptUpsamplerInvocation)
    assert [region.model_dump() for region in caption_node.regions] == GRAPHS[name]["nodes"][node_id]["regions"]


def test_the_layout_prompt_is_what_is_encoded_and_recorded(graph: Graph) -> None:
    # The prompt that batching varies feeds the caption node, whose output is the only thing the encoder sees.
    caption_id = "prompt_upsampler" if "prompt_upsampler" in graph.nodes else "layout_prompt"
    assert _source(graph, caption_id, "prompt") == ("positive_prompt", "value")
    assert _source(graph, "pos_cond", "prompt") == (caption_id, "value")
    metadata_id = next(node.id for node in graph.nodes.values() if node.get_type() == "core_metadata")
    assert _source(graph, metadata_id, "flux3_caption") == (caption_id, "value")
    # Recall restores the prompt the user typed, not the caption with its box table.
    assert _source(graph, metadata_id, "positive_prompt") == ("positive_prompt", "value")


@pytest.mark.parametrize("name", sorted(EXPECTED_CAPTIONS))
def test_the_layout_node_places_the_drawn_regions(name: str) -> None:
    layout = Graph.model_validate(GRAPHS[name]).nodes["layout_prompt"]
    assert isinstance(layout, Flux3LayoutPromptInvocation)
    # The positive prompt arrives through the edge at run time; this stands in for the batch value.
    runtime = layout.model_copy(update={"prompt": "A sunny beach."})

    assert runtime.invoke(None).value == EXPECTED_CAPTIONS[name]  # type: ignore[arg-type]


@pytest.mark.parametrize("name", UPSAMPLED_GRAPHS)
def test_the_upsampler_replaces_the_layout_node_and_sees_what_the_denoiser_sees(name: str) -> None:
    graph = Graph.model_validate(GRAPHS[name])
    graph.validate_self()
    assert not any(node.get_type() == "flux3_layout_prompt" for node in graph.nodes.values())
    node = graph.nodes["prompt_upsampler"]
    assert isinstance(node, Flux3PromptUpsamplerInvocation)
    assert node.upsampler_model.key == "pup"
    # The edit derives its edit targets from the references: the one collect feeds the upsampler and the denoiser.
    denoise_id = next(n.id for n in graph.nodes.values() if n.get_type() == "flux3_denoise")
    if name == "upsampled_edit":
        assert _source(graph, "prompt_upsampler", "reference_images") == _source(
            graph, denoise_id, "kontext_conditioning"
        )
    else:
        assert not [e for e in graph.edges if e.destination.field == "reference_images"]


@pytest.mark.parametrize("name", UPSAMPLED_GRAPHS)
def test_the_upsampler_locks_the_rows_the_layout_node_writes(name: str) -> None:
    graph = Graph.model_validate(GRAPHS[name])
    node = graph.nodes["prompt_upsampler"]
    assert isinstance(node, Flux3PromptUpsamplerInvocation)
    has_references = any(
        e.destination.node_id == "prompt_upsampler" and e.destination.field == "reference_images" for e in graph.edges
    )

    caption, rows = flux3_locked_rows(PROMPT, node.regions, edit_targets=has_references)

    layout_caption = EXPECTED_CAPTIONS[name.removeprefix(UPSAMPLED_PREFIX)]
    assert caption == PROMPT
    assert rows == json.loads(layout_caption.removeprefix(PROMPT + " "))
