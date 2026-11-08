"""The MOSS-Image layout canvas graphs webv2 compiles are graphs this backend accepts, and lay out what was drawn.

`mossImageLayoutCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with `vitest -u`)
from the real canvas compiler: a txt2img and an inpaint submission on a 768x1024 canvas with three Regional Guidance
layers, one of each layout kind (a text region whose text has a line break, a character, an object touching the
right and bottom edges). pydantic validates every node against its invocation, the regions against
`MossImageLayoutRegion` included; `Graph.validate_self` checks every edge type, the one into
`core_metadata.moss_image_layout_prompt` included; and the layout node runs on its literal inputs. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.moss_image.moss_image_layout_prompt import MossImageLayoutPromptInvocation
from invokeai.app.services.shared.graph import Graph
from invokeai.backend.moss_image.layout import MossImageLayoutRegion

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
    / "mossImageLayoutCanvasGraphs.json"
)

GRAPHS: dict[str, Any] = {
    name: graph for name, graph in json.loads(FIXTURE_PATH.read_text(encoding="utf-8")).items() if name != "_comment"
}


@pytest.fixture(params=sorted(GRAPHS), ids=str)
def graph(request: pytest.FixtureRequest) -> Graph:
    graph = Graph.model_validate(GRAPHS[request.param])
    graph.validate_self()
    return graph


def _source(graph: Graph, node_id: str, field: str) -> tuple[str, str]:
    (edge,) = [e for e in graph.edges if e.destination.node_id == node_id and e.destination.field == field]
    return edge.source.node_id, edge.source.field


def test_the_fixtures_cover_a_txt2img_and_an_inpaint_submission() -> None:
    modes = {
        name: next(node["generation_mode"] for node in graph["nodes"].values() if node["type"] == "core_metadata")
        for name, graph in GRAPHS.items()
    }
    assert modes == {"txt2img": "moss_image_txt2img", "inpaint": "moss_image_inpaint"}


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`label` for a region's `kind`, `txt` for its `text`)
    would validate and run at its default. Region values must also survive validation unchanged."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"

    layout = graph.nodes["layout_prompt"]
    assert isinstance(layout, MossImageLayoutPromptInvocation)
    raw_regions = GRAPHS[name]["nodes"]["layout_prompt"]["regions"]
    for raw_region in raw_regions:
        assert set(raw_region) <= set(MossImageLayoutRegion.model_fields)
    assert [region.model_dump(exclude_none=True) for region in layout.regions] == raw_regions


def test_the_layout_prompt_is_what_is_encoded_and_recorded(graph: Graph) -> None:
    # The prompt that batching varies feeds the layout node, whose output is the only thing the encoder sees.
    assert _source(graph, "layout_prompt", "prompt") == ("positive_prompt", "value")
    assert _source(graph, "pos_cond", "prompt") == ("layout_prompt", "value")
    metadata_id = next(node.id for node in graph.nodes.values() if node.get_type() == "core_metadata")
    assert _source(graph, metadata_id, "moss_image_layout_prompt") == ("layout_prompt", "value")
    # Recall restores the prompt the user typed, not the compiled layout.
    assert _source(graph, metadata_id, "positive_prompt") == ("positive_prompt", "value")


def test_the_boxes_are_laid_out_on_the_image_being_denoised(graph: Graph) -> None:
    layout = graph.nodes["layout_prompt"]
    denoise = graph.nodes["denoise_latents"]
    assert isinstance(layout, MossImageLayoutPromptInvocation)
    assert (layout.width, layout.height) == (denoise.width, denoise.height) == (768, 1024)


def test_the_layout_node_compiles_the_drawn_regions(graph: Graph) -> None:
    layout = graph.nodes["layout_prompt"]
    assert isinstance(layout, MossImageLayoutPromptInvocation)
    # The positive prompt arrives through the edge at run time; this stands in for the batch value.
    runtime = layout.model_copy(update={"prompt": "A cosy cafe at dawn."})

    prompt = runtime.invoke(None).value  # type: ignore[arg-type]

    # 768x1024: x - 384 and y - 512, in [y1, x1, y2, x2]; the object box ends on the right and bottom edges.
    assert prompt == (
        '{"summary":{"caption":"A cosy cafe at dawn."},"items":['
        '{"label":"text","caption":"a hand-lettered cafe sign","bbox":[-512,-384,-256,128],"text":"一杯温暖\\n一刻从容"},'
        '{"label":"character","caption":"a barista pouring latte art","bbox":[-384,-320,-128,-128]},'
        '{"label":"object","caption":"a steaming ceramic cup","bbox":[128,128,512,384]}'
        '],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
    )
