"""`moss_image_layout_prompt`: the regions a graph may send, the prompt the node hands on, and where it is recorded.

How boxes and prompts compile is covered with the builder in `tests/backend/moss_image/test_layout.py`.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.metadata import CoreMetadataInvocation
from invokeai.app.invocations.moss_image.moss_image_layout_prompt import MossImageLayoutPromptInvocation
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph


def _node(*regions: dict, **fields: object) -> MossImageLayoutPromptInvocation:
    # Regions arrive as JSON objects in the graph payload, so they are validated from dicts here too.
    return MossImageLayoutPromptInvocation.model_validate({"id": "layout_prompt", "regions": list(regions), **fields})


def test_the_node_lays_its_regions_out_on_its_own_width_and_height() -> None:
    # A portrait canvas: with width and height swapped the girl's box would leave the image.
    node = _node(
        {"kind": "character", "caption": "a girl", "bbox": [0, 0, 768, 1344]},
        {"kind": "text", "caption": "a title", "text": "MOSS", "bbox": [0, 0, 768, 200]},
        prompt="a portrait",
        width=768,
        height=1344,
    )

    assert node.invoke(MagicMock()).value == (
        '{"summary":{"caption":"a portrait"},"items":['
        '{"label":"character","caption":"a girl","bbox":[-672,-384,672,384]},'
        '{"label":"text","caption":"a title","bbox":[-672,-384,-472,384],"text":"MOSS"}'
        '],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
    )


@pytest.mark.parametrize(
    "region",
    [
        pytest.param({"kind": "panel", "caption": "a frame", "bbox": [0, 0, 16, 16]}, id="kind-not-offered"),
        pytest.param({"kind": "text", "caption": "a sign", "bbox": [0, 0, 16, 16]}, id="text-region-without-text"),
        pytest.param({"kind": "text", "caption": "a sign", "text": " \n", "bbox": [0, 0, 16, 16]}, id="blank-text"),
        pytest.param(
            {"kind": "object", "caption": "a sign", "text": "OPEN", "bbox": [0, 0, 16, 16]}, id="text-on-an-object"
        ),
        pytest.param({"kind": "object", "caption": "  ", "bbox": [0, 0, 16, 16]}, id="blank-caption"),
        pytest.param({"kind": "object", "caption": "a cup", "bbox": [16, 0, 8, 16]}, id="x-reversed"),
        pytest.param({"kind": "object", "caption": "a cup", "bbox": [0, 16, 16, 8]}, id="y-reversed"),
    ],
)
def test_a_region_outside_the_contract_is_refused_with_the_graph(region: dict) -> None:
    with pytest.raises(ValidationError):
        _node(region, prompt="a scene")


def test_the_layout_prompt_can_be_recorded_in_core_metadata() -> None:
    # Graph validation refuses an edge into a field `core_metadata` does not declare.
    graph = Graph(
        nodes={
            "layout_prompt": MossImageLayoutPromptInvocation(id="layout_prompt"),
            "core_metadata": CoreMetadataInvocation(id="core_metadata"),
        }
    )
    edge = Edge(
        source=EdgeConnection(node_id="layout_prompt", field="value"),
        destination=EdgeConnection(node_id="core_metadata", field="moss_image_layout_prompt"),
    )

    graph.add_edge(edge)

    assert graph.edges == [edge]
