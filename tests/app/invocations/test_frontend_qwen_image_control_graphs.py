"""The Qwen Image ControlNet canvas graphs webv2 compiles are graphs this backend accepts.

`qwenImageControlCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with
`vitest -u`) from the real canvas compiler: a text-to-image submission with one control layer, an inpainting
submission where that layer also hands the control the canvas composite and the inverted canvas mask, and an
outpainting submission where it hands over the infilled image and the combined mask through edges. This module
checks more than names: pydantic validates every node against its invocation (bounds such as
`qwen_image_control.control_context_scale` in [0, 2] included), `Graph.validate_self` checks every edge type into
`qwen_image_denoise.control` and `.vae`, the control node's own `invoke` runs on the fields it would receive, and
`core_metadata` is serialised the way the gallery persists it. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.image import ImageLerpInvocation
from invokeai.app.invocations.metadata import CoreMetadataInvocation
from invokeai.app.invocations.qwen_image.qwen_image_control import QwenImageControlField, QwenImageControlInvocation
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.app.invocations.sd.create_gradient_mask import CreateGradientMaskInvocation
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
    / "qwenImageControlCanvasGraphs.json"
)

GRAPHS: dict[str, Any] = {
    name: graph for name, graph in json.loads(FIXTURE_PATH.read_text(encoding="utf-8")).items() if name != "_comment"
}


@pytest.fixture(params=sorted(GRAPHS), ids=str)
def graph(request: pytest.FixtureRequest) -> Graph:
    graph = Graph.model_validate(GRAPHS[request.param])
    graph.validate_self()
    return graph


def _control_node(graph: Graph) -> QwenImageControlInvocation:
    nodes = [node for node in graph.nodes.values() if isinstance(node, QwenImageControlInvocation)]
    assert len(nodes) == 1
    return nodes[0]


def _denoise_node(graph: Graph) -> QwenImageDenoiseInvocation:
    nodes = [node for node in graph.nodes.values() if isinstance(node, QwenImageDenoiseInvocation)]
    assert len(nodes) == 1
    return nodes[0]


def _inputs(graph: Graph, node_id: str) -> dict[str, Any]:
    """A node's edge-fed inputs, by destination field."""
    return {e.destination.field: e.source for e in graph.edges if e.destination.node_id == node_id}


def _control_field(graph: Graph, node: QwenImageControlInvocation) -> QwenImageControlField:
    """The field the node would emit. `invoke` only repackages the node's own fields, so an image that an edge
    supplies at run time stands in as the image its source node would produce."""
    runtime = node.model_copy(
        update={
            field: ImageField(image_name=f"{node.id}.{field}")
            for field in _inputs(graph, node.id)
            if field in {"image", "inpaint_image", "inpaint_mask"}
        }
    )
    return runtime.invoke(None).control  # type: ignore[arg-type]


def test_the_fixtures_cover_every_canvas_mode_with_one_control_layer() -> None:
    assert sorted(GRAPHS) == ["inpaint_control_layer", "outpaint_control_layer", "txt2img_control_layer"]
    for name, raw in GRAPHS.items():
        types = [node["type"] for node in raw["nodes"].values()]
        assert types.count("qwen_image_control") == 1, name
        assert types.count("qwen_image_denoise") == 1, name


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`weight` for `control_context_scale`) would validate
    and run at its default. Values must also survive validation unchanged rather than being coerced."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"
        if isinstance(node, QwenImageControlInvocation):
            written = {
                field: raw[field] for field in ("control_context_scale", "begin_step_percent", "end_step_percent")
            }
            assert {field: getattr(node, field) for field in written} == written


def test_the_control_reaches_the_denoise_directly_beside_the_vae_it_needs(graph: Graph) -> None:
    """`qwen_image_denoise.control` is one field, not a collection, and the denoise refuses a control unless the
    VAE that encodes its images is connected too."""
    control = _control_node(graph)
    denoise = _denoise_node(graph)
    inputs = _inputs(graph, denoise.id)

    assert (inputs["control"].node_id, inputs["control"].field) == (control.id, "control")
    assert graph.nodes[inputs["vae"].node_id].get_type() == "qwen_image_model_loader"
    assert inputs["vae"].field == "vae"
    assert not any(node.get_type() == "collect" for node in graph.nodes.values())
    assert control.control_model.base.value == "qwen-image"
    assert control.control_model.type.value == "controlnet"
    assert control.control_context_scale == 0.85
    assert (control.begin_step_percent, control.end_step_percent) == (0.0, 1.0)


def test_text_to_image_sends_the_control_image_alone() -> None:
    graph = Graph.model_validate(GRAPHS["txt2img_control_layer"])
    control = _control_node(graph)

    assert control.image is not None and control.image.image_name == "q-control.png"
    assert control.inpaint_image is None and control.inpaint_mask is None
    assert _inputs(graph, control.id) == {}
    assert not any(isinstance(node, ImageLerpInvocation) for node in graph.nodes.values())
    field = _control_field(graph, control)
    assert field.image_name == "q-control.png"
    assert field.inpaint_image_name is None and field.inpaint_mask_name is None


def test_inpainting_adds_the_composite_and_the_inverted_uploaded_mask() -> None:
    """Without Scale Before Processing the canvas uploads are read as fixed image fields, not through resize nodes.
    The canvas mask is white = keep; the control regenerates white, so an `img_lerp` 255 -> 0 feeds its mask."""
    graph = Graph.model_validate(GRAPHS["inpaint_control_layer"])
    graph.validate_self()
    control = _control_node(graph)

    assert control.image is not None and control.image.image_name == "q-control.png"
    assert control.inpaint_image is not None and control.inpaint_image.image_name == "canvas-composite.png"
    inputs = _inputs(graph, control.id)
    assert set(inputs) == {"inpaint_mask"}
    invert = graph.nodes[inputs["inpaint_mask"].node_id]
    assert isinstance(invert, ImageLerpInvocation)
    assert (invert.min, invert.max) == (255, 0)
    assert invert.image.image_name == "canvas-mask.png"
    assert inputs["inpaint_mask"].field == "image"
    assert _inputs(graph, invert.id) == {}
    # The denoise mask is the uploaded canvas mask as is; only the control gets the inverted copy.
    gradient_mask = graph.nodes[_inputs(graph, _denoise_node(graph).id)["denoise_mask"].node_id]
    assert isinstance(gradient_mask, CreateGradientMaskInvocation)
    assert gradient_mask.mask.image_name == "canvas-mask.png"
    field = _control_field(graph, control)
    assert (field.image_name, field.inpaint_image_name) == ("q-control.png", "canvas-composite.png")
    assert field.inpaint_mask_name == f"{control.id}.inpaint_mask"


def test_outpainting_adds_the_infilled_image_and_the_inverted_combined_mask() -> None:
    graph = Graph.model_validate(GRAPHS["outpaint_control_layer"])
    graph.validate_self()
    control = _control_node(graph)

    inputs = _inputs(graph, control.id)
    assert set(inputs) == {"inpaint_image", "inpaint_mask"}
    assert graph.nodes[inputs["inpaint_image"].node_id].get_type() == "infill_lama"
    assert inputs["inpaint_image"].field == "image"
    invert = graph.nodes[inputs["inpaint_mask"].node_id]
    assert isinstance(invert, ImageLerpInvocation)
    assert (invert.min, invert.max) == (255, 0)
    assert inputs["inpaint_mask"].field == "image"
    assert graph.nodes[_inputs(graph, invert.id)["image"].node_id].get_type() == "mask_combine"
    # The control reads the same infilled image the encode the denoise starts from reads.
    assert _inputs(graph, "canvas_i2l")["image"].node_id == inputs["inpaint_image"].node_id
    field = _control_field(graph, control)
    assert field.image_name == "q-control.png"
    assert field.inpaint_image_name is not None and field.inpaint_mask_name is not None


def test_the_control_node_refuses_the_half_pair_the_frontend_never_sends() -> None:
    """The frontend sends the inpaint image and mask together or not at all; pin the node constraint it mirrors."""
    graph = Graph.model_validate(GRAPHS["inpaint_control_layer"])
    control = _control_node(graph)
    half = control.model_copy(update={"inpaint_mask": None})

    with pytest.raises(ValueError, match="go together"):
        half.invoke(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("txt2img_control_layer", "qwen_image_txt2img"),
        ("inpaint_control_layer", "qwen_image_inpaint"),
        ("outpaint_control_layer", "qwen_image_outpaint"),
    ],
)
def test_persisted_metadata_records_the_canvas_mode(name: str, mode: str) -> None:
    graph = Graph.model_validate(GRAPHS[name])
    metadata = next(node for node in graph.nodes.values() if isinstance(node, CoreMetadataInvocation))

    persisted = metadata.invoke(None).metadata.root  # type: ignore[arg-type]

    assert persisted["generation_mode"] == mode
