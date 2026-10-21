"""The Fun-ControlNet-Union canvas graphs webv2 compiles are graphs this backend accepts.

`qwenImageControlCanvasGraphs.json` and `flux2ControlCanvasGraphs.json` are written by webv2's
`compileCanvasGraph.test.ts` (regenerate with `vitest -u`) from the real canvas compiler, one per family: a
text-to-image submission with one control layer, an inpainting submission where that layer also hands the control the
canvas composite and the inverted canvas mask, and an outpainting submission where it hands over the infilled image
and the combined mask through edges. This module checks more than names: pydantic validates every node against its
invocation (bounds such as `control_context_scale` in [0, 2] included), `Graph.validate_self` checks every edge type
into the denoise's `control` and `vae`, the control node's own `invoke` runs on the fields it would receive, and
`core_metadata` is serialised the way the gallery persists it. No model is loaded.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.flux2.flux2_control import Flux2ControlInvocation
from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.image import ImageLerpInvocation
from invokeai.app.invocations.metadata import CoreMetadataInvocation
from invokeai.app.invocations.qwen_image.qwen_image_control import QwenImageControlInvocation
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.app.invocations.sd.create_gradient_mask import CreateGradientMaskInvocation
from invokeai.app.services.shared.graph import Graph

SNAPSHOTS = (
    Path(__file__).parents[3]
    / "invokeai"
    / "frontend"
    / "webv2"
    / "src"
    / "features"
    / "generation"
    / "core"
    / "__snapshots__"
)


@dataclass(frozen=True)
class Family:
    """What one family's canvas graphs are made of, as `compileCanvasGraph.test.ts` writes them."""

    fixture: str
    control: type[BaseInvocation]
    denoise: type[BaseInvocation]
    loader: str
    base: str
    control_image: str
    scale: float
    metadata_prefix: str


FAMILIES = {
    "qwen_image": Family(
        fixture="qwenImageControlCanvasGraphs.json",
        control=QwenImageControlInvocation,
        denoise=QwenImageDenoiseInvocation,
        loader="qwen_image_model_loader",
        base="qwen-image",
        control_image="q-control.png",
        scale=0.85,
        metadata_prefix="qwen_image",
    ),
    "flux2": Family(
        fixture="flux2ControlCanvasGraphs.json",
        control=Flux2ControlInvocation,
        denoise=Flux2DenoiseInvocation,
        loader="flux2_dev_model_loader",
        base="flux2",
        control_image="f-control.png",
        scale=0.75,
        metadata_prefix="flux2",
    ),
}

GRAPHS: dict[str, dict[str, Any]] = {
    family: {
        name: graph
        for name, graph in json.loads((SNAPSHOTS / spec.fixture).read_text(encoding="utf-8")).items()
        if name != "_comment"
    }
    for family, spec in FAMILIES.items()
}
CASES = [(family, name) for family in sorted(GRAPHS) for name in sorted(GRAPHS[family])]


def _graph(family: str, name: str) -> Graph:
    graph = Graph.model_validate(GRAPHS[family][name])
    graph.validate_self()
    return graph


def _only(graph: Graph, cls: type[BaseInvocation]) -> Any:
    nodes = [node for node in graph.nodes.values() if isinstance(node, cls)]
    assert len(nodes) == 1
    return nodes[0]


def _inputs(graph: Graph, node_id: str) -> dict[str, Any]:
    """A node's edge-fed inputs, by destination field."""
    return {e.destination.field: e.source for e in graph.edges if e.destination.node_id == node_id}


def _control_field(graph: Graph, node: Any) -> Any:
    """The field the node would emit. `invoke` only repackages the node's own fields, so an image that an edge
    supplies at run time stands in as the image its source node would produce."""
    runtime = node.model_copy(
        update={
            field: ImageField(image_name=f"{node.id}.{field}")
            for field in _inputs(graph, node.id)
            if field in {"image", "inpaint_image", "inpaint_mask"}
        }
    )
    return runtime.invoke(None).control


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_fixtures_cover_every_canvas_mode_with_one_control_layer(family: str) -> None:
    spec = FAMILIES[family]
    assert sorted(GRAPHS[family]) == ["inpaint_control_layer", "outpaint_control_layer", "txt2img_control_layer"]
    for name, raw in GRAPHS[family].items():
        types = [node["type"] for node in raw["nodes"].values()]
        assert types.count(spec.control.get_type()) == 1, name
        assert types.count(spec.denoise.get_type()) == 1, name


@pytest.mark.parametrize(("family", "name"), CASES)
def test_every_field_the_frontend_writes_is_one_the_invocation_declares(family: str, name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field (`weight` for `control_context_scale`) would validate
    and run at its default. Values must also survive validation unchanged rather than being coerced."""
    graph = Graph.model_validate(GRAPHS[family][name])
    for node_id, raw in GRAPHS[family][name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it, feature fields included, and persists the extras.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"
        if isinstance(node, FAMILIES[family].control):
            written = {
                field: raw[field] for field in ("control_context_scale", "begin_step_percent", "end_step_percent")
            }
            assert {field: getattr(node, field) for field in written} == written


@pytest.mark.parametrize(("family", "name"), CASES)
def test_the_control_reaches_the_denoise_directly_beside_the_vae_it_needs(family: str, name: str) -> None:
    """The denoise's `control` is one field, not a collection, and it encodes the control's images with the VAE
    connected beside it."""
    spec = FAMILIES[family]
    graph = _graph(family, name)
    control = _only(graph, spec.control)
    inputs = _inputs(graph, _only(graph, spec.denoise).id)

    assert (inputs["control"].node_id, inputs["control"].field) == (control.id, "control")
    assert graph.nodes[inputs["vae"].node_id].get_type() == spec.loader
    assert inputs["vae"].field == "vae"
    assert control.control_model.base.value == spec.base
    assert control.control_model.type.value == "controlnet"
    assert control.control_context_scale == spec.scale
    assert (control.begin_step_percent, control.end_step_percent) == (0.0, 1.0)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_text_to_image_sends_the_control_image_alone(family: str) -> None:
    spec = FAMILIES[family]
    graph = _graph(family, "txt2img_control_layer")
    control = _only(graph, spec.control)

    assert control.image is not None and control.image.image_name == spec.control_image
    assert control.inpaint_image is None and control.inpaint_mask is None
    assert _inputs(graph, control.id) == {}
    assert not any(isinstance(node, ImageLerpInvocation) for node in graph.nodes.values())
    field = _control_field(graph, control)
    assert field.image_name == spec.control_image
    assert field.inpaint_image_name is None and field.inpaint_mask_name is None


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_inpainting_adds_the_composite_and_the_inverted_uploaded_mask(family: str) -> None:
    """Without Scale Before Processing the canvas uploads are read as fixed image fields, not through resize nodes.
    The canvas mask is white = keep; the control regenerates white, so an `img_lerp` 255 -> 0 feeds its mask."""
    spec = FAMILIES[family]
    graph = _graph(family, "inpaint_control_layer")
    control = _only(graph, spec.control)

    assert control.image is not None and control.image.image_name == spec.control_image
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
    gradient_mask = graph.nodes[_inputs(graph, _only(graph, spec.denoise).id)["denoise_mask"].node_id]
    assert isinstance(gradient_mask, CreateGradientMaskInvocation)
    assert gradient_mask.mask.image_name == "canvas-mask.png"
    field = _control_field(graph, control)
    assert (field.image_name, field.inpaint_image_name) == (spec.control_image, "canvas-composite.png")
    assert field.inpaint_mask_name == f"{control.id}.inpaint_mask"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_outpainting_adds_the_infilled_image_and_the_inverted_combined_mask(family: str) -> None:
    spec = FAMILIES[family]
    graph = _graph(family, "outpaint_control_layer")
    control = _only(graph, spec.control)

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
    assert field.image_name == spec.control_image
    assert field.inpaint_image_name is not None and field.inpaint_mask_name is not None


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_control_node_refuses_the_half_pair_the_frontend_never_sends(family: str) -> None:
    """The frontend sends the inpaint image and mask together or not at all; pin the node constraint it mirrors."""
    graph = Graph.model_validate(GRAPHS[family]["inpaint_control_layer"])
    half = _only(graph, FAMILIES[family].control).model_copy(update={"inpaint_mask": None})

    with pytest.raises(ValueError, match="go together"):
        half.invoke(None)


@pytest.mark.parametrize(("family", "name"), CASES)
def test_persisted_metadata_records_the_canvas_mode(family: str, name: str) -> None:
    graph = Graph.model_validate(GRAPHS[family][name])
    metadata = next(node for node in graph.nodes.values() if isinstance(node, CoreMetadataInvocation))

    persisted = metadata.invoke(None).metadata.root  # type: ignore[arg-type]

    mode = name.removesuffix("_control_layer")
    assert persisted["generation_mode"] == f"{FAMILIES[family].metadata_prefix}_{mode}"
