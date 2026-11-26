"""The Krea-2 Control LoRA canvas graphs webv2 compiles are graphs this backend accepts.

`krea2ControlLoraCanvasGraphs.json` is written by webv2's `compileCanvasGraph.test.ts` (regenerate with `vitest -u`)
from the real canvas compiler: a text-to-image and an inpainting submission, each with one control layer running a
Krea-2 Control LoRA. Pydantic validates every node against its invocation, `Graph.validate_self` checks the edge types
from the control image encode through `krea2_control_lora` into `krea2_denoise.control_lora`, and the encode is checked
to produce latents of the size the denoise accepts. No model is loaded.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.app.invocations.krea2.krea2_control_lora import Krea2ControlLoRAInvocation
from invokeai.app.invocations.krea2.krea2_denoise import Krea2DenoiseInvocation
from invokeai.app.invocations.vae.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
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
    / "krea2ControlLoraCanvasGraphs.json"
)

GRAPHS: dict[str, Any] = {
    name: graph for name, graph in json.loads(FIXTURE_PATH.read_text(encoding="utf-8")).items() if name != "_comment"
}


@pytest.fixture(params=sorted(GRAPHS), ids=str)
def name(request: pytest.FixtureRequest) -> str:
    return request.param


def _graph(name: str) -> Graph:
    graph = Graph.model_validate(GRAPHS[name])
    graph.validate_self()
    return graph


def _only(graph: Graph, cls: type) -> Any:
    nodes = [node for node in graph.nodes.values() if isinstance(node, cls)]
    assert len(nodes) == 1
    return nodes[0]


def _inputs(graph: Graph, node_id: str) -> dict[str, Any]:
    """A node's edge-fed inputs, by destination field."""
    return {e.destination.field: e.source for e in graph.edges if e.destination.node_id == node_id}


def test_every_field_the_frontend_writes_is_one_the_invocation_declares(name: str) -> None:
    """Pydantic ignores an unknown key, so a misnamed field would validate and run at its default."""
    graph = Graph.model_validate(GRAPHS[name])
    for node_id, raw in GRAPHS[name]["nodes"].items():
        node = graph.nodes[node_id]
        if type(node).model_config.get("extra") == "allow":
            # `core_metadata` records whatever the graph gives it.
            continue
        unknown = set(raw) - set(type(node).model_fields)
        assert unknown == set(), f"{raw['type']} has no field(s) {sorted(unknown)}"


def test_the_control_lora_reaches_the_denoise_with_latents_encoded_at_its_size(name: str) -> None:
    graph = _graph(name)
    denoise = _only(graph, Krea2DenoiseInvocation)
    control = _only(graph, Krea2ControlLoRAInvocation)

    assert control.lora.base.value == "krea-2"
    assert control.lora.type.value == "control_lora"
    assert control.weight == 0.8
    denoise_inputs = _inputs(graph, denoise.id)
    assert (denoise_inputs["control_lora"].node_id, denoise_inputs["control_lora"].field) == (
        control.id,
        "control_lora",
    )

    latents_source = _inputs(graph, control.id)["control_latents"]
    encode = graph.nodes[latents_source.node_id]
    assert isinstance(encode, QwenImageImageToLatentsInvocation)
    assert latents_source.field == "latents"
    assert encode.image.image_name == "k-control.png"
    # `reference` sizes the image itself; the denoise refuses control latents of any size but its own.
    assert not encode.reference
    assert (encode.width, encode.height) == (denoise.width, denoise.height)


def test_the_control_image_is_encoded_with_the_vae_the_graph_decodes_with(name: str) -> None:
    graph = _graph(name)
    control = _only(graph, Krea2ControlLoRAInvocation)
    encode_id = _inputs(graph, control.id)["control_latents"].node_id
    vae = _inputs(graph, encode_id)["vae"]
    decode = next(node for node in graph.nodes.values() if node.get_type() == "qwen_image_l2i")

    assert graph.nodes[vae.node_id].get_type() == "krea2_model_loader"
    assert (vae.node_id, vae.field) == (_inputs(graph, decode.id)["vae"].node_id, "vae")
