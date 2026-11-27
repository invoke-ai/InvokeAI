"""First-party graph and execution contracts for image manipulation nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from invokeai.app.invocations import load_all_modules
from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.backend.image_util.clarity import clarity
from invokeai.backend.image_util.contrast_dehaze import contrast_dehaze
from invokeai.backend.image_util.texture import texture

NODE_CONTRACTS = (
    ("clarity", "Clarity", clarity, 40, (4, 1000), True),
    ("texture", "Texture", texture, 8, (1, 100), True),
    ("contrast_dehaze", "Contrast Dehaze (approximation)", contrast_dehaze, None, None, True),
)


def _registered_nodes() -> dict[str, type[Any]]:
    load_all_modules()
    return {node.get_type(): node for node in InvocationRegistry.get_invocation_classes()}


@pytest.fixture(scope="module")
def nodes() -> dict[str, type[Any]]:
    return _registered_nodes()


@pytest.mark.parametrize(("node_id", "title", "effect", "size_default", "size_bounds", "metadata"), NODE_CONTRACTS)
def test_node_is_registered_with_stable_identity_and_defaults(
    nodes: dict[str, type[Any]],
    node_id: str,
    title: str,
    effect: Callable[..., Image.Image],
    size_default: int | None,
    size_bounds: tuple[int, int] | None,
    metadata: bool,
) -> None:
    del effect
    node_class = nodes[node_id]
    assert node_id in nodes
    assert node_class.get_type() == node_id
    assert node_class.UIConfig.title == title
    assert node_class.UIConfig.tags
    assert node_class.UIConfig.category == "image"
    assert node_class.UIConfig.version == "1.0.1"
    assert node_class.UIConfig.node_pack == "invokeai"
    assert "image" in node_class.model_fields
    assert node_class.model_fields["amount"].default == 0
    if size_default is not None:
        assert node_class.model_fields["size"].default == size_default
        assert node_class.model_fields["size"].metadata
        lower, upper = size_bounds
        for invalid_size in (lower - 1, upper + 1, float("nan")):
            with pytest.raises(ValidationError):
                node_class.model_validate({"image": {"image_name": "input"}, "size": invalid_size})
    assert ("board" in node_class.model_fields) is metadata
    assert ("metadata" in node_class.model_fields) is metadata
    if metadata:
        node_class.model_validate(
            {
                "image": {"image_name": "input"},
                "board": {"board_id": "board-id"},
                "metadata": {},
            }
        )


@pytest.mark.parametrize(("node_id", "title", "effect", "size_default", "size_bounds", "metadata"), NODE_CONTRACTS)
def test_amount_bounds_and_non_finite_values_are_rejected(
    nodes: dict[str, type[Any]],
    node_id: str,
    title: str,
    effect: Callable[..., Image.Image],
    size_default: int | None,
    size_bounds: tuple[int, int] | None,
    metadata: bool,
) -> None:
    del title, effect, size_default, size_bounds, metadata
    node_class = nodes[node_id]
    for amount in (-200, -100, 0, 100, 200):
        node_class.model_validate({"image": {"image_name": "input"}, "amount": amount})
    for amount in (-200.01, 200.01, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValidationError):
            node_class.model_validate({"image": {"image_name": "input"}, "amount": amount})


class _ImageStore:
    """Small real PIL image store implementing the image service methods wrappers need."""

    def __init__(self, source: Image.Image) -> None:
        self.images = {"source": source.copy()}
        self.saved: list[Image.Image] = []

    def get_pil(self, image_name: str) -> Image.Image:
        return self.images[image_name].copy()

    def save(self, image: Image.Image, **_: Any) -> Any:
        name = f"saved-{len(self.saved)}"
        self.saved.append(image.copy())
        self.images[name] = image.copy()
        return type("ImageRecord", (), {"image_name": name, "width": image.width, "height": image.height})()


class _Context:
    def __init__(self, images: _ImageStore) -> None:
        self.images = images


def _source_image() -> Image.Image:
    y, x = np.mgrid[:48, :64]
    rgb = np.stack(((x * 7 + y * 3) % 256, (x * 2 + y * 9) % 256, (x * 11 + y * 5) % 256), axis=-1)
    alpha = ((x * 3 + y * 5) % 256)[..., None]
    rgba = np.concatenate((rgb, alpha), axis=-1)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


@pytest.mark.parametrize(("node_id", "title", "effect", "size_default", "size_bounds", "metadata"), NODE_CONTRACTS)
def test_wrapper_reads_applies_and_saves_once(
    nodes: dict[str, type[Any]],
    node_id: str,
    title: str,
    effect: Callable[..., Image.Image],
    size_default: int | None,
    size_bounds: tuple[int, int] | None,
    metadata: bool,
) -> None:
    del title, size_bounds, metadata
    source = _source_image()
    store = _ImageStore(source)
    values: dict[str, Any] = {"image": ImageField(image_name="source"), "amount": 35}
    if size_default is not None:
        values["size"] = size_default
    invocation = nodes[node_id](**values)
    output = invocation.invoke(_Context(store))
    expected = effect(source, 35, size_default) if size_default is not None else effect(source, 35)
    assert isinstance(output, ImageOutput)
    assert (output.width, output.height) == source.size
    assert len(store.saved) == 1
    assert output.image.image_name == "saved-0"
    assert np.array_equal(np.asarray(store.saved[0]), np.asarray(expected))


@pytest.mark.parametrize(("node_id", "title", "effect", "size_default", "size_bounds", "metadata"), NODE_CONTRACTS)
def test_zero_amount_saves_an_unchanged_image(
    nodes: dict[str, type[Any]],
    node_id: str,
    title: str,
    effect: Callable[..., Image.Image],
    size_default: int | None,
    size_bounds: tuple[int, int] | None,
    metadata: bool,
) -> None:
    del title, effect, size_bounds, metadata
    source = _source_image()
    store = _ImageStore(source)
    values: dict[str, Any] = {"image": ImageField(image_name="source"), "amount": 0}
    if size_default is not None:
        values["size"] = size_default
    output = nodes[node_id](**values).invoke(_Context(store))
    assert isinstance(output, ImageOutput)
    assert len(store.saved) == 1
    assert np.array_equal(np.asarray(store.saved[0]), np.asarray(source))


def test_image_manipulation_nodes_are_registered(nodes: dict[str, type[Any]]) -> None:
    assert all(node_id in nodes for node_id, *_ in NODE_CONTRACTS)


def test_image_manipulation_graph_executes_with_connected_images(nodes: dict[str, type[Any]]) -> None:
    source = _source_image()
    store = _ImageStore(source)
    graph = Graph.model_validate(
        {
            "nodes": {
                "clarity": {
                    "id": "clarity",
                    "type": "clarity",
                    "image": {"image_name": "source"},
                    "amount": -40,
                    "size": 12,
                },
                "texture": {
                    "id": "texture",
                    "type": "texture",
                    "image": {"image_name": "unused"},
                    "amount": 30,
                    "size": 6,
                },
                "dehaze": {
                    "id": "dehaze",
                    "type": "contrast_dehaze",
                    "image": {"image_name": "unused"},
                    "amount": -25,
                },
            },
            "edges": [
                {
                    "source": {"node_id": "clarity", "field": "image"},
                    "destination": {"node_id": "texture", "field": "image"},
                },
                {
                    "source": {"node_id": "texture", "field": "image"},
                    "destination": {"node_id": "dehaze", "field": "image"},
                },
            ],
        }
    )
    graph.validate_self()
    state = GraphExecutionState(graph=graph)
    context = _Context(store)
    completed: list[tuple[str, str]] = []

    while (invocation := state.next()) is not None:
        output = invocation.invoke(context)
        assert isinstance(output, ImageOutput)
        completed.append((state.prepared_source_mapping[invocation.id], output.image.image_name))
        state.complete(invocation.id, output)

    assert completed == [("clarity", "saved-0"), ("texture", "saved-1"), ("dehaze", "saved-2")]
    assert state.is_complete()
    assert len(store.saved) == 3
    expected = contrast_dehaze(texture(clarity(source, -40, 12), 30, 6), -25)
    final_image = store.images["saved-2"]
    assert (final_image.width, final_image.height) == source.size
    assert final_image.mode == "RGBA"
    assert np.array_equal(np.asarray(final_image), np.asarray(expected))
    assert np.array_equal(np.asarray(final_image.getchannel("A")), np.asarray(source.getchannel("A")))


def test_graph_rejects_non_finite_image_manipulation_input(nodes: dict[str, type[Any]]) -> None:
    del nodes
    with pytest.raises(ValidationError):
        Graph.model_validate(
            {
                "nodes": {
                    "clarity": {
                        "id": "clarity",
                        "type": "clarity",
                        "image": {"image_name": "source"},
                        "amount": float("nan"),
                    }
                },
                "edges": [],
            }
        )
