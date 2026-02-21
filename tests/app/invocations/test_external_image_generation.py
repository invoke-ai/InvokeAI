from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.invocations.external_image_generation import ExternalImageGenerationInvocation
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationResult,
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


def _build_model() -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key="external_test",
        name="External Test",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(
            modes=["txt2img"],
            supports_reference_images=True,
            supports_negative_prompt=True,
            supports_seed=True,
        ),
    )


def _build_context(model_config: ExternalApiModelConfig, generated_image: Image.Image) -> MagicMock:
    context = MagicMock()
    context.models.get_config.return_value = model_config
    context.images.get_pil.return_value = generated_image
    context.images.save.return_value = SimpleNamespace(image_name="result.png")
    context._services.external_generation.generate.return_value = ExternalGenerationResult(
        images=[ExternalGeneratedImage(image=generated_image, seed=42)],
        provider_request_id="req-123",
        provider_metadata={"model": model_config.provider_model_id},
    )
    return context


def test_external_invocation_builds_request_and_outputs() -> None:
    model_config = _build_model()
    model_field = ModelIdentifierField.from_config(model_config)
    generated_image = Image.new("RGB", (16, 16), color="black")
    context = _build_context(model_config, generated_image)

    invocation = ExternalImageGenerationInvocation(
        id="external_node",
        model=model_field,
        mode="txt2img",
        prompt="A prompt",
        negative_prompt="bad",
        seed=123,
        num_images=1,
        width=512,
        height=512,
        steps=10,
        guidance=4.5,
        reference_images=[ImageField(image_name="ref.png")],
        reference_image_weights=[0.6],
    )

    output = invocation.invoke(context)

    request = context._services.external_generation.generate.call_args[0][0]
    assert request.prompt == "A prompt"
    assert request.negative_prompt == "bad"
    assert request.seed == 123
    assert len(request.reference_images) == 1
    assert request.reference_images[0].weight == 0.6
    assert output.collection[0].image_name == "result.png"


def test_external_invocation_rejects_mismatched_reference_weights() -> None:
    model_config = _build_model()
    model_field = ModelIdentifierField.from_config(model_config)
    generated_image = Image.new("RGB", (16, 16), color="black")
    context = _build_context(model_config, generated_image)

    invocation = ExternalImageGenerationInvocation(
        id="external_node",
        model=model_field,
        mode="txt2img",
        prompt="A prompt",
        reference_images=[ImageField(image_name="ref.png")],
        reference_image_weights=[0.1, 0.2],
    )

    with pytest.raises(ValueError, match="reference_image_weights"):
        invocation.invoke(context)


def test_external_graph_execution_state_runs_node() -> None:
    model_config = _build_model()
    model_field = ModelIdentifierField.from_config(model_config)
    generated_image = Image.new("RGB", (16, 16), color="black")
    context = _build_context(model_config, generated_image)

    invocation = ExternalImageGenerationInvocation(
        id="external_node",
        model=model_field,
        mode="txt2img",
        prompt="A prompt",
    )

    graph = Graph()
    graph.add_node(invocation)

    session = GraphExecutionState(graph=graph)
    node = session.next()
    assert node is not None
    output = node.invoke(context)
    session.complete(node.id, output)

    assert session.results[node.id] == output
