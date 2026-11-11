import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from invokeai.app.invocations.external_image_generation import MiniMaxImageGenerationInvocation
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64
from invokeai.app.services.external_generation.providers.minimax import MiniMaxProvider
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.starter_models import STARTER_MODELS


@pytest.fixture
def model(request: pytest.FixtureRequest) -> ExternalApiModelConfig:
    model_id = getattr(request, "param", "MiniMax-H3-Image")
    source = f"external://minimax/{model_id}"
    starter = next(model for model in STARTER_MODELS if model.source == source)
    assert starter.capabilities is not None
    return ExternalApiModelConfig(
        key="minimax-test",
        name=starter.name,
        source=source,
        provider_id="minimax",
        provider_model_id=model_id,
        capabilities=starter.capabilities,
    )


@pytest.fixture
def generation_request(model: ExternalApiModelConfig) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode="txt2img",
        prompt="A spring landscape",
        seed=123,
        num_images=2,
        width=1024,
        height=1024,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )


def _service(base_url: str | None = None) -> ExternalGenerationService:
    config = InvokeAIAppConfig(external_minimax_api_key="test-key", external_minimax_base_url=base_url)
    logger = logging.getLogger(__name__)
    return ExternalGenerationService({"minimax": MiniMaxProvider(config, logger)}, logger)


def _response(payload: object, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode()
    return response


def _jpeg() -> str:
    return encode_image_base64(Image.new("RGB", (1024, 1024), "red"), format="JPEG")


@pytest.mark.parametrize("model", ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"], indirect=True)
def test_generation_uses_inline_jpeg_and_returns_successes_from_a_partial_response(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest
) -> None:
    post = Mock(
        return_value=_response(
            {
                "request_id": "minimax-request",
                "data": [
                    {"error": {"code": "1027", "message": "test-key and private prompt"}},
                    {"b64_json": _jpeg()},
                ],
            }
        )
    )
    monkeypatch.setattr(requests, "post", post)

    result = _service().generate(generation_request)

    post.assert_called_once()
    assert post.call_args.args == ("https://api.minimax.io/v1/images/generations",)
    kwargs = post.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    connect_timeout, read_timeout = kwargs["timeout"]
    assert 0 < connect_timeout < 600 < read_timeout
    assert kwargs["allow_redirects"] is False
    assert json.loads(kwargs["data"]) == {
        "model": generation_request.model.provider_model_id,
        "prompt": "A spring landscape",
        "n": 2,
        "size": "1024x1024",
        "output_format": "jpeg",
        "output_compression": 100,
        "response_format": "b64_json",
    }
    assert result.provider_request_id == "minimax-request"
    assert result.provider_metadata == {
        "model": generation_request.model.provider_model_id,
        "partial_failures": [
            {
                "image_index": 0,
                "code": "1027",
                "message": (
                    "The generated image failed MiniMax's content safety check. Try adjusting the prompt or reference images."
                ),
            }
        ],
    }
    assert result.seed_used is None
    assert len(result.images) == 1
    assert result.images[0].image.size == (1024, 1024)
    assert result.images[0].seed is None


@pytest.mark.parametrize(
    ("mode", "init_name", "expected_colors"),
    [
        ("txt2img", None, [(0, 0, 255), (0, 128, 0)]),
        ("img2img", "init.png", [(255, 0, 0), (0, 0, 255), (0, 128, 0)]),
    ],
)
@pytest.mark.parametrize("model", ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"], indirect=True)
def test_graph_invocation_sends_init_then_references_to_edits(
    monkeypatch: pytest.MonkeyPatch,
    model: ExternalApiModelConfig,
    mode: str,
    init_name: str | None,
    expected_colors: list[tuple[int, int, int]],
) -> None:
    post = Mock(return_value=_response({"request_id": "edit-request", "data": [{"b64_json": _jpeg()}]}))
    monkeypatch.setattr(requests, "post", post)
    graph = Graph.model_validate(
        {
            "nodes": {
                "edit": {
                    "id": "edit",
                    "type": "minimax_image_generation",
                    "model": ModelIdentifierField.from_config(model),
                    "mode": mode,
                    "prompt": "Replace the foreground.",
                    "init_image": ImageField(image_name=init_name) if init_name else None,
                    "reference_images": [ImageField(image_name="blue.png"), ImageField(image_name="green.png")],
                },
            },
        }
    )
    session = GraphExecutionState(graph=graph)
    node = session.next()
    assert isinstance(node, MiniMaxImageGenerationInvocation)
    images = {
        "init.png": Image.new("RGB", (128, 128), "red"),
        "blue.png": Image.new("RGB", (128, 128), "blue"),
        "green.png": Image.new("RGB", (128, 128), "green"),
    }
    context = Mock()
    context.models.get_config.return_value = model
    context.images.get_pil.side_effect = lambda name, **_: images[name]
    context.images.save.return_value = SimpleNamespace(image_name="result.png")
    context._services.external_generation = _service("https://minimax.test/")

    output = node.invoke(context)
    session.complete(node.id, output)

    assert session.results[node.id] == output
    assert output.collection[0].image_name == "result.png"
    assert post.call_args.args == ("https://minimax.test/v1/images/edits",)
    payload = json.loads(post.call_args.kwargs["data"])
    assert payload["model"] == model.provider_model_id
    assert payload["output_format"] == "jpeg"
    assert payload["output_compression"] == 100
    assert payload["response_format"] == "b64_json"
    data_urls = [item["image_url"] for item in payload["images"]]
    assert all(url.startswith("data:image/png;base64,") for url in data_urls)
    assert [decode_image_base64(url.split(",", 1)[1]).getpixel((0, 0)) for url in data_urls] == expected_colors
    metadata = context.images.save.call_args.kwargs["metadata"].root
    assert metadata["external_provider"] == "minimax"
    assert metadata["external_model_id"] == model.provider_model_id
    assert metadata["external_request_id"] == "edit-request"


@pytest.mark.parametrize(("init_name", "reference_count"), [(None, 17), ("init.png", 16)])
def test_invocation_rejects_too_many_inputs_before_loading_images(
    model: ExternalApiModelConfig, init_name: str | None, reference_count: int
) -> None:
    node = MiniMaxImageGenerationInvocation(
        id="edit",
        model=ModelIdentifierField.from_config(model),
        mode="img2img" if init_name else "txt2img",
        prompt="Replace the foreground.",
        init_image=ImageField(image_name=init_name) if init_name else None,
        reference_images=[ImageField(image_name=f"ref-{index}.png") for index in range(reference_count)],
    )
    context = Mock()
    context.models.get_config.return_value = model
    context.images.get_pil.side_effect = AssertionError("Invalid reference count must not load images")

    with pytest.raises(ExternalProviderCapabilityError, match="at most 16 inputs"):
        node.invoke(context)

    context.images.get_pil.assert_not_called()


def test_edits_keep_the_credentials_selected_before_encoding(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest
) -> None:
    config = InvokeAIAppConfig(external_minimax_api_key="old-key", external_minimax_base_url="https://old.test/")
    provider = MiniMaxProvider(config, logging.getLogger(__name__))
    encode_inputs = provider._encode_inputs

    def update_credentials_during_encoding(inputs: list[Image.Image]) -> list[dict[str, str]]:
        config.update_config({"external_minimax_api_key": "new-key", "external_minimax_base_url": "https://new.test/"})
        return encode_inputs(inputs)

    monkeypatch.setattr(provider, "_encode_inputs", update_credentials_during_encoding)
    post = Mock(return_value=_response({"data": [{"b64_json": _jpeg()}]}))
    monkeypatch.setattr(requests, "post", post)
    request = replace(generation_request, reference_images=[ExternalReferenceImage(Image.new("RGB", (128, 128)))])

    provider.generate(request)
    provider.generate(request)

    first, second = post.call_args_list
    assert first.args == ("https://old.test/v1/images/edits",)
    assert first.kwargs["headers"]["Authorization"] == "Bearer old-key"
    assert second.args == ("https://new.test/v1/images/edits",)
    assert second.kwargs["headers"]["Authorization"] == "Bearer new-key"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"width": 512, "height": 512}, "655,360"),
        ({"width": 4096, "height": 1024}, "aspect ratio"),
        ({"width": 4097}, "4096"),
        ({"width": 4096, "height": 4096, "num_images": 5}, "80,000,000"),
        ({"num_images": 11}, "at most 10"),
        ({"prompt": " "}, "prompt"),
        ({"mask_image": Image.new("L", (128, 128))}, "without masks"),
        ({"reference_images": [ExternalReferenceImage(Image.new("RGB", (127, 128)))]}, "edit inputs"),
        ({"reference_images": [ExternalReferenceImage(Image.new("RGB", (640, 128)))]}, "aspect ratio"),
        (
            {
                "mode": "img2img",
                "init_image": Image.new("RGB", (128, 128)),
                "reference_images": [ExternalReferenceImage(Image.new("RGB", (128, 128)))] * 16,
            },
            "at most 16 inputs",
        ),
    ],
)
def test_rejects_invalid_requests_before_sending(
    monkeypatch: pytest.MonkeyPatch,
    generation_request: ExternalGenerationRequest,
    updates: dict[str, object],
    message: str,
) -> None:
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalProviderCapabilityError, match=message):
        _service().generate(replace(generation_request, **updates))
    post.assert_not_called()


def test_custom_edit_model_still_requires_an_input(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest
) -> None:
    custom_model = generation_request.model.model_copy(
        update={
            "provider_model_id": "custom-minimax-image",
            "source": "external://minimax/custom-minimax-image",
            "capabilities": generation_request.model.capabilities.model_copy(
                update={"input_image_required_for": ["inpaint"]}
            ),
        }
    )
    post = Mock()
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(ExternalProviderCapabilityError, match="MiniMax image edits require an input image"):
        _service().generate(replace(generation_request, model=custom_model, mode="img2img"))

    post.assert_not_called()


def test_edit_body_limit_includes_json_overhead(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest
) -> None:
    image = Image.new("RGB", (128, 128), "blue")
    image_url_size = len("data:image/png;base64,") + len(encode_image_base64(image))
    monkeypatch.setattr(
        "invokeai.app.services.external_generation.providers.minimax._MAX_EDIT_BODY_BYTES", image_url_size
    )
    post = Mock()
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(ExternalProviderCapabilityError, match="64 MiB request body"):
        _service().generate(replace(generation_request, reference_images=[ExternalReferenceImage(image)]))
    post.assert_not_called()


@pytest.mark.parametrize("status_code", [307, 401, 429, 500])
def test_http_failures_are_sanitized_and_never_retried(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest, status_code: int
) -> None:
    post = Mock(return_value=_response({"error": {"message": "test-key and private prompt"}}, status_code))
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(ExternalProviderRequestError) as caught:
        _service().generate(generation_request)

    assert str(caught.value) == f"MiniMax image request failed (HTTP {status_code})"
    post.assert_called_once()


@pytest.mark.parametrize(
    ("status_code", "error_type", "code", "reason"),
    [
        (402, "insufficient_balance_error", "1008", "Your MiniMax account has insufficient balance."),
        (402, "insufficient_balance_error", "2068", "Your MiniMax plan does not include this model."),
        (401, "authorized_error", "9999", "Check your MiniMax API key."),
        (
            422,
            "unprocessable_entity_error",
            "1026",
            "The input failed MiniMax's content safety check. Try adjusting the prompt or reference images.",
        ),
        (
            422,
            "unprocessable_entity_error",
            "1027",
            "The generated image failed MiniMax's content safety check. Try adjusting the prompt or reference images.",
        ),
        (429, "rate_limit_error", "1002", "MiniMax's image generation rate limit was reached. Try again later."),
    ],
)
def test_http_failure_reports_a_sanitized_reason_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
    generation_request: ExternalGenerationRequest,
    status_code: int,
    error_type: str,
    code: str,
    reason: str,
) -> None:
    post = Mock(
        return_value=_response(
            {"error": {"type": error_type, "message": f"test-key and private prompt ({code})"}}, status_code
        )
    )
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(ExternalProviderRequestError) as caught:
        _service().generate(generation_request)

    assert str(caught.value) == f"MiniMax image request failed (HTTP {status_code}): {reason}"
    post.assert_called_once()


def test_network_errors_do_not_expose_the_request(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest
) -> None:
    post = Mock(side_effect=requests.Timeout("test-key and private prompt"))
    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(ExternalProviderRequestError, match="^MiniMax image request timed out$"):
        _service().generate(generation_request)

    post.assert_called_once()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": [{"url": "https://example.test/private"}]}, "no inline images"),
        ({"data": [{"b64_json": "not an image"}]}, "invalid inline image"),
        (
            {"data": [{"error": {"code": 1027, "message": "test-key and private prompt"}}]},
            "MiniMax returned no images. The generated image failed",
        ),
        ({"data": ["bad"] * 3}, "invalid image data"),
    ],
)
def test_rejects_responses_without_usable_inline_images(
    monkeypatch: pytest.MonkeyPatch, generation_request: ExternalGenerationRequest, payload: object, message: str
) -> None:
    monkeypatch.setattr(requests, "post", Mock(return_value=_response(payload)))
    get = Mock()
    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ExternalProviderRequestError, match=message):
        _service().generate(generation_request)

    get.assert_not_called()
