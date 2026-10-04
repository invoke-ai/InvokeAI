"""Tests for TextLLM API request/response models and validation."""

from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.utilities import (
    ExpandPromptRequest,
    ExpandPromptResponse,
    ImageToPromptRequest,
    _model_load_lock,
    _run_expand_prompt,
    _run_image_to_prompt,
)
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice
from tests.fixtures.device_pool import GPU0, GPU1, two_gpu_pool


class TestExpandPromptRequest:
    def test_defaults(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123")
        assert req.max_tokens == 300
        assert req.system_prompt is None
        assert req.seed is None

    def test_max_tokens_upper_bound(self):
        """max_tokens should be capped at 2048."""
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=2049)

    def test_max_tokens_lower_bound(self):
        """max_tokens must be at least 1."""
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=0)

    def test_max_tokens_valid_range(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=2048)
        assert req.max_tokens == 2048
        req2 = ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=1)
        assert req2.max_tokens == 1

    def test_custom_system_prompt(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123", system_prompt="Be brief.")
        assert req.system_prompt == "Be brief."

    def test_seed_range(self):
        assert ExpandPromptRequest(prompt="a cat", model_key="abc-123", seed=42).seed == 42
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", seed=-1)


class TestImageToPromptRequest:
    def test_defaults(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123")
        assert "Describe" in req.instruction

    def test_custom_instruction(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123", instruction="What is this?")
        assert req.instruction == "What is this?"


class TestExpandPromptResponse:
    def test_success_response(self):
        resp = ExpandPromptResponse(expanded_prompt="A detailed scene", seed=42)
        assert resp.expanded_prompt == "A detailed scene"
        assert resp.seed == 42
        assert resp.error is None

    def test_error_response(self):
        resp = ExpandPromptResponse(expanded_prompt="", seed=42, error="Model failed")
        assert resp.error == "Model failed"


def test_expand_prompt_uses_fresh_seed() -> None:
    model_config = MagicMock(type=ModelType.TextLLM, path="model")
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    loaded_model = MagicMock()
    loaded_model.model_on_device.return_value.__enter__.return_value = (None, model)
    services = MagicMock()
    services.model_manager.store.get_model.return_value = model_config
    services.model_manager.load.load_model.return_value = loaded_model

    with (
        patch.object(ApiDependencies, "invoker", MagicMock(services=services), create=True),
        patch("invokeai.app.api.routers.utilities._resolve_model_path", return_value="model"),
        patch("invokeai.app.api.routers.utilities.AutoTokenizer.from_pretrained"),
        patch("invokeai.app.api.routers.utilities.get_random_seed", return_value=123),
        patch("invokeai.app.api.routers.utilities.TextLLMPipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.return_value = "expanded"
        assert _run_expand_prompt("cat", "model", 10, None, None, None, "user") == ("expanded", 123)
        assert pipeline_class.return_value.run.call_args.kwargs["seed"] == 123

        pipeline_class.return_value.run.reset_mock()
        assert _run_expand_prompt("cat", "model", 10, None, 456, None, "user") == ("expanded", 456)
        assert pipeline_class.return_value.run.call_args.kwargs["seed"] == 456


def test_expand_prompt_with_an_image_runs_it_through_the_processor() -> None:
    """The stored image reaches the pipeline as RGB, next to the model's processor, whose tokenizer is reused."""
    model_config = MagicMock(type=ModelType.TextLLM, path="model", supports_images=True)
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    loaded_model = MagicMock()
    loaded_model.model_on_device.return_value.__enter__.return_value = (None, model)
    services = MagicMock()
    services.model_manager.store.get_model.return_value = model_config
    services.model_manager.load.load_model.return_value = loaded_model
    services.images.get_pil_image.return_value = Image.new("RGBA", (8, 8))

    with (
        patch.object(ApiDependencies, "invoker", MagicMock(services=services), create=True),
        patch("invokeai.app.api.routers.utilities._resolve_model_path", return_value="model"),
        patch("invokeai.app.api.routers.utilities.AutoTokenizer.from_pretrained") as load_tokenizer,
        patch("invokeai.app.api.routers.utilities.AutoProcessor.from_pretrained") as load_processor,
        patch("invokeai.app.api.routers.utilities.TextLLMPipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.return_value = "expanded"
        _run_expand_prompt("she waves", "model", 10, None, 1, None, "user", image_name="frame.png")

    services.images.get_pil_image.assert_called_once_with("frame.png")
    processor = load_processor.return_value
    load_tokenizer.assert_not_called()
    assert pipeline_class.call_args.args[1:] == (processor.tokenizer, processor)
    assert pipeline_class.return_value.run.call_args.kwargs["image"].mode == "RGB"


def _services_recording_devices(model: object, seen: list[object]) -> MagicMock:
    """Model-manager services whose load and image read record the thread's session device."""
    loaded_model = MagicMock()
    loaded_model.model_on_device.return_value.__enter__.return_value = (None, model)
    services = MagicMock()

    def load_model(*args: object, **kwargs: object) -> MagicMock:
        seen.append(("load", TorchDevice.get_session_device()))
        return loaded_model

    def read_image(name: str) -> Image.Image:
        seen.append(("image read", TorchDevice.get_session_device()))
        return Image.new("RGB", (8, 8))

    services.model_manager.load.load_model.side_effect = load_model
    services.images.get_pil_image.side_effect = read_image
    return services


def _recording(seen: list[object], label: str, result: object) -> Callable[..., object]:
    def record(*args: object, **kwargs: object) -> object:
        seen.append((label, TorchDevice.get_session_device()))
        return result

    return record


def _borrow_recording_load_lock(seen: list[object]) -> Callable[[str], object]:
    """The real off-queue borrow, noting whether the model-load lock was already held."""
    real = GENERATION_DEVICE_POOL.try_borrow_off_queue

    def borrow(device_type: str) -> object:
        seen.append(("borrow under load lock", _model_load_lock.locked()))
        return real(device_type)

    return borrow


def test_expand_prompt_loads_and_runs_on_the_idle_gpu() -> None:
    """With a render holding GPU 0, the LLM loads into and runs on GPU 1, and the thread is unpinned after.

    The tokenizer read comes before the borrow and the borrow after the load lock, so neither disk I/O
    nor a wait for another request's load holds a GPU a session could use.
    """
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    seen: list[object] = []
    services = _services_recording_devices(model, seen)
    services.model_manager.store.get_model.return_value = MagicMock(type=ModelType.TextLLM, path="model")

    with (
        two_gpu_pool(busy=(GPU0,)),
        patch.object(ApiDependencies, "invoker", MagicMock(services=services), create=True),
        patch("invokeai.app.api.routers.utilities._resolve_model_path", return_value="model"),
        patch(
            "invokeai.app.api.routers.utilities.AutoTokenizer.from_pretrained",
            side_effect=_recording(seen, "tokenizer read", MagicMock()),
        ),
        patch.object(GENERATION_DEVICE_POOL, "try_borrow_off_queue", side_effect=_borrow_recording_load_lock(seen)),
        patch("invokeai.app.api.routers.utilities.TextLLMPipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.side_effect = _recording(seen, "run", "expanded")
        _run_expand_prompt("cat", "model", 10, None, 1, None, "user")
        assert TorchDevice.get_session_device() is None

    assert seen == [
        ("tokenizer read", None),
        ("borrow under load lock", True),
        ("load", GPU1),
        ("run", GPU1),
    ]


def test_image_to_prompt_loads_and_runs_on_the_idle_gpu() -> None:
    model = MagicMock(spec=LlavaOnevisionForConditionalGeneration)
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    seen: list[object] = []
    services = _services_recording_devices(model, seen)
    services.model_manager.store.get_model.return_value = MagicMock(type=ModelType.LlavaOnevision, path="model")

    with (
        two_gpu_pool(busy=(GPU0,)),
        patch.object(ApiDependencies, "invoker", MagicMock(services=services), create=True),
        patch("invokeai.app.api.routers.utilities._resolve_model_path", return_value="model"),
        patch(
            "invokeai.app.api.routers.utilities.AutoProcessor.from_pretrained",
            side_effect=_recording(seen, "processor read", MagicMock(spec=LlavaOnevisionProcessor)),
        ),
        patch.object(GENERATION_DEVICE_POOL, "try_borrow_off_queue", side_effect=_borrow_recording_load_lock(seen)),
        patch("invokeai.app.api.routers.utilities.LlavaOnevisionPipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.side_effect = _recording(seen, "run", "a photo")
        _run_image_to_prompt("frame.png", "llava", "describe", None, "user")
        assert TorchDevice.get_session_device() is None

    assert seen == [
        ("image read", None),
        ("processor read", None),
        ("borrow under load lock", True),
        ("load", GPU1),
        ("run", GPU1),
    ]
