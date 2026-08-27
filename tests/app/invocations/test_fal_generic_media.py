from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.external_media_generation import FalGenericMediaInvocation


def test_fal_generic_media_invocation_submits_declared_json_and_returns_raw_result() -> None:
    context = MagicMock()
    context._services.external_generation.generate_generic.return_value = {
        "video": {"url": "https://cdn.test/video.mp4"},
        "seed": 7,
    }
    invocation = FalGenericMediaInvocation(
        id="fal_generic",
        model_id="fal-ai/video",
        input_json='{"prompt": "A moving test", "duration": "5"}',
    )

    output = invocation.invoke(context)

    context._services.external_generation.generate_generic.assert_called_once_with(
        provider_id="fal",
        model_id="fal-ai/video",
        payload={"prompt": "A moving test", "duration": "5"},
    )
    assert output.value == '{"seed": 7, "video": {"url": "https://cdn.test/video.mp4"}}'


def test_fal_generic_media_invocation_rejects_non_object_json() -> None:
    context = MagicMock()
    invocation = FalGenericMediaInvocation(id="fal_generic", model_id="fal-ai/video", input_json="[]")

    with pytest.raises(ValueError, match="JSON object"):
        invocation.invoke(context)
