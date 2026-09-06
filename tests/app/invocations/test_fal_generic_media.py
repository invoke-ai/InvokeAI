from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.invocations.external_media_generation import FalGenericMediaInvocation
from invokeai.app.invocations.fields import ImageField, VideoField


def test_fal_generic_media_invocation_uploads_local_media_fields() -> None:
    context = MagicMock()
    context.images.get_pil.return_value = Image.new("RGB", (2, 2))
    context.videos.get_path.return_value = Path("/tmp/input.mp4")
    context._services.external_generation.generate_generic.return_value = {}
    invocation = FalGenericMediaInvocation(
        id="fal_generic",
        model_id="fal-ai/video",
        input_json='{"start_image_url": "${image_url}", "video_url": "${video_url}"}',
        image=ImageField(image_name="input.png"),
        mask=ImageField(image_name="mask.png"),
        reference_images=[ImageField(image_name="ref.png")],
        video=VideoField(video_name="input.mp4"),
    )

    invocation.invoke(context)

    call = context._services.external_generation.generate_generic.call_args
    assert call.kwargs["image"].size == (2, 2)
    assert call.kwargs["mask_image"].size == (2, 2)
    assert len(call.kwargs["reference_images"]) == 1
    assert call.kwargs["video_path"] == Path("/tmp/input.mp4")


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
