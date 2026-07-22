from unittest.mock import MagicMock, patch

import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.primitives import VideoInvocation


@pytest.mark.parametrize("decoded_count,expected", [(7, 7), (None, 8)])
def test_video_primitive_prefers_exact_decoder_frame_count(decoded_count: int | None, expected: int) -> None:
    invocation = VideoInvocation(video=VideoField(video_name="input.mp4"))
    context = MagicMock()
    context.videos.get_dto.return_value = MagicMock(video_name="input.mp4", width=64, height=64, duration=1.0, fps=8.0)
    context.videos.get_path.return_value = "input.mp4"

    with patch("invokeai.app.util.video_thumbnails.decoder_frame_count", return_value=decoded_count):
        output = invocation.invoke(context)

    assert output.num_frames == expected
