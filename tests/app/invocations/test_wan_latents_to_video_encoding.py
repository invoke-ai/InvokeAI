from unittest.mock import MagicMock

import numpy as np
import pytest

from invokeai.app.invocations.wan_latents_to_video import _write_video_frames
from invokeai.app.services.session_processor.session_processor_common import CanceledException


def test_video_encoding_stops_when_canceled() -> None:
    writer = MagicMock()
    frames = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    checks = iter([False, False, True])

    with pytest.raises(CanceledException):
        _write_video_frames(writer, frames, lambda: next(checks))

    assert writer.append_data.call_count == 2
