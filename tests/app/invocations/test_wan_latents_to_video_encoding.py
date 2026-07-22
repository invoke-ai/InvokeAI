from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from invokeai.app.invocations.wan_latents_to_video import (
    _iter_decoded_frames,
    _validate_video_latent_batch,
    _write_video_frames,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException


def test_video_encoding_stops_when_canceled() -> None:
    writer = MagicMock()
    yielded: list[int] = []

    def frames():
        for index in range(5):
            yielded.append(index)
            yield np.zeros((4, 4, 3), dtype=np.uint8)

    checks = iter([False, False, True])

    with pytest.raises(CanceledException):
        _write_video_frames(writer, frames(), lambda: next(checks))

    assert writer.append_data.call_count == 2
    assert yielded == [0, 1]


def test_decoded_frames_are_converted_lazily_in_temporal_order() -> None:
    decoded = torch.tensor([-2.0, 0.0, 0.0, 2.0]).view(1, 4, 1, 1).expand(3, -1, -1, -1).clone()
    frames = _iter_decoded_frames(decoded)

    first = next(frames)
    decoded[:, 1] = 1.0
    second = next(frames)
    third = next(frames)
    fourth = next(frames)

    assert first.shape == (1, 1, 3)
    assert first.dtype == np.uint8
    assert first.tolist() == [[[0, 0, 0]]]
    assert second.tolist() == [[[255, 255, 255]]]
    assert third.tolist() == [[[128, 128, 128]]]
    assert fourth.tolist() == [[[255, 255, 255]]]
    with pytest.raises(StopIteration):
        next(frames)


@pytest.mark.parametrize("shape", [(1, 16, 2, 4, 4), (1, 16, 4, 4)])
def test_single_video_latent_batch_is_accepted(shape: tuple[int, ...]) -> None:
    _validate_video_latent_batch(torch.zeros(shape))


@pytest.mark.parametrize("shape", [(2, 16, 2, 4, 4), (2, 16, 4, 4)])
def test_multiple_video_latent_batches_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="batch size 1"):
        _validate_video_latent_batch(torch.zeros(shape))
