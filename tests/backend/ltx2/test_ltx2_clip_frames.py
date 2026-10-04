"""Reading and encoding a gallery clip's frames onto an LTX-2 canvas.

Every caller of these two functions mocks them away, so without this file `tail` could be a no-op
and the whole suite would still pass -- while every video extension silently anchored on the
opening of the clip it was meant to continue.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import invokeai.backend.ltx2.clip_frames as clip_frames
from invokeai.backend.ltx2.clip_frames import encode_canvas_clip, read_canvas_frames
from invokeai.backend.ltx2.constants import LTX2_LATENT_CHANNELS

CANVAS = {"height": 64, "width": 96}


def _source(count: int, height: int = 128, width: int = 192) -> list[np.ndarray]:
    """Frames whose top-left pixel is their own index, so a slice is identifiable."""
    return [np.full((height, width, 3), index, dtype=np.uint8) for index in range(count)]


@pytest.fixture
def decoded(monkeypatch: pytest.MonkeyPatch):
    def install(frames: list[np.ndarray]) -> dict[str, int]:
        counter = {"decoded": 0}

        def iter_frames(*_args, **_kwargs):
            for frame in frames:
                counter["decoded"] += 1
                yield frame

        monkeypatch.setattr(clip_frames, "iter_video_frames", iter_frames)
        return counter

    return install


def test_the_head_read_stops_at_the_cap(decoded) -> None:
    counter = decoded(_source(40))

    frames, source_frames = read_canvas_frames(None, cap=10, **CANVAS)

    assert [int(frame[0, 0, 0]) for frame in frames] == list(range(10))
    # Stopped rather than decoded and discarded: nothing upstream bounds a clip's length.
    assert counter["decoded"] == 10
    # And so it cannot say how long the clip is -- it never reached the end.
    assert source_frames is None


def test_the_tail_read_keeps_the_END_of_the_clip(decoded) -> None:
    """The whole point of an extension is to continue from where the clip stopped. Keeping the head
    still encodes, still validates, and anchors the continuation to the wrong moment."""
    decoded(_source(40))

    frames, source_frames = read_canvas_frames(None, cap=5, tail=True, **CANVAS)

    assert [int(frame[0, 0, 0]) for frame in frames] == [35, 36, 37, 38, 39]
    # A tail read runs to the end, so it knows the clip's whole length -- which is the only way a
    # caller can convert the picture's span into seconds and line a soundtrack up against it.
    assert source_frames == 40


def test_a_clip_shorter_than_the_cap_comes_back_whole(decoded) -> None:
    decoded(_source(3))

    assert len(read_canvas_frames(None, cap=10, tail=True, **CANVAS).frames) == 3


def test_every_frame_returned_is_on_the_canvas(decoded) -> None:
    decoded(_source(4, height=128, width=192))

    for tail in (False, True):
        frames, _ = read_canvas_frames(None, cap=4, tail=tail, **CANVAS)
        assert all(frame.shape == (64, 96, 3) for frame in frames), tail


def test_the_tail_read_fits_only_the_frames_it_keeps(monkeypatch: pytest.MonkeyPatch, decoded) -> None:
    """A tail read cannot stop early -- the last frames are only known once the clip ends -- so
    fitting on the way past would run a LANCZOS resize over the whole recording to keep a second of
    it: ~145 s for a five-minute source, and twice that on a two-stage run."""
    decoded(_source(40))
    fits = {"count": 0}
    real_fit = clip_frames.fit_to_canvas

    def counting_fit(image, height, width):
        fits["count"] += 1
        return real_fit(image, height, width)

    monkeypatch.setattr(clip_frames, "fit_to_canvas", counting_fit)

    read_canvas_frames(None, cap=5, tail=True, **CANVAS)

    assert fits["count"] == 5


def test_the_encode_releases_the_frames_it_was_handed() -> None:
    """The stack is a second copy of the clip; holding both across the encode is hundreds of MiB at
    the callers' own caps that no model-cache budget accounts for."""
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(9)]
    captured: dict[str, torch.Tensor] = {}

    def encode(pixels):
        captured["pixels"] = pixels
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, LTX2_LATENT_CHANNELS, 2, 2, 3)))

    vae = SimpleNamespace(
        buffers=lambda: iter([]),
        config=SimpleNamespace(scaling_factor=1.0),
        enable_tiling=lambda **_kwargs: None,
        encode=encode,
        encoder=torch.nn.Identity(),
        latents_mean=torch.zeros(LTX2_LATENT_CHANNELS),
        latents_std=torch.ones(LTX2_LATENT_CHANNELS),
        parameters=lambda: iter([torch.zeros(1)]),
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
        use_framewise_decoding=False,
        use_framewise_encoding=False,
        use_tiling=False,
    )

    encode_canvas_clip(vae, frames, tile_size=512, temporal_tile=16)

    assert frames == []
    # And the clip reached the VAE as [1, 3, T, H, W] in [-1, 1].
    assert captured["pixels"].shape == (1, 3, 9, 64, 96)
    assert float(captured["pixels"].min()) == pytest.approx(-1.0, abs=2e-3)
