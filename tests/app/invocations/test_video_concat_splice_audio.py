"""Verify the audio timeline, soundtrack splice, presentation tags and cancellation of the H.264/AAC splice."""

import importlib
import struct
import subprocess
from unittest.mock import MagicMock, patch

import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.h264_splice import read_h264_layout
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import (
    write_stereo_wav,
)
from tests.app.invocations.video_concat_splice_helpers import (
    FPS,
    RATE,
    _concat,
    _decode,
    _encode,
    _frames,
    _logged,
    _tone,
)

pytestmark = pytest.mark.ffmpeg


def test_soundtrack_from_another_encoder_is_re_encoded(tmp_path):
    a_video = _encode(tmp_path / "av.mp4", _frames(64, seed=1))
    a_wav = tmp_path / "a.wav"
    write_stereo_wav(a_wav, _tone(440.0, 64), RATE)
    a = tmp_path / "a.mp4"
    # Mono AAC: a different AudioSpecificConfig from the stereo tracks this app writes.
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(a_video), "-i", str(a_wav)]
        + ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-ac", "1", str(a)],
        check=True,
    )
    b = _encode(tmp_path / "b.mp4", _frames(64, seed=2), tone=880.0)
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "audio samples of the first clip" not in _logged(context)
    extracted = extract_audio_pcm(out)
    assert extracted is not None
    assert extracted[0].shape[1] / RATE == pytest.approx(128 / FPS, abs=0.1)


@pytest.mark.parametrize("b_tone", [660.0, None], ids=["b-audible", "b-silent"])
@pytest.mark.parametrize(
    "transition,transition_frames", [("cut", 0), ("crossfade", 2), ("crossfade", 8), ("fade_through_black", 3)]
)
def test_spliced_soundtrack_matches_a_full_re_encode(tmp_path, monkeypatch, transition, transition_frames, b_tone):
    """Copying the head must neither move what follows nor glitch the seam: the spliced
    soundtrack matches a full re-encode of the same timeline everywhere, seam included, to
    within the coding noise of two lossy encodes. A silent continuation (a Wan clip) must
    splice too: its silence codes with no error and is no measure of a seam."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2), tone=b_tone)
    fields = {"transition": transition, "transition_frames": transition_frames}
    spliced, context = _concat(tmp_path, [a, b], "spliced", **fields)
    assert "audio samples of the first clip without re-encoding" in _logged(context)

    monkeypatch.setattr(VideoConcatInvocation, "_mux_spliced_audio", lambda *args, **kwargs: False)
    full, _ = _concat(tmp_path, [a, b], "full", **fields)

    got, want = extract_audio_pcm(spliced), extract_audio_pcm(full)
    assert got is not None and want is not None
    n = min(got[0].shape[1], want[0].shape[1])
    assert float(np.abs(got[0][:, :n] - want[0][:, :n]).max()) < 0.1


@pytest.mark.parametrize(
    "tag",
    [("-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"), ("-aspect", "16:9")],
    ids=["colr", "pasp"],
)
def test_first_clip_with_presentation_tags_is_re_encoded(tmp_path, tag):
    """The joined track would present the re-encoded continuation with the copied head's colour
    or aspect tags, which were never applied to it."""
    _encode(tmp_path / "plain.mp4", _frames(100, seed=1))
    a = tmp_path / "a.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp_path / "plain.mp4")]
        + ["-c", "copy", *tag, str(a)],
        check=True,
    )
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, context = _concat(tmp_path, [a, b], "out", transition="crossfade", transition_frames=2)
    assert "Stream-copied" not in _logged(context)
    layout = read_h264_layout(out)
    assert layout is not None and b"colr" not in layout.sample_entry and b"pasp" not in layout.sample_entry
    assert len(_decode(out)) == 138


@pytest.mark.parametrize(
    "stage",
    [
        ("invokeai.app.invocations.video_concat.copy_frames", "crossfade"),
        ("invokeai.app.invocations.video_concat.concat_stream_copy", "cut"),
        ("invokeai.app.invocations.video_concat.splice_soundtrack", "cut"),
        # The audio decodes: an input's soundtrack, and the splice's own validation decodes.
        ("invokeai.app.invocations.video_concat.extract_audio_pcm", "cut"),
        ("invokeai.app.util.aac_splice.extract_audio_pcm", "cut"),
    ],
)
def test_cancellation_during_a_splice_saves_nothing(tmp_path, stage):
    """Cancelling while a splice stage runs stops it — the stage itself raises, not just the
    pre-save check — and nothing is saved."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2), tone=660.0)
    canceled = [False]
    raised: list[BaseException] = []
    stage, transition = stage
    module, name = stage.rsplit(".", 1)
    real = getattr(importlib.import_module(module), name)

    def cancel_then_run(*args, **kwargs):
        canceled[0] = True
        try:
            return real(*args, **kwargs)
        except BaseException as e:
            raised.append(e)
            raise

    context = MagicMock()
    context.videos.get_path.side_effect = lambda name: {"a": a, "b": b}[name]
    context.util.is_canceled.side_effect = lambda: canceled[0]
    node = VideoConcatInvocation(
        videos=[VideoField(video_name="a"), VideoField(video_name="b")],
        transition=transition,
        transition_frames=2 if transition == "crossfade" else 0,
    )
    with patch(stage, side_effect=cancel_then_run):
        with pytest.raises(CanceledException):
            node.invoke(context)
    assert len(raised) == 1 and isinstance(raised[0], CanceledException)
    context.videos.save.assert_not_called()


def test_tkhd_display_size_does_not_leak_onto_the_join(tmp_path):
    """ffmpeg turns a track header display size that disagrees with the coded size into a
    pasp when it copies the track: the copied head would then stretch the new clip too."""
    source = _encode(tmp_path / "plain.mp4", _frames(100, seed=1))
    data = bytearray(source.read_bytes())
    tkhd = data.find(b"tkhd")
    assert data[tkhd + 4] == 0  # version 0: width at payload offset 76, as 16.16 fixed point
    (width,) = struct.unpack_from(">I", data, tkhd + 4 + 76)
    struct.pack_into(">I", data, tkhd + 4 + 76, width * 2)
    a = tmp_path / "a.mp4"
    a.write_bytes(bytes(data))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))

    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    layout = read_h264_layout(out)
    assert layout is not None and b"pasp" not in layout.sample_entry
    assert len(_decode(out)) == 140
