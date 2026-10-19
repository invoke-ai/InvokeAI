"""Audio-preservation tests for ExtractVideoRangeInvocation.

The node originally re-encoded video frames only, so trimming a clip silently stripped
its soundtrack — feeding a trimmed clip into Concatenate Videos then produced correct
silence for its span (the "extend video" workflow lost the source's audio). These tests
pin the carried-through audio: slice alignment against the frame range, fps-override
retiming, the frame-rate fallback chain, and silent-source passthrough.

Real ffmpeg (the imageio-ffmpeg bundled binary) encodes and decodes throughout — tone
clips are tiny (64x64) so the suite stays fast.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_frame_extract_range import ExtractVideoRangeInvocation
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav

RATE = 32000
FPS = 16.0
SIZE = 64


class _Util:
    def signal_progress(self, *args, **kwargs) -> None:
        pass

    def is_canceled(self) -> bool:
        return False


class _Ctx:
    util = _Util()


def _tone(freq: float, seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def _dominant_freq(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm[0]))
    spectrum[0] = 0.0  # ignore DC
    return float(np.fft.rfftfreq(pcm.shape[1], 1.0 / rate)[int(np.argmax(spectrum))])


def _make_clip(dirpath: Path, name: str, n_frames: int, audio: np.ndarray | None, fps: float = FPS) -> Path:
    path = dirpath / f"{name}.mp4"
    audio_path = None
    if audio is not None:
        audio_path = dirpath / f"{name}.wav"
        write_stereo_wav(audio_path, audio, RATE)
    writer = make_mp4_writer(path, fps, audio_path=audio_path)
    frame = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    for _ in range(n_frames):
        writer.append_data(frame)
    writer.close()
    return path


def _invoke_and_capture(invocation: ExtractVideoRangeInvocation, clip_path: Path, out_dir: Path) -> Path:
    """Runs invoke() with a mocked context and returns a copy of the saved MP4.

    The node unlinks its temp files in a finally block, so ``videos.save`` copies the
    file out before returning.
    """
    saved = out_dir / "saved.mp4"

    def _save(source_path: Path, **kwargs) -> MagicMock:
        shutil.copyfile(source_path, saved)
        return MagicMock()

    context = MagicMock()
    context.videos.get_path.return_value = clip_path
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save
    base_output = MagicMock(
        video=VideoField(video_name="saved.mp4"), width=SIZE, height=SIZE, num_frames=1, fps=FPS, duration=1.0
    )
    with patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output):
        invocation.invoke(context)
    assert saved.exists()
    return saved


class TestRangeAudio:
    def test_trim_keeps_the_matching_audio_slice(self, tmp_path):
        # 2 s clip: first second 440 Hz, second second 880 Hz. Trimming the back half
        # must carry the 880 Hz second, not the 440 Hz one (slice offset correctness).
        audio = np.concatenate([_tone(440.0, 1.0), _tone(880.0, 1.0)], axis=1)
        clip = _make_clip(tmp_path, "twotone", n_frames=32, audio=audio)
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="twotone"), start_frame=16, end_frame=31)
        saved = _invoke_and_capture(node, clip, tmp_path)

        extracted = extract_audio_pcm(saved)
        assert extracted is not None
        pcm, rate = extracted
        assert abs(pcm.shape[1] / rate - 1.0) < 0.05  # ~1 s of audio for 16 frames @ 16 fps
        assert abs(_dominant_freq(pcm, rate) - 880.0) < 20.0

    def test_full_range_round_trips_audio(self, tmp_path):
        clip = _make_clip(tmp_path, "tone", n_frames=32, audio=_tone(440.0, 2.0))
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="tone"), start_frame=0, end_frame=-1)
        saved = _invoke_and_capture(node, clip, tmp_path)

        extracted = extract_audio_pcm(saved)
        assert extracted is not None
        pcm, rate = extracted
        assert abs(pcm.shape[1] / rate - 2.0) < 0.05
        assert abs(_dominant_freq(pcm, rate) - 440.0) < 20.0

    def test_silent_source_stays_silent(self, tmp_path):
        clip = _make_clip(tmp_path, "silent", n_frames=16, audio=None)
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="silent"), start_frame=0, end_frame=7)
        saved = _invoke_and_capture(node, clip, tmp_path)
        assert extract_audio_pcm(saved) is None

    def test_fps_override_retimes_audio_with_video(self, tmp_path):
        # Doubling the frame rate halves the duration; the audio must follow the video's
        # retime (half the samples, pitch doubled) rather than play at original speed.
        clip = _make_clip(tmp_path, "tone", n_frames=32, audio=_tone(440.0, 2.0))
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="tone"), start_frame=0, end_frame=-1, fps=32)
        saved = _invoke_and_capture(node, clip, tmp_path)

        extracted = extract_audio_pcm(saved)
        assert extracted is not None
        pcm, rate = extracted
        assert abs(pcm.shape[1] / rate - 1.0) < 0.05
        assert abs(_dominant_freq(pcm, rate) - 880.0) < 25.0


class TestBuildAudioTrackFallbacks:
    """The frame->sample mapping's frame-rate fallback chain, with synthetic extraction."""

    def _build(self, node, source_fps, source_duration, pcm, start, end, n_frames, output_fps=FPS):
        with patch(
            "invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm",
            return_value=(pcm, RATE),
        ):
            return node._build_audio_track(
                context=_Ctx(),  # type: ignore[arg-type]
                video_path=Path("unused.mp4"),
                start=start,
                end=end,
                n_frames=n_frames,
                source_fps=source_fps,
                source_duration=source_duration,
                output_fps=output_fps,
            )

    def test_unknown_fps_uses_probed_duration(self):
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="x"))
        pcm = np.concatenate([_tone(440.0, 1.0), _tone(880.0, 1.0)], axis=1)
        result = self._build(node, source_fps=None, source_duration=2.0, pcm=pcm, start=16, end=31, n_frames=32)
        assert result is not None
        out, rate = result
        assert out.shape[1] == round(16 / FPS * RATE)
        assert abs(_dominant_freq(out, rate) - 880.0) < 20.0

    def test_unknown_fps_and_duration_falls_back_to_extracted_length(self):
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="x"))
        pcm = np.concatenate([_tone(440.0, 1.0), _tone(880.0, 1.0)], axis=1)
        result = self._build(node, source_fps=None, source_duration=None, pcm=pcm, start=16, end=31, n_frames=32)
        assert result is not None
        out, rate = result
        assert abs(_dominant_freq(out, rate) - 880.0) < 20.0

    def test_short_audio_track_is_padded_not_stretched(self):
        # Audio covers only the first second of a 2 s video; trimming the back half must
        # yield silence (aligned), not the front half's audio stretched over the range.
        node = ExtractVideoRangeInvocation(video=VideoField(video_name="x"))
        pcm = _tone(440.0, 1.0)
        result = self._build(node, source_fps=FPS, source_duration=2.0, pcm=pcm, start=16, end=31, n_frames=32)
        assert result is not None
        out, _rate = result
        assert out.shape[1] == round(16 / FPS * RATE)
        assert float(np.abs(out).max()) < 1e-6
