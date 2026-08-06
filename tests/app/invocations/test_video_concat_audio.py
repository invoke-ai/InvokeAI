"""Audio-preservation tests for VideoConcatInvocation.

The concat node originally decoded and re-encoded video frames only, silently dropping
the audio track of any input that had one (H3 clips carry AAC; Wan clips are silent).
These tests pin the rebuilt soundtrack path: extraction, per-clip timeline mapping,
boundary blending per transition mode, silent-input handling, fps-override retiming, and
the final mux.

Real ffmpeg (the imageio-ffmpeg bundled binary) encodes and decodes throughout — tone
clips are tiny (64x64) so the suite stays fast.
"""

from pathlib import Path

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.util.video_audio import extract_audio_pcm, mux_audio_into_video, resample_linear
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


def _make_clip(dirpath: Path, name: str, n_frames: int, freq: float | None, gray: int = 128, fps: float = FPS) -> Path:
    path = dirpath / f"{name}.mp4"
    audio_path = None
    if freq is not None:
        audio_path = dirpath / f"{name}.wav"
        write_stereo_wav(audio_path, _tone(freq, n_frames / fps), RATE)
    writer = make_mp4_writer(path, fps, audio_path=audio_path)
    frame = np.full((SIZE, SIZE, 3), gray, dtype=np.uint8)
    for _ in range(n_frames):
        writer.append_data(frame)
    writer.close()
    return path


def _dominant_freq(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm[0]))
    spectrum[0] = 0.0  # ignore DC
    return float(np.fft.rfftfreq(pcm.shape[1], 1.0 / rate)[int(np.argmax(spectrum))])


def _invocation(n_videos: int = 2, transition: str = "cut", transition_frames: int = 8) -> VideoConcatInvocation:
    return VideoConcatInvocation(
        videos=[VideoField(video_name=f"clip{i}") for i in range(n_videos)],
        transition=transition,  # type: ignore[arg-type]
        transition_frames=transition_frames,
    )


class TestExtraction:
    def test_roundtrip_tone(self, tmp_path):
        clip = _make_clip(tmp_path, "a", 32, freq=440.0)
        extracted = extract_audio_pcm(clip)
        assert extracted is not None
        pcm, rate = extracted
        assert rate == RATE
        assert pcm.shape[0] == 2
        assert abs(pcm.shape[1] / rate - 2.0) < 0.15  # ~2 s, AAC padding tolerated
        assert abs(_dominant_freq(pcm, rate) - 440.0) < 10.0

    def test_silent_clip_returns_none(self, tmp_path):
        clip = _make_clip(tmp_path, "s", 16, freq=None)
        assert extract_audio_pcm(clip) is None


class TestResample:
    def test_lengths_and_identity(self):
        pcm = _tone(440.0, 1.0)
        assert resample_linear(pcm, pcm.shape[1]).shape == pcm.shape
        assert resample_linear(pcm, 1234).shape == (2, 1234)
        assert resample_linear(np.zeros((2, 0), dtype=np.float32), 100).shape == (2, 100)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shaped"):
            resample_linear(np.zeros(10, dtype=np.float32), 5)


class TestBuildAudioTrack:
    def _build(self, paths, frame_counts, invocation=None, output_fps=FPS, native=None, durations=None):
        inv = invocation or _invocation(n_videos=len(paths))
        return inv._build_audio_track(
            context=_Ctx(),  # type: ignore[arg-type]
            paths=paths,
            frame_counts=frame_counts,
            native_rates=native if native is not None else [FPS] * len(paths),
            native_durations=durations if durations is not None else [None] * len(paths),
            output_fps=output_fps,
            num_output_frames=sum(frame_counts)
            if (invocation or _invocation()).transition != "crossfade"
            else sum(frame_counts) - (len(paths) - 1) * (invocation or _invocation()).transition_frames,
        )

    def test_cut_splices_tracks(self, tmp_path):
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = self._build([a, b], [32, 32])
        assert rate == RATE
        assert pcm.shape[1] == round(64 * RATE / FPS)
        half = pcm.shape[1] // 2
        assert abs(_dominant_freq(pcm[:, :half], rate) - 440.0) < 10.0
        assert abs(_dominant_freq(pcm[:, half:], rate) - 880.0) < 10.0

    def test_silent_input_contributes_silence(self, tmp_path):
        a = _make_clip(tmp_path, "a", 32, freq=None)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = self._build([a, b], [32, 32])
        half = pcm.shape[1] // 2
        assert np.abs(pcm[:, : half - RATE // 10]).max() == 0.0
        assert np.abs(pcm[:, half:]).max() > 0.2

    def test_all_silent_returns_none(self, tmp_path):
        a = _make_clip(tmp_path, "a", 16, freq=None)
        b = _make_clip(tmp_path, "b", 16, freq=None)
        assert self._build([a, b], [16, 16]) is None

    def test_crossfade_length_and_blend(self, tmp_path):
        tf = 8
        inv = _invocation(transition="crossfade", transition_frames=tf)
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = inv._build_audio_track(
            context=_Ctx(),  # type: ignore[arg-type]
            paths=[a, b],
            frame_counts=[32, 32],
            native_rates=[FPS, FPS],
            native_durations=[None, None],
            output_fps=FPS,
            num_output_frames=32 + 32 - tf,
        )
        assert pcm.shape[1] == round((64 - tf) * RATE / FPS)
        # Ends stay pure.
        quarter = pcm.shape[1] // 4
        assert abs(_dominant_freq(pcm[:, :quarter], rate) - 440.0) < 10.0
        assert abs(_dominant_freq(pcm[:, -quarter:], rate) - 880.0) < 10.0
        # The overlap window contains BOTH tones (equal-power blend, not a hard cut).
        window = round(tf * RATE / FPS)
        start = round((32 - tf) * RATE / FPS)
        overlap = pcm[:, start : start + window]
        spectrum = np.abs(np.fft.rfft(overlap[0]))
        freqs = np.fft.rfftfreq(overlap.shape[1], 1.0 / rate)
        e440 = spectrum[(np.abs(freqs - 440.0) < 30)].max()
        e880 = spectrum[(np.abs(freqs - 880.0) < 30)].max()
        assert e440 > 0.1 * e880 and e880 > 0.1 * e440

        # Orientation: the outgoing tone dominates early in the overlap, the incoming
        # tone late — a swapped tail/head blend fails these.
        def band_energy(seg, freq):
            spec = np.abs(np.fft.rfft(seg[0]))
            f = np.fft.rfftfreq(seg.shape[1], 1.0 / rate)
            return spec[np.abs(f - freq) < 60].max()

        third = window // 3
        early, late = overlap[:, :third], overlap[:, -third:]
        assert band_energy(early, 440.0) > band_energy(early, 880.0)
        assert band_energy(late, 880.0) > band_energy(late, 440.0)

    def test_fade_through_black_preserves_length_and_dips(self, tmp_path):
        tf = 8
        inv = _invocation(transition="fade_through_black", transition_frames=tf)
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = inv._build_audio_track(
            context=_Ctx(),  # type: ignore[arg-type]
            paths=[a, b],
            frame_counts=[32, 32],
            native_rates=[FPS, FPS],
            native_durations=[None, None],
            output_fps=FPS,
            num_output_frames=64,
        )
        assert pcm.shape[1] == round(64 * RATE / FPS)
        boundary = round(32 * RATE / FPS)
        window = round(1 * RATE / FPS)  # one frame around the silence point
        rms_boundary = float(np.sqrt(np.mean(pcm[:, boundary - window : boundary + window] ** 2)))
        rms_mid = float(np.sqrt(np.mean(pcm[:, boundary // 2 : boundary // 2 + 2 * window] ** 2)))
        assert rms_boundary < 0.25 * rms_mid

    def test_fps_override_retimes_audio(self, tmp_path):
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=440.0)
        pcm, rate = self._build([a, b], [32, 32], output_fps=2 * FPS)
        # Video plays twice as fast -> audio spans half the wall clock (and pitches up).
        assert pcm.shape[1] == round(64 * RATE / (2 * FPS))
        assert abs(_dominant_freq(pcm, rate) - 880.0) < 20.0

    def test_fade_through_black_tf1_keeps_audio_aligned(self, tmp_path):
        """Regression: tf=1 has an EMPTY tail window (1 // 2 == 0) but still a boundary.

        The head frame's audio used to be silently dropped, shifting everything after the
        boundary earlier and leaving trailing silence masked by the final pad."""
        inv = _invocation(transition="fade_through_black", transition_frames=1)
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = inv._build_audio_track(
            context=_Ctx(),  # type: ignore[arg-type]
            paths=[a, b],
            frame_counts=[32, 32],
            native_rates=[FPS, FPS],
            native_durations=[None, None],
            output_fps=FPS,
            num_output_frames=64,
        )
        assert pcm.shape[1] == round(64 * RATE / FPS)
        # No trailing silence: the final one-frame window carries the full 880 Hz tone.
        one_frame = round(RATE / FPS)
        assert float(np.sqrt(np.mean(pcm[:, -one_frame:] ** 2))) > 0.2
        # And the boundary is where it belongs: 880 Hz dominates right after it.
        boundary = round(32 * RATE / FPS)
        after = pcm[:, boundary + one_frame : boundary + 5 * one_frame]
        assert abs(_dominant_freq(after, rate) - 880.0) < 10.0

    def test_unknown_native_fps_uses_probed_duration_for_retime(self, tmp_path):
        """Regression: with probe fps None and an fps override, the retime used to be
        skipped entirely — audio played at the wrong speed and truncated/padded."""
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=440.0)
        pcm, rate = self._build([a, b], [32, 32], output_fps=2 * FPS, native=[None, None], durations=[2.0, 2.0])
        assert pcm.shape[1] == round(64 * RATE / (2 * FPS))
        assert abs(_dominant_freq(pcm, rate) - 880.0) < 20.0
        # Both halves stay audible — the broken path left the second half silent.
        half = pcm.shape[1] // 2
        assert float(np.sqrt(np.mean(pcm[:, :half] ** 2))) > 0.2
        assert float(np.sqrt(np.mean(pcm[:, half:] ** 2))) > 0.2

    def test_unknown_fps_and_duration_falls_back_to_extracted_length(self, tmp_path):
        a = _make_clip(tmp_path, "a", 32, freq=440.0)
        b = _make_clip(tmp_path, "b", 32, freq=880.0)
        pcm, rate = self._build([a, b], [32, 32], native=[None, None], durations=[None, None])
        assert pcm.shape[1] == round(64 * RATE / FPS)
        half = pcm.shape[1] // 2
        assert abs(_dominant_freq(pcm[:, : half - RATE // 4], rate) - 440.0) < 15.0
        assert abs(_dominant_freq(pcm[:, half + RATE // 4 :], rate) - 880.0) < 15.0


class TestMux:
    def test_end_to_end_cut_produces_audible_mp4(self, tmp_path):
        """The full pipeline the node runs: silent video encode -> audio build -> mux."""
        inv = _invocation()
        a = _make_clip(tmp_path, "a", 32, freq=440.0, gray=60)
        b = _make_clip(tmp_path, "b", 32, freq=880.0, gray=200)

        from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video

        video_only = tmp_path / "joined_video_only.mp4"
        writer = make_mp4_writer(video_only, FPS)
        frame_counts: list[int] = []
        for frame in inv._iter_joined_frames([iter_video_frames(a), iter_video_frames(b)], frame_counts=frame_counts):
            writer.append_data(frame)
        writer.close()
        assert frame_counts == [32, 32]
        assert extract_audio_pcm(video_only) is None

        pcm, rate = inv._build_audio_track(
            context=_Ctx(),  # type: ignore[arg-type]
            paths=[a, b],
            frame_counts=frame_counts,
            native_rates=[FPS, FPS],
            native_durations=[None, None],
            output_fps=FPS,
            num_output_frames=64,
        )
        wav = tmp_path / "joined.wav"
        write_stereo_wav(wav, pcm, rate)
        out = tmp_path / "joined.mp4"
        mux_audio_into_video(video_only, wav, out)

        width, height, _, fps = probe_video(out)
        assert (width, height) == (SIZE, SIZE)
        extracted = extract_audio_pcm(out)
        assert extracted is not None
        out_pcm, out_rate = extracted
        assert abs(out_pcm.shape[1] / out_rate - 4.0) < 0.2
        half = out_pcm.shape[1] // 2
        assert abs(_dominant_freq(out_pcm[:, : half - out_rate // 4], out_rate) - 440.0) < 10.0
        assert abs(_dominant_freq(out_pcm[:, half + out_rate // 4 :], out_rate) - 880.0) < 10.0
