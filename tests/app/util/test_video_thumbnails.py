"""Tests for the subprocess-bounded video decode helpers (PR #9163 review).

The bug: ``probe_video`` / ``extract_video_frame`` decoded untrusted uploads in-process
with no timeout, despite the module itself noting that cv2 has historically hung on some
containers. A crafted MP4 that makes the imageio probe fail and then blocks inside
``cv2.VideoCapture()`` would pin the FastAPI request worker that called it forever;
repeated uploads could exhaust the worker pool. Decoding now runs in a killable child
process with a hard timeout.

The hang tests substitute a worker command that never returns and assert the helpers
fail within a bounded interval; the happy-path tests run the real worker against a real
synthetic MP4 so the subprocess plumbing is actually validated end to end.
"""

import io
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import imageio.v3 as iio
import numpy as np
import pytest

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util import video_thumbnails
from invokeai.app.util.video_thumbnails import decoder_frame_count, extract_video_frame, iter_video_frames, probe_video

FRAMES = 12
FPS = 8.0


@pytest.fixture
def synthetic_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "synth.mp4"
    frames = [np.full((32, 48, 3), 32 + i * 16, dtype=np.uint8) for i in range(FRAMES)]
    iio.imwrite(path, frames, plugin="FFMPEG", codec="libx264", fps=FPS, macro_block_size=1)
    return path


@pytest.fixture
def hanging_worker(monkeypatch: pytest.MonkeyPatch):
    """Replaces the decode worker with a child process that sleeps forever."""

    def _hang_command(*args: str) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(600)"]

    monkeypatch.setattr(video_thumbnails, "_worker_command", _hang_command)


class TestHappyPathThroughSubprocess:
    def test_probe_returns_metadata(self, synthetic_mp4: Path) -> None:
        width, height, duration, fps = probe_video(synthetic_mp4)
        assert (width, height) == (48, 32)
        assert fps == pytest.approx(FPS)
        assert duration == pytest.approx(FRAMES / FPS, abs=0.5)

    def test_extract_frame_returns_image(self, synthetic_mp4: Path) -> None:
        frame = extract_video_frame(synthetic_mp4, frame_index=0)
        assert frame is not None
        assert frame.size == (48, 32)

    def test_frame_count_matches(self, synthetic_mp4: Path) -> None:
        assert decoder_frame_count(synthetic_mp4) == FRAMES


class TestUnreadableInput:
    def test_probe_rejects_garbage_bytes(self, tmp_path: Path) -> None:
        bogus = tmp_path / "junk.mp4"
        bogus.write_bytes(b"not actually an mp4")
        with pytest.raises(FileNotFoundError):
            probe_video(bogus)

    def test_extract_frame_returns_none_for_garbage_bytes(self, tmp_path: Path) -> None:
        bogus = tmp_path / "junk.mp4"
        bogus.write_bytes(b"not actually an mp4")
        assert extract_video_frame(bogus) is None


class TestHungDecoderIsBounded:
    """A decode backend that never returns must fail within the timeout, not hang the caller."""

    TIMEOUT = 2.0
    # Generous ceiling for CI scheduling jitter; the point is "seconds, not forever".
    MAX_ELAPSED = 15.0

    def test_probe_fails_within_bounded_interval(self, hanging_worker, tmp_path: Path) -> None:
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        started = time.monotonic()
        with pytest.raises(FileNotFoundError):
            probe_video(target, timeout=self.TIMEOUT)
        assert time.monotonic() - started < self.MAX_ELAPSED

    def test_extract_frame_fails_within_bounded_interval(self, hanging_worker, tmp_path: Path) -> None:
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        started = time.monotonic()
        assert extract_video_frame(target, timeout=self.TIMEOUT) is None
        assert time.monotonic() - started < self.MAX_ELAPSED

    def test_frame_count_fails_within_bounded_interval(self, hanging_worker, tmp_path: Path) -> None:
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        started = time.monotonic()
        assert decoder_frame_count(target, timeout=self.TIMEOUT) is None
        assert time.monotonic() - started < self.MAX_ELAPSED


class TestStreamedDecoderIsBounded:
    def test_streams_real_frames_through_worker(self, synthetic_mp4: Path) -> None:
        frames = list(iter_video_frames(synthetic_mp4))
        assert len(frames) == FRAMES
        assert frames[0].shape == (32, 48, 3)

    def test_consumer_time_does_not_count_as_decoder_inactivity(self, synthetic_mp4: Path) -> None:
        frames = iter_video_frames(synthetic_mp4, timeout=2.0)
        next(frames)
        # Sleep longer than the inactivity timeout. This time belongs to the consumer and
        # must not expire the decoder, while the two-second window avoids treating normal
        # process/FFmpeg scheduling latency on macOS CI as a decoder hang.
        time.sleep(2.2)
        assert next(frames).shape == (32, 48, 3)

    def test_times_out_when_worker_stops_producing_frames(self, hanging_worker, tmp_path: Path) -> None:
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timed out decoding"):
            next(iter_video_frames(target, timeout=0.2))
        assert time.monotonic() - started < 5

    def test_cancellation_terminates_blocked_decoder(self, hanging_worker, tmp_path: Path) -> None:
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        canceled = Event()
        canceled.set()
        with pytest.raises(CanceledException):
            next(iter_video_frames(target, timeout=5, is_canceled=canceled.is_set))

    def test_midstream_worker_failure_includes_stderr(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "failed.mp4"
        target.write_bytes(b"unused")
        script = (
            "import io, struct, sys, numpy as np; "
            "record = io.BytesIO(); "
            "np.save(record, np.zeros((2, 2, 3), dtype=np.uint8), allow_pickle=False); "
            "payload = record.getvalue(); "
            "sys.stdout.buffer.write(struct.pack('>Q', len(payload)) + payload); "
            "sys.stdout.buffer.flush(); "
            "print('decoder exploded', file=sys.stderr); "
            "raise SystemExit(3)"
        )
        monkeypatch.setattr(video_thumbnails, "_worker_command", lambda *args: [sys.executable, "-c", script])

        frames = iter_video_frames(target)
        assert next(frames).shape == (2, 2, 3)
        with pytest.raises(ValueError, match="decoder exploded"):
            next(frames)

    def test_worker_stderr_capture_is_bounded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "noisy.mp4"
        target.write_bytes(b"unused")
        script = "import sys; sys.stderr.write('x' * 100000); raise SystemExit(3)"
        monkeypatch.setattr(video_thumbnails, "_worker_command", lambda *args: [sys.executable, "-c", script])

        with pytest.raises(ValueError) as error:
            next(iter_video_frames(target))

        assert len(str(error.value)) < video_thumbnails.MAX_DECODE_STDERR_BYTES + 500

    def test_closed_stream_from_live_worker_does_not_leak_timeout_expired(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "stalled.mp4"
        target.write_bytes(b"unused")
        proc = MagicMock()
        proc.stdout = io.BytesIO()
        proc.stderr = io.BytesIO()
        proc.poll.return_value = None

        def wait(timeout: float | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired("decoder-worker", timeout)
            return -9

        proc.wait.side_effect = wait
        monkeypatch.setattr(video_thumbnails, "_spawn_worker", lambda *args, **kwargs: proc)
        monkeypatch.setattr(video_thumbnails, "_terminate_process_tree", lambda worker: None)

        with pytest.raises(TimeoutError, match="decoder worker"):
            next(iter_video_frames(target, timeout=0.2))


def test_timeout_kills_worker_descendants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"

    def _descendant_command(*args: str) -> list[str]:
        script = (
            "import pathlib, subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
            "time.sleep(600)"
        )
        return [sys.executable, "-c", script, str(child_pid_path)]

    monkeypatch.setattr(video_thumbnails, "_worker_command", _descendant_command)
    # The timeout must comfortably cover interpreter startup + the child spawn on a
    # loaded CI machine — if the worker is killed before it writes the pid file, the
    # test can't observe the descendant and fails spuriously.
    assert video_thumbnails._run_worker(["probe", "unused"], timeout=2.0) is None
    child_pid = int(child_pid_path.read_text())

    deadline = time.monotonic() + 5
    while video_thumbnails._is_process_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not video_thumbnails._is_process_running(child_pid)


class TestProbeMetadataValidation:
    """probe_video must reject decoder-reported metadata no sane video has (JPPhoto
    review 2026-07-21): the upload path persists these values and sizes thumbnail
    decoding by them, so zero/negative/absurd dimensions and non-finite durations must
    fail before ``videos.create`` ever sees them."""

    def _probe_with(self, monkeypatch: pytest.MonkeyPatch, payload: dict) -> tuple:
        monkeypatch.setattr(video_thumbnails, "_run_worker", lambda args, timeout: payload)
        return probe_video(Path("fake.mp4"))

    @pytest.mark.parametrize(
        "payload",
        [
            {"width": 0, "height": 32, "duration": 1.0, "fps": 8.0},
            {"width": 48, "height": -32, "duration": 1.0, "fps": 8.0},
            {"width": 100_000, "height": 100_000, "duration": 1.0, "fps": 8.0},
            {"width": 48, "height": 32, "duration": float("nan"), "fps": 8.0},
            {"width": 48, "height": 32, "duration": float("inf"), "fps": 8.0},
            {"width": 48, "height": 32, "duration": -1.0, "fps": 8.0},
            {"width": float("inf"), "height": 32, "duration": 1.0, "fps": 8.0},
            {"width": float("nan"), "height": 32, "duration": 1.0, "fps": 8.0},
        ],
        ids=[
            "zero-width",
            "negative-height",
            "over-limit-pixels",
            "nan-duration",
            "inf-duration",
            "negative-duration",
            "inf-width",
            "nan-width",
        ],
    )
    def test_invalid_metadata_is_rejected(self, monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
        with pytest.raises(FileNotFoundError):
            self._probe_with(monkeypatch, payload)

    def test_boundary_dimensions_are_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 8192x8192 = 64 MP, exactly at the cap.
        width, height, duration, fps = self._probe_with(
            monkeypatch, {"width": 8192, "height": 8192, "duration": 0.0, "fps": 8.0}
        )
        assert (width, height, duration, fps) == (8192, 8192, 0.0, 8.0)

    @pytest.mark.parametrize("bad_fps", [float("nan"), float("inf"), -1.0, 0.0])
    def test_unusable_fps_is_coerced_to_unknown(self, monkeypatch: pytest.MonkeyPatch, bad_fps: float) -> None:
        # fps=None already means "unknown" to every caller, so garbage fps degrades to
        # that instead of rejecting an otherwise-decodable file.
        *_rest, fps = self._probe_with(monkeypatch, {"width": 48, "height": 32, "duration": 1.0, "fps": bad_fps})
        assert fps is None


class TestWorkerDecodeBounds:
    """The worker refuses to decode frames from files whose reported dimensions exceed
    the bound — the full frame is only allocated at decode time, so a small crafted
    container claiming 100k x 100k would otherwise attempt a ~30 GB allocation."""

    def test_oversized_dims_refused_before_frame_decode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from invokeai.app.util import video_decode_worker as worker

        monkeypatch.setattr(worker, "_probe", lambda path: (100_000, 100_000, 1.0, 8.0))
        with pytest.raises(ValueError, match="exceed the maximum decodable size"):
            worker._assert_decodable_dims(Path("fake.mp4"))

    def test_unprobeable_file_is_refused_before_decode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from invokeai.app.util import video_decode_worker as worker

        def _raise(path: Path):
            raise FileNotFoundError("no metadata")

        monkeypatch.setattr(worker, "_probe", _raise)
        with pytest.raises(ValueError, match="Unable to validate video dimensions"):
            worker._assert_decodable_dims(Path("fake.mp4"))

    @pytest.mark.parametrize("width,height", [(0, 32), (48, 0), (-1, 32), (48, -1)])
    def test_non_positive_dims_refused_before_decode(
        self, monkeypatch: pytest.MonkeyPatch, width: int, height: int
    ) -> None:
        from invokeai.app.util import video_decode_worker as worker

        monkeypatch.setattr(worker, "_probe", lambda path: (width, height, 1.0, 8.0))
        with pytest.raises(ValueError, match="invalid dimensions"):
            worker._assert_decodable_dims(Path("fake.mp4"))

    def test_in_bounds_dims_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from invokeai.app.util import video_decode_worker as worker

        monkeypatch.setattr(worker, "_probe", lambda path: (48, 32, 1.0, 8.0))
        worker._assert_decodable_dims(Path("fake.mp4"))  # must not raise


class TestStreamBackendFallback:
    """_stream must fall back to cv2 like probe/frame/count do (JPPhoto review
    2026-07-21) — a file accepted at upload via the cv2 fallback previously worked in
    single-frame extract but failed in the frame-range and concat nodes."""

    def _collect_stream(self, monkeypatch: pytest.MonkeyPatch, video_path: Path) -> list[np.ndarray]:
        from invokeai.app.util import video_decode_worker as worker

        buffer = io.BytesIO()
        fake_stdout = MagicMock()
        fake_stdout.buffer = buffer
        monkeypatch.setattr(worker.sys, "stdout", fake_stdout)
        worker._stream(video_path)

        import struct

        frames: list[np.ndarray] = []
        buffer.seek(0)
        while size_bytes := buffer.read(8):
            (size,) = struct.unpack(">Q", size_bytes)
            frames.append(np.load(io.BytesIO(buffer.read(size)), allow_pickle=False))
        return frames

    def test_falls_back_to_cv2_when_imageio_cannot_stream(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_mp4: Path
    ) -> None:
        from invokeai.app.util import video_decode_worker as worker

        def _imiter_fails(*args, **kwargs):
            raise RuntimeError("imageio cannot decode this container")
            yield  # pragma: no cover - makes this a generator

        monkeypatch.setattr(worker.iio, "imiter", _imiter_fails)

        frames = self._collect_stream(monkeypatch, synthetic_mp4)

        assert len(frames) == FRAMES
        assert frames[0].shape == (32, 48, 3)

    def test_falls_back_to_cv2_when_imageio_stream_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_mp4: Path
    ) -> None:
        from invokeai.app.util import video_decode_worker as worker

        monkeypatch.setattr(worker.iio, "imiter", lambda *args, **kwargs: iter(()))

        frames = self._collect_stream(monkeypatch, synthetic_mp4)

        assert len(frames) == FRAMES
        assert frames[0].shape == (32, 48, 3)

    def test_midstream_failure_does_not_restart_from_frame_zero(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_mp4: Path
    ) -> None:
        from invokeai.app.util import video_decode_worker as worker

        def _imiter_dies_midstream(*args, **kwargs):
            yield np.zeros((32, 48, 3), dtype=np.uint8)
            raise RuntimeError("decoder died mid-stream")

        monkeypatch.setattr(worker.iio, "imiter", _imiter_dies_midstream)

        with pytest.raises(RuntimeError, match="mid-stream"):
            self._collect_stream(monkeypatch, synthetic_mp4)

    def test_totally_undecodable_file_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        bogus = tmp_path / "junk.mp4"
        bogus.write_bytes(b"not actually an mp4")
        # imageio raises first; cv2 then fails to open -> FileNotFoundError. Either way
        # the worker must error rather than emit an empty, successful stream.
        with pytest.raises((FileNotFoundError, ValueError)):
            self._collect_stream(monkeypatch, bogus)
