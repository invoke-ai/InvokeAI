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

import sys
import time
from pathlib import Path
from threading import Event

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
    assert video_thumbnails._run_worker(["probe", "unused"], timeout=0.5) is None
    child_pid = int(child_pid_path.read_text())

    deadline = time.monotonic() + 5
    while video_thumbnails._is_process_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not video_thumbnails._is_process_running(child_pid)
