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
import threading
import time
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util import video_decode_worker, video_thumbnails
from invokeai.app.util.video_thumbnails import (
    FrameScore,
    decoder_frame_count,
    extract_video_frame,
    iter_video_frames,
    probe_video,
)

FRAMES = 12
FPS = 8.0


def test_decoder_worker_concurrency_is_globally_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    release = threading.Event()

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout: float | None = None):
            del timeout
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            release.wait(timeout=1)
            with lock:
                active -= 1
            return ("{}", "")

    class FakeMonitor:
        def join(self, timeout: float | None = None) -> None:
            del timeout

    monkeypatch.setattr(video_thumbnails, "_spawn_worker", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(
        video_thumbnails,
        "_start_worker_memory_monitor",
        lambda _proc: (threading.Event(), threading.Event(), FakeMonitor()),
    )

    threads = [
        threading.Thread(target=video_thumbnails._run_worker, args=(["probe", f"{index}.mp4"], 1)) for index in range(4)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert maximum_active <= 2


def test_streamed_decoders_leave_capacity_for_short_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    active_streams = 0
    stream_lock = threading.Lock()
    stream_started = threading.Event()
    release = threading.Event()
    one_shot_started = threading.Event()

    def stream(*_args, **_kwargs):
        nonlocal active_streams
        with stream_lock:
            active_streams += 1
            stream_started.set()
        release.wait(timeout=1)
        return
        yield  # pragma: no cover

    def one_shot(*_args, **_kwargs):
        one_shot_started.set()
        return {}

    monkeypatch.setattr(video_thumbnails, "_iter_video_frames_unbounded", stream)
    monkeypatch.setattr(video_thumbnails, "_run_worker_unbounded", one_shot)

    stream_threads = [
        threading.Thread(target=lambda path=Path(f"{index}.mp4"): list(video_thumbnails.iter_video_frames(path)))
        for index in range(2)
    ]
    for thread in stream_threads:
        thread.start()
    assert stream_started.wait(timeout=1)
    time.sleep(0.1)

    worker_thread = threading.Thread(target=video_thumbnails._run_worker, args=(["probe", "short.mp4"], 1))
    worker_thread.start()
    try:
        assert one_shot_started.wait(timeout=0.2)
    finally:
        release.set()
        for thread in stream_threads:
            thread.join(timeout=2)
        worker_thread.join(timeout=2)


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

    @pytest.mark.slow
    def test_consumer_time_does_not_count_as_decoder_inactivity(self, synthetic_mp4: Path) -> None:
        """Marked slow: this one only holds on a quiet machine.

        The decoder has to deliver its first frame inside the same window the consumer then
        sleeps past, so it fails whenever FFmpeg startup is starved -- observed at 16 and 24 xdist
        workers, and still failing after the window was widened from 2s to 5s. Widening it further
        is not the answer; running it where nothing competes for the cores is, until an injectable
        clock makes the accounting observable without a real timer.
        """
        timeout = 5.0
        frames = iter_video_frames(synthetic_mp4, timeout=timeout)
        next(frames)
        # Sleep past the inactivity timeout; this time belongs to the consumer and must not
        # expire the decoder.
        time.sleep(timeout + 0.5)
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

    def test_cancellation_does_not_wait_for_stream_capacity(self, tmp_path: Path) -> None:
        target = tmp_path / "never-opened.mp4"
        errors: list[BaseException] = []
        finished = Event()
        assert video_thumbnails._VIDEO_STREAM_SLOTS.acquire(timeout=1)

        def consume() -> None:
            try:
                next(iter_video_frames(target, timeout=5, is_canceled=lambda: True))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=consume)
        thread.start()
        try:
            assert finished.wait(timeout=0.2), "canceled decoder waited for an occupied stream slot"
        finally:
            video_thumbnails._VIDEO_STREAM_SLOTS.release()
            thread.join(timeout=2)

        assert len(errors) == 1
        assert isinstance(errors[0], CanceledException)

    def test_timeout_includes_waiting_for_stream_capacity(self, tmp_path: Path) -> None:
        target = tmp_path / "never-opened.mp4"
        errors: list[BaseException] = []
        finished = Event()
        assert video_thumbnails._VIDEO_STREAM_SLOTS.acquire(timeout=1)

        def consume() -> None:
            try:
                next(iter_video_frames(target, timeout=0.1))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=consume)
        thread.start()
        try:
            assert finished.wait(timeout=0.3), "decoder timeout did not include the wait for stream capacity"
        finally:
            video_thumbnails._VIDEO_STREAM_SLOTS.release()
            thread.join(timeout=2)

        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)

    def test_capacity_wait_and_first_frame_share_one_deadline(self, hanging_worker, tmp_path: Path) -> None:
        """Waiting for capacity and waiting for the first frame draw on ONE budget.

        Each used to get a full ``timeout``: a caller that waited nearly the whole budget
        for a stream slot then got a fresh full budget on the hung decoder, so the call
        could take ~2x the bound it advertised. Later frames still get a full timeout
        each — after the first frame the budget is an inactivity bound, not a queueing one.
        """
        target = tmp_path / "malicious.mp4"
        target.write_bytes(b"pretend this hangs the decoder")
        timeout = 2.0
        # Occupy the single stream slot for most (but not all) of the budget, so the
        # decode still starts and the two waits are distinguishable in the total.
        hold = 1.6
        errors: list[BaseException] = []
        finished = Event()
        assert video_thumbnails._VIDEO_STREAM_SLOTS.acquire(timeout=1)
        released = False

        def consume() -> None:
            try:
                next(iter_video_frames(target, timeout=timeout))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        started = time.monotonic()
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            time.sleep(hold)
            video_thumbnails._VIDEO_STREAM_SLOTS.release()
            released = True
            assert finished.wait(timeout=10), "decoder never gave up"
        finally:
            if not released:
                video_thumbnails._VIDEO_STREAM_SLOTS.release()
            thread.join(timeout=10)
        elapsed = time.monotonic() - started

        assert len(errors) == 1
        # "Timed out decoding" (not "Timed out waiting to decode") — the slot was acquired
        # and the worker was spawned, so this is the first-frame wait expiring.
        assert isinstance(errors[0], TimeoutError)
        assert "Timed out decoding" in str(errors[0])
        # Midway between the fixed behavior (~timeout) and the old one (hold + timeout),
        # leaving room for worker-spawn and scheduling jitter on CI.
        assert elapsed < timeout + hold / 2, f"first frame got a fresh timeout after the capacity wait ({elapsed:.2f}s)"

    def test_probe_timeout_includes_waiting_for_decoder_capacity(self, tmp_path: Path) -> None:
        target = tmp_path / "never-opened.mp4"
        errors: list[BaseException] = []
        finished = Event()
        acquired = 0
        for _ in range(video_thumbnails.MAX_CONCURRENT_VIDEO_DECODERS):
            assert video_thumbnails._VIDEO_DECODER_SLOTS.acquire(timeout=1)
            acquired += 1

        def probe() -> None:
            try:
                probe_video(target, timeout=0.1)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=probe)
        thread.start()
        try:
            finished_within_timeout = finished.wait(timeout=0.3)
        finally:
            for _ in range(acquired):
                video_thumbnails._VIDEO_DECODER_SLOTS.release()
            thread.join(timeout=2)

        assert finished_within_timeout, "probe timeout did not include the wait for decoder capacity"
        assert len(errors) == 1
        assert isinstance(errors[0], FileNotFoundError)

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
        monkeypatch.setattr(video_thumbnails, "_worker_tree_rss", lambda worker: 0)
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


class TestWorkerMemoryMonitor:
    def test_terminates_worker_over_memory_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = MagicMock()
        proc.poll.return_value = None
        terminated = Event()
        monkeypatch.setattr(
            video_thumbnails,
            "_worker_tree_rss",
            lambda worker: video_thumbnails.MAX_VIDEO_DECODE_WORKER_RSS_BYTES + 1,
        )
        monkeypatch.setattr(video_thumbnails, "_terminate_process_tree", lambda worker: terminated.set())

        stop, exceeded, monitor = video_thumbnails._start_worker_memory_monitor(proc)
        assert terminated.wait(timeout=1)
        stop.set()
        monitor.join(timeout=1)

        assert exceeded.is_set()

    def test_leaves_worker_within_memory_limit_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = MagicMock()
        proc.poll.return_value = None
        terminated = Event()
        monkeypatch.setattr(
            video_thumbnails,
            "_worker_tree_rss",
            lambda worker: video_thumbnails.MAX_VIDEO_DECODE_WORKER_RSS_BYTES,
        )
        monkeypatch.setattr(video_thumbnails, "_terminate_process_tree", lambda worker: terminated.set())

        stop, exceeded, monitor = video_thumbnails._start_worker_memory_monitor(proc)
        time.sleep(video_thumbnails.WORKER_MEMORY_POLL_SECONDS * 2)
        stop.set()
        monitor.join(timeout=1)

        assert not exceeded.is_set()
        assert not terminated.is_set()


class TestRunWorkerUnexpectedFailureCleanup:
    """An unexpected exception from ``proc.communicate()`` (e.g. OSError on a broken
    pipe or fd exhaustion) must not leak the worker tree: ``_run_worker``'s finally
    stops the RSS-monitor backstop, so if the except path doesn't terminate the tree,
    nothing ever will (JPPhoto non-merge-blocker, 2026-07-22)."""

    def _proc_with_failing_communicate(self, error: Exception) -> MagicMock:
        proc = MagicMock()
        proc.communicate.side_effect = error
        proc.poll.return_value = None
        return proc

    def test_oserror_from_communicate_terminates_worker_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = self._proc_with_failing_communicate(OSError("broken pipe"))
        terminated: list[object] = []
        monkeypatch.setattr(video_thumbnails, "_spawn_worker", lambda *args, **kwargs: proc)
        monkeypatch.setattr(video_thumbnails, "_worker_tree_rss", lambda worker: 0)
        monkeypatch.setattr(video_thumbnails, "_terminate_process_tree", terminated.append)

        assert video_thumbnails._run_worker(["probe", "unused"], timeout=1.0) is None
        assert terminated == [proc]

    def test_spawn_failure_returns_none_without_terminate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail_spawn(*args: object, **kwargs: object) -> None:
            raise OSError("fork failed")

        terminated: list[object] = []
        monkeypatch.setattr(video_thumbnails, "_spawn_worker", _fail_spawn)
        monkeypatch.setattr(video_thumbnails, "_terminate_process_tree", terminated.append)

        assert video_thumbnails._run_worker(["probe", "unused"], timeout=1.0) is None
        assert terminated == []


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


class TestDecodedFrameValidation:
    def test_accepts_frame_within_pixel_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(video_decode_worker, "MAX_FRAME_PIXELS", 4)

        video_decode_worker._validate_decoded_frame(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_rejects_later_frame_over_pixel_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(video_decode_worker, "MAX_FRAME_PIXELS", 4)

        with pytest.raises(ValueError, match="Decoded frame dimensions"):
            video_decode_worker._validate_decoded_frame(np.zeros((3, 2, 3), dtype=np.uint8))

    @pytest.mark.parametrize(
        "frame",
        [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8)],
    )
    def test_rejects_non_rgb_frames(self, frame: np.ndarray) -> None:
        with pytest.raises(ValueError, match="RGB"):
            video_decode_worker._validate_decoded_frame(frame)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (320, 180)) -> Image.Image:
    return Image.new("RGB", size, color)


def _textured(size: tuple[int, int] = (320, 180), brightness: float = 1.0) -> Image.Image:
    """A photograph stand-in: smooth colour gradients with a little grain, dimmable."""
    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(7)
    base = np.stack([40 + x * 0.6, 60 + y, 250 - x * 0.5], axis=-1) + rng.normal(0, 3, (height, width, 3))
    return Image.fromarray(np.clip(base * brightness, 0, 255).astype(np.uint8))


def _title_card(size: tuple[int, int] = (640, 360), noise: float = 1.2) -> Image.Image:
    """Three lines of anti-aliased text on black, over the kind of grain an x264 round trip
    leaves on a black background. Three real title cards measured 0.74-0.94 bits with 93-96%
    of pixels on the background level; at this noise the synthetic one measures ~1.1 bits."""
    rng = np.random.default_rng(3)
    background = np.clip(rng.normal(0, noise, (size[1], size[0])), 0, 255).astype(np.uint8)
    frame = Image.fromarray(np.repeat(background[:, :, None], 3, axis=2))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=size[1] // 10)
    y = size[1] // 2 - 3 * size[1] // 16
    for line in ("Chocolate", "Covered", "Cherry"):
        draw.text(((size[0] - draw.textlength(line, font=font)) / 2, y), line, fill=(255, 255, 255), font=font)
        y += size[1] // 8
    return frame


def _score(frame: Image.Image) -> FrameScore:
    return FrameScore.measure(frame)


class TestFrameScoring:
    """The three gates a thumbnail candidate has to pass: not flat (entropy), not dark
    (highlight luma) and not a card (background occupancy)."""

    def test_solid_frames_carry_no_information(self) -> None:
        for color in ((0, 0, 0), (128, 128, 128), (40, 90, 200)):
            assert _score(_solid(color)).entropy == 0.0
            assert not _score(_solid(color)).informative

    def test_titles_over_black_score_below_the_entropy_floor(self) -> None:
        """Two levels carry almost nothing however sharp the text, and why the score is
        entropy rather than contrast (the standard deviation of this frame is enormous)."""
        assert _score(_title_card(noise=0.0)).entropy < video_thumbnails.FRAME_ENTROPY_FLOOR
        assert not _score(_title_card(noise=0.0)).informative

    def test_an_encoded_title_card_is_rejected_by_the_flat_gate(self) -> None:
        """Codec grain around anti-aliased text lifts a real title card over the entropy floor
        (three from a real library straddled it), and its text is bright, so the first two
        gates pass. What gives it away is that nearly every pixel sits on one level."""
        score = _score(_title_card())
        assert score.entropy > video_thumbnails.FRAME_ENTROPY_FLOOR, "the entropy gate alone would keep it"
        assert score.highlight_luma >= video_thumbnails.FRAME_LUMA_FLOOR, "the luma gate alone would keep it"
        assert score.flat_fraction >= video_thumbnails.FRAME_FLAT_FRACTION_CEILING
        assert not score.informative

    def test_a_dark_but_real_scene_is_kept(self) -> None:
        """A night scene is almost entirely dark, so a dark-pixel count would reject it; what
        separates it from a title card is tone everywhere rather than two levels."""
        frame = Image.new("RGB", (320, 180))
        frame.putdata([(v // 8, v // 8, max(0, v // 8 - 2)) for v in range(180) for _ in range(320)])
        assert _score(frame).entropy > video_thumbnails.FRAME_ENTROPY_FLOOR
        assert _score(frame).informative

    def test_ordinary_content_is_nowhere_near_the_gates(self) -> None:
        score = _score(_textured())
        assert score.entropy > 4 * video_thumbnails.FRAME_ENTROPY_FLOOR
        assert score.flat_fraction < 0.2
        assert score.informative

    def test_a_fade_to_black_is_rejected_though_its_entropy_passes(self) -> None:
        """A photograph dimmed to a peak luma of ~2/255 is black to any viewer, but the grain
        spread over a few levels clears the entropy floor. Only the highlight gate catches it."""
        score = _score(_textured(brightness=0.01))
        assert score.entropy > video_thumbnails.FRAME_ENTROPY_FLOOR
        assert score.highlight_luma < video_thumbnails.FRAME_LUMA_FLOOR
        assert not score.informative

    def test_black_carrying_only_sensor_noise_is_rejected(self) -> None:
        rng = np.random.default_rng(11)
        noise = np.clip(rng.normal(6, 0.5, (180, 320)), 0, 255).astype(np.uint8)
        score = _score(Image.fromarray(np.repeat(noise[:, :, None], 3, axis=2)))
        assert score.entropy > video_thumbnails.FRAME_ENTROPY_FLOOR
        assert not score.informative

    @pytest.mark.parametrize("brightness", [0.10, 0.15, 0.25])
    def test_a_dim_but_lit_scene_survives_the_luma_gate(self, brightness: float) -> None:
        """Dusk is not a fade. The gate must not cost real low-light footage."""
        assert _score(_textured(brightness=brightness)).informative

    def test_a_subject_on_a_plain_background_survives_the_flat_gate(self) -> None:
        """Fireworks on a night sky, a pillarboxed phone video, a portrait on white seamless:
        overwhelmingly one level, yet with a real subject filling a good part of the frame. A
        mean-luma test would throw the dark ones away, and a flat gate set too low would
        throw all of them away — real accepted frames reach 0.68 on this measure."""
        for background in ((2, 2, 4), (255, 255, 255)):
            frame = Image.new("RGB", (640, 360), background)
            frame.paste(_textured((200, 360)), (220, 0))
            score = _score(frame)
            assert 0.6 < score.flat_fraction < video_thumbnails.FRAME_FLAT_FRACTION_CEILING
            assert score.informative

    def test_a_sprinkle_of_hot_pixels_cannot_vouch_for_a_black_frame(self) -> None:
        """Why a quantile and not the maximum: a stuck pixel, a timecode burn-in or a logo is
        a handful of pixels. At 320x180 the 0.999 headroom is 57 pixels, so 40 must not
        count and 400 must."""
        frame = _textured(brightness=0.01)
        for i in range(40):
            frame.putpixel((i, 0), (255, 255, 255))
        assert np.asarray(frame.convert("L")).max() == 255
        assert _score(frame).highlight_luma < video_thumbnails.FRAME_LUMA_FLOOR
        for i in range(400):
            frame.putpixel((i % 320, 1 + i // 320), (255, 255, 255))
        assert _score(frame).highlight_luma == 255.0

    def test_a_pure_white_frame_is_not_scored_as_black(self) -> None:
        """The histogram walk starts at 255: off by one and an all-white frame falls through
        to 0.0 and fails the luma gate on top of the entropy one."""
        assert _score(_solid((255, 255, 255))).highlight_luma == 255.0

    def test_the_quantile_boundary_is_inclusive(self) -> None:
        """1000 px -> headroom int(1000 * 0.001) == 1, so a single bright pixel sits exactly
        on the boundary; a float headroom (1.0000000000000009) would miss it."""
        frame = Image.new("L", (1000, 1), 0)
        frame.putpixel((0, 0), 200)
        assert _score(frame.convert("RGB")).highlight_luma == 200.0

    @pytest.mark.parametrize("floor", [0, 253])
    def test_the_flat_band_absorbs_codec_noise_on_the_background(self, floor: int) -> None:
        """A background that decodes to three adjacent levels rather than exactly one still
        counts as one level, at both ends of the histogram; without the band the noisy title
        card would pass as 'many levels'."""
        rng = np.random.default_rng(5)
        background = rng.integers(floor, floor + 3, (180, 320)).astype(np.uint8)
        frame = Image.fromarray(np.repeat(background[:, :, None], 3, axis=2))
        assert _score(frame).flat_fraction == 1.0

    def test_an_empty_frame_is_kept(self) -> None:
        """An unmeasurable frame is never the reason to discard one."""
        assert _score(Image.new("RGB", (0, 0))).informative

    def test_a_legible_card_outranks_grain_over_black(self) -> None:
        """Among rejected frames, fewest failed gates wins before entropy: grain over black has
        more entropy than a title card, but the card is what a person would pick."""
        card = _score(_title_card())
        rng = np.random.default_rng(11)
        noise = np.clip(rng.normal(6, 0.5, (180, 320)), 0, 255).astype(np.uint8)
        grain = _score(Image.fromarray(np.repeat(noise[:, :, None], 3, axis=2)))
        assert grain.entropy > card.entropy, "the case only matters if entropy alone would pick the grain"
        assert card.rank > grain.rank > _score(_solid((0, 0, 0))).rank


class TestThumbnailFrameCandidates:
    """The seek ladder: ~1s in (capped at the midpoint), deeper rungs at fractions of the
    duration that each move meaningfully later, then frame 0 as the walk-back."""

    @pytest.mark.parametrize(
        ("duration", "fps", "expected"),
        [
            (None, 24.0, [0]),  # unknown duration -> first-frame behavior
            (0.0, 24.0, [0]),
            (-1.0, 24.0, [0]),
            (float("inf"), 24.0, [0]),  # untrusted metadata degrades safely
            (float("nan"), 24.0, [0]),
            (10.0, 24.0, [24, 84, 144, 0]),  # 1s, then 0.35 and 0.6 (0.1 is too close to 1s)
            (10.0, 8.0, [8, 28, 48, 0]),
            (5.0, 24.0, [24, 72, 0]),  # a generated clip: only the 0.6 rung clears the gap
            (2.0, 24.0, [24, 0]),  # every deeper rung is within a second of the opening one
            (1.0, 24.0, [12, 0]),  # short clip: opening rung capped at the midpoint
            (10.0, None, [24, 84, 144, 0]),  # unknown fps -> 24 fps assumption
            (10.0, 0.0, [24, 84, 144, 0]),
            (10.0, float("inf"), [24, 84, 144, 0]),
            (0.01, 24.0, [0]),  # sub-frame midpoint resolves to frame 0 once
            (600.0, 24.0, [24, 1440, 5040, 0]),  # long file: the 0.6 rung is over the cap
        ],
    )
    def test_ladder(self, duration, fps, expected) -> None:
        assert video_thumbnails.thumbnail_frame_candidates(duration, fps) == expected

    @pytest.mark.parametrize("duration", [0.3, 1.2, 2.4, 3.0, 4.7, 30.0, 3600.0])
    def test_rungs_move_later_and_stay_inside_the_clip(self, duration: float) -> None:
        candidates = video_thumbnails.thumbnail_frame_candidates(duration, 24.0)
        rungs = candidates[:-1]
        assert candidates[-1] == 0
        assert rungs == sorted(set(rungs))
        assert len(rungs) <= 1 + video_thumbnails.MAX_DEEPER_SEEK_ATTEMPTS
        assert all(index / 24.0 <= 0.6 * duration for index in rungs)

    @pytest.mark.parametrize(("duration", "fps"), [(10.0, 1e308), (1.7e308, 24.0), (1e200, 1e200)])
    def test_finite_but_absurd_metadata_does_not_raise(self, duration: float, fps: float) -> None:
        """The inputs pass the finiteness guards but their product overflows; such a rung is
        dropped, never raised on (the callers would turn that into a 415 / a lost thumbnail)."""
        candidates = video_thumbnails.thumbnail_frame_candidates(duration, fps)
        assert candidates[-1] == 0
        assert all(isinstance(index, int) for index in candidates)


@pytest.fixture
def ladder_mp4(tmp_path: Path) -> Path:
    """Six seconds at 8 fps: three seconds of black, then a textured scene. The ladder's
    opening rung (1s) and first deeper rung (2.1s) both land in the black lead-in."""
    path = tmp_path / "ladder.mp4"
    black = np.zeros((32, 48, 3), dtype=np.uint8)
    lit = np.asarray(_textured((48, 32)))
    frames = [black] * 24 + [lit] * 24
    iio.imwrite(path, frames, plugin="FFMPEG", codec="libx264", fps=FPS, macro_block_size=1)
    return path


class TestRepresentativeThumbnailFrame:
    """The gallery thumbnail is the first informative frame on the seek ladder, else the best
    of the empty ones — never a black lead-in or title card when the clip has more to offer."""

    def _fake_extract(self, monkeypatch: pytest.MonkeyPatch, served: dict[int, object]) -> list[int]:
        calls: list[int] = []

        def fake_extract(path, frame_index=0, timeout=0.0, raise_on_timeout=False):
            calls.append(frame_index)
            result = served[frame_index]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(video_thumbnails, "extract_video_frame", fake_extract)
        return calls

    def test_returns_the_first_informative_frame(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        lit = _textured()
        calls = self._fake_extract(monkeypatch, {24: _solid((0, 0, 0)), 84: lit})
        frame = video_thumbnails.extract_representative_video_frame(tmp_path / "v.mp4", duration=10.0, fps=24.0)
        assert frame is lit
        assert calls == [24, 84]

    def test_returns_the_best_empty_frame_when_the_whole_ladder_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A clip that really is all dark still gets a thumbnail — the least empty one, which
        for the waveform track wrapping an audio upload is the busiest window."""
        fade = _textured(brightness=0.01)
        card = _title_card()
        calls = self._fake_extract(monkeypatch, {24: _solid((0, 0, 0)), 84: fade, 144: card, 0: _solid((0, 0, 0))})
        frame = video_thumbnails.extract_representative_video_frame(tmp_path / "v.mp4", duration=10.0, fps=24.0)
        assert frame is card
        assert calls == [24, 84, 144, 0]

    def test_an_opening_rung_with_no_frame_goes_straight_to_the_walk_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No frame ~1s in means the metadata overstates the range or the file is damaged
        there; the deeper rungs are later still, so proving them empty would only spend
        worker spawns and budget that the frame-0 walk-back needs."""
        lit = _textured()
        calls = self._fake_extract(monkeypatch, {24: None, 0: lit})
        frame = video_thumbnails.extract_representative_video_frame(tmp_path / "v.mp4", duration=10.0, fps=24.0)
        assert frame is lit
        assert calls == [24, 0]

    def test_a_deeper_rung_with_no_frame_is_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        lit = _textured()
        calls = self._fake_extract(monkeypatch, {24: _solid((0, 0, 0)), 84: None, 144: lit})
        frame = video_thumbnails.extract_representative_video_frame(tmp_path / "v.mp4", duration=10.0, fps=24.0)
        assert frame is lit
        assert calls == [24, 84, 144]

    @pytest.mark.parametrize("raise_on_timeout", [True, False])
    def test_an_exhausted_budget_is_a_timeout_not_a_missing_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raise_on_timeout: bool
    ) -> None:
        """A rung that fails slowly without tripping its own timeout can leave no budget for
        the rest. That is inconclusive, like a timeout — the upload path must not 415 it as
        'no decodable frame'."""
        clock = [0.0]
        calls: list[int] = []

        def slow_failure(path, frame_index=0, timeout=0.0, raise_on_timeout=False):
            calls.append(frame_index)
            clock[0] += 25.0
            return None

        monkeypatch.setattr(video_thumbnails.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(video_thumbnails, "extract_video_frame", slow_failure)
        if raise_on_timeout:
            with pytest.raises(video_thumbnails.VideoDecodeTimeoutError):
                video_thumbnails.extract_representative_video_frame(
                    tmp_path / "v.mp4", duration=10.0, fps=24.0, timeout=10.0, raise_on_timeout=True
                )
        else:
            assert (
                video_thumbnails.extract_representative_video_frame(
                    tmp_path / "v.mp4", duration=10.0, fps=24.0, timeout=10.0, raise_on_timeout=False
                )
                is None
            )
        assert calls == [24]

    def test_single_attempt_when_the_duration_is_unknown(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = self._fake_extract(monkeypatch, {0: None})
        assert video_thumbnails.extract_representative_video_frame(tmp_path / "v.mp4", duration=None, fps=None) is None
        assert calls == [0]

    @pytest.mark.parametrize("raise_on_timeout", [True, False])
    def test_timeout_before_any_frame_ends_the_search(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raise_on_timeout: bool
    ) -> None:
        """A timeout is contention, not evidence about the frame index — another rung would
        hold a request worker for another full budget. Raise mode propagates; the disk
        store's non-raise mode gets None."""
        calls = self._fake_extract(monkeypatch, {24: video_thumbnails.VideoDecodeTimeoutError("busy")})
        if raise_on_timeout:
            with pytest.raises(video_thumbnails.VideoDecodeTimeoutError):
                video_thumbnails.extract_representative_video_frame(
                    tmp_path / "v.mp4", duration=10.0, fps=24.0, raise_on_timeout=True
                )
        else:
            assert (
                video_thumbnails.extract_representative_video_frame(
                    tmp_path / "v.mp4", duration=10.0, fps=24.0, raise_on_timeout=False
                )
                is None
            )
        assert calls == [24]

    @pytest.mark.parametrize("raise_on_timeout", [True, False])
    def test_timeout_after_a_decoded_frame_returns_that_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raise_on_timeout: bool
    ) -> None:
        """A dark thumbnail beats none, and the decoded frame still proves decodability."""
        dark = _solid((0, 0, 0))
        calls = self._fake_extract(monkeypatch, {24: dark, 84: video_thumbnails.VideoDecodeTimeoutError("busy")})
        frame = video_thumbnails.extract_representative_video_frame(
            tmp_path / "v.mp4", duration=10.0, fps=24.0, raise_on_timeout=raise_on_timeout
        )
        assert frame is dark
        assert calls == [24, 84]

    def test_the_whole_search_shares_one_budget_of_two_timeouts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Each rung gets the remainder of the budget, not a fresh timeout, so a slow ladder
        is bounded at 2 * timeout like the two-attempt version was."""
        clock = [0.0]
        timeouts: list[float] = []
        dark = _solid((0, 0, 0))

        def slow_extract(path, frame_index=0, timeout=0.0, raise_on_timeout=False):
            timeouts.append(timeout)
            clock[0] += 15.0
            return dark

        monkeypatch.setattr(video_thumbnails.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(video_thumbnails, "extract_video_frame", slow_extract)
        frame = video_thumbnails.extract_representative_video_frame(
            tmp_path / "v.mp4", duration=10.0, fps=24.0, timeout=10.0
        )
        assert frame is dark
        assert timeouts == [10.0, 5.0]

    def test_walks_past_a_black_lead_in_through_the_worker(self, ladder_mp4: Path) -> None:
        """End to end: the opening rung and the first deeper rung both decode to black and are
        rejected; the search continues to the lit frame instead of settling."""
        duration = 48 / FPS
        assert video_thumbnails.thumbnail_frame_candidates(duration, FPS) == [8, 16, 28, 0]
        frame = video_thumbnails.extract_representative_video_frame(ladder_mp4, duration, FPS)
        assert frame is not None
        assert FrameScore.measure(frame).informative
        assert np.asarray(frame).mean() > 60

    def test_falls_back_to_frame_zero_when_metadata_overstates(self, synthetic_mp4: Path) -> None:
        """Container metadata is untrusted: fps/duration claiming frames that don't exist
        (indices 100+ of a 12-frame clip) must still produce a thumbnail (frame 0), not a
        gallery placeholder."""
        frame = video_thumbnails.extract_representative_video_frame(synthetic_mp4, duration=10.0, fps=100.0)
        assert frame is not None
        value = np.asarray(frame)[0, 0, 0].astype(int)
        assert abs(value - 32) <= 8  # frame 0
