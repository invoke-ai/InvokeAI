"""Bounded bridge encoding and timeline planning for smart video concatenation."""

import io
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.h264_splice import SpliceError
from invokeai.app.util.video_smart_concat import _bridge_sample_entry, encode_rgb_frames


def test_bridge_encoder_cancellation_reaps_ffmpeg_and_discards_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "canceled.mp4"
    canceled = [False]
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def frames():
        yield frame
        canceled[0] = True
        yield frame

    with pytest.raises(CanceledException):
        encode_rgb_frames(frames(), output, (64, 64), 16.0, 8, is_canceled=lambda: canceled[0])

    assert not output.exists()


def test_bridge_encoder_timeout_stops_a_blocked_frame_source(tmp_path: Path) -> None:
    output = tmp_path / "timed-out.mp4"
    stop = threading.Event()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def frames():
        stop.wait()
        yield frame

    with pytest.raises(SpliceError, match="timed out"):
        encode_rgb_frames(frames(), output, (64, 64), 16.0, 8, abort_event=stop, timeout=0.05)

    assert stop.is_set()
    assert not output.exists()


def test_bridge_encoder_reports_process_start_failure(tmp_path: Path) -> None:
    output = tmp_path / "no-ffmpeg.mp4"
    with patch("invokeai.app.util.video_smart_concat.subprocess.Popen", side_effect=OSError("not found")):
        with pytest.raises(SpliceError, match="could not start ffmpeg"):
            encode_rgb_frames([], output, (64, 64), 16.0, 8)

    assert not output.exists()


def test_bridge_encoder_does_not_treat_internal_abort_as_user_cancellation(tmp_path: Path) -> None:
    output = tmp_path / "internal-abort.mp4"
    abort = threading.Event()
    user_canceled = threading.Event()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def frames():
        abort.wait()
        yield frame

    with pytest.raises(SpliceError, match="timed out"):
        encode_rgb_frames(
            frames(),
            output,
            (64, 64),
            16.0,
            8,
            is_canceled=user_canceled.is_set,
            abort_event=abort,
            timeout=0.05,
        )

    assert abort.is_set()
    assert not output.exists()


def test_bridge_source_value_error_is_a_fallback_eligible_splice_error(tmp_path: Path) -> None:
    output = tmp_path / "decoder-error.mp4"
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def frames():
        yield frame
        raise ValueError("source decoder failed")

    with pytest.raises(SpliceError, match="source decoder failed"):
        encode_rgb_frames(frames(), output, (64, 64), 16.0, 8)

    assert not output.exists()


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"bounded diagnostic test")
        self.returncode = None
        self.waited = False

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_bridge_encoder_reaps_process_if_worker_thread_cannot_start(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "worker-start-failure.mp4"
    process = _FakeProcess()
    monkeypatch.setattr("invokeai.app.util.video_smart_concat.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "invokeai.app.util.video_smart_concat.threading.Thread.start",
        lambda self: (_ for _ in ()).throw(RuntimeError("thread limit")),
    )

    with pytest.raises(SpliceError, match="could not start bridge worker"):
        encode_rgb_frames([], output, (64, 64), 16.0, 8)

    assert process.waited
    assert process.returncode == -9
    assert not output.exists()


def test_bridge_encoder_reaps_process_and_feeder_if_cancel_callback_raises(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "callback-failure.mp4"
    process = _FakeProcess()
    monkeypatch.setattr("invokeai.app.util.video_smart_concat.subprocess.Popen", lambda *args, **kwargs: process)
    source_started = threading.Event()
    abort = threading.Event()
    checks = 0

    def frames():
        source_started.set()
        abort.wait()
        if not abort.is_set():
            yield np.zeros((64, 64, 3), dtype=np.uint8)

    def callback() -> bool:
        nonlocal checks
        checks += 1
        if checks == 1:
            return False
        assert source_started.wait(timeout=1)
        raise RuntimeError("cancellation callback failed")

    with pytest.raises(RuntimeError, match="cancellation callback failed"):
        encode_rgb_frames(frames(), output, (64, 64), 16.0, 8, is_canceled=callback, abort_event=abort)

    assert process.waited
    assert process.returncode == -9
    assert not output.exists()
    assert not any(
        thread.name == "invokeai-video-bridge-feed" and thread.is_alive() for thread in threading.enumerate()
    )


def test_bridge_sample_entry_probe_caches_success(monkeypatch) -> None:
    monkeypatch.setattr("invokeai.app.util.video_smart_concat._bridge_sample_entry_cache", {})
    calls = 0

    def count_encode(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr("invokeai.app.util.video_smart_concat.encode_rgb_frames", count_encode)
    monkeypatch.setattr(
        "invokeai.app.util.video_smart_concat.read_h264_layout",
        lambda path: SimpleNamespace(sample_entry=b"entry"),
    )
    assert _bridge_sample_entry((64, 64), 17.0) is not None
    assert _bridge_sample_entry((64, 64), 17.0) is not None
    assert _bridge_sample_entry((64, 64), 18.0) is not None
    assert _bridge_sample_entry((32, 64), 17.0) is not None
    assert calls == 3


def test_bridge_sample_entry_probe_retries_failed_read(monkeypatch) -> None:
    monkeypatch.setattr("invokeai.app.util.video_smart_concat._bridge_sample_entry_cache", {})
    calls = 0
    reads = iter([None, SimpleNamespace(sample_entry=b"entry")])

    def count_encode(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr("invokeai.app.util.video_smart_concat.encode_rgb_frames", count_encode)
    monkeypatch.setattr("invokeai.app.util.video_smart_concat.read_h264_layout", lambda path: next(reads))
    assert _bridge_sample_entry((64, 64), 17.0) is None
    assert _bridge_sample_entry((64, 64), 17.0) == b"entry"
    assert calls == 2


def test_bridge_sample_entry_probe_cancellation_after_read_does_not_cache(monkeypatch) -> None:
    cache = {}
    monkeypatch.setattr("invokeai.app.util.video_smart_concat._bridge_sample_entry_cache", cache)
    canceled = [False]
    monkeypatch.setattr("invokeai.app.util.video_smart_concat.encode_rgb_frames", lambda *args, **kwargs: None)

    def read_layout(path):
        canceled[0] = True
        return SimpleNamespace(sample_entry=b"entry")

    monkeypatch.setattr("invokeai.app.util.video_smart_concat.read_h264_layout", read_layout)
    with pytest.raises(CanceledException):
        _bridge_sample_entry((64, 64), 17.0, is_canceled=lambda: canceled[0])
    assert ((64, 64), 17.0) not in cache
