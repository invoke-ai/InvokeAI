"""Video frame/probe helpers used by the video file store, upload router, and video nodes.

Decoding runs in a short-lived child process (``video_decode_worker.py``) with a hard
timeout. Files reaching these helpers are user uploads, and both decode backends can
hang indefinitely on crafted or malformed containers (cv2 wheels historically, ffmpeg in
degenerate cases). An in-process hang would tie up the FastAPI request worker that
called us — repeated crafted uploads could exhaust the worker pool — and a hung thread
cannot be killed from Python, so process isolation is the only reliable bound.
The parent explicitly terminates the worker and its descendants when the timeout
expires, so a hostile file costs at most ``timeout`` seconds and cannot leak a stuck
FFmpeg process.
"""

import io
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import psutil
from PIL import Image

from invokeai.app.services.session_processor.session_processor_common import CanceledException

# Generous — a healthy decode of a single frame or of container metadata takes well
# under a second even for large files. The timeout exists to bound adversarial or hung
# decodes, not to police slow ones.
VIDEO_DECODE_TIMEOUT_SECONDS = 30.0
MAX_DECODED_FRAME_RECORD_BYTES = 256 * 1024 * 1024

_WORKER_PATH = Path(__file__).parent / "video_decode_worker.py"


def get_video_thumbnail_name(video_name: str) -> str:
    """Given a video file name (e.g. <uuid>.mp4), returns the matching thumbnail name (e.g. <uuid>.webp)."""
    return os.path.splitext(video_name)[0] + ".webp"


def _worker_command(*args: str) -> list[str]:
    """Command line for one decode-worker invocation (patchable in tests).

    The worker is run by file path rather than ``-m`` so the child process doesn't
    import the invokeai package (and transitively torch) just to decode a frame.
    """
    return [sys.executable, str(_WORKER_PATH), *args]


def _spawn_worker(*args: str, **kwargs: Any) -> subprocess.Popen[Any]:
    """Starts a worker in an independently killable process group on POSIX."""
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(_worker_command(*args), **kwargs)


def _is_process_running(pid: int) -> bool:
    """Returns whether a process exists and is not a zombie."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Kills a worker and every descendant it spawned."""
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        parent = psutil.Process(proc.pid)
        processes = parent.children(recursive=True)
        processes.append(parent)
        for process in processes:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(processes, timeout=5)
    except psutil.Error:
        try:
            proc.kill()
        except OSError:
            pass


def _run_worker(args: list[str], timeout: float) -> Optional[dict[str, Any]]:
    """Runs the decode worker; returns its parsed JSON output, or None on failure or timeout."""
    try:
        proc = _spawn_worker(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        proc.communicate()
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        result = json.loads(stdout)
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


def iter_video_frames(
    video_path: Path,
    timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> Iterator[np.ndarray]:
    """Streams decoded frames from an isolated worker with bounded memory and wait time."""
    proc = _spawn_worker(
        "stream",
        str(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdout is None:
        _terminate_process_tree(proc)
        raise RuntimeError("Unable to open video decoder output stream")

    results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    stopped = threading.Event()

    def read_exactly(size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                if remaining == size:
                    raise EOFError
                raise OSError("Truncated frame record from video decoder")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def put_result(result: tuple[str, object]) -> None:
        while not stopped.is_set():
            try:
                results.put(result, timeout=0.1)
                return
            except queue.Full:
                continue

    def read_frames() -> None:
        try:
            while not stopped.is_set():
                record_size = struct.unpack(">Q", read_exactly(8))[0]
                if record_size > MAX_DECODED_FRAME_RECORD_BYTES:
                    raise ValueError(f"Decoded frame record exceeds {MAX_DECODED_FRAME_RECORD_BYTES} bytes")
                payload = read_exactly(record_size)
                put_result(("frame", np.load(io.BytesIO(payload), allow_pickle=False)))
        except (EOFError, ValueError, OSError) as error:
            put_result(("done", error))

    reader = threading.Thread(target=read_frames, name="video-frame-reader", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            if is_canceled is not None and is_canceled():
                raise CanceledException
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out decoding frames from {video_path}")
            try:
                kind, value = results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if kind == "frame":
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"Decoder returned an invalid frame for {video_path}")
                yield value
                deadline = time.monotonic() + timeout
                continue
            return_code = proc.wait(timeout=1)
            if return_code != 0:
                raise ValueError(f"Unable to decode video at {video_path}") from value
            return
    finally:
        stopped.set()
        if proc.poll() is None:
            _terminate_process_tree(proc)
        proc.stdout.close()
        proc.wait()
        reader.join(timeout=1)


def extract_video_frame(
    video_path: Path, frame_index: int = 0, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS
) -> Optional[Image.Image]:
    """Extracts a single frame from a video file as a PIL Image. Returns None on failure or timeout."""
    fd, tmp_name = tempfile.mkstemp(prefix="invokeai_frame_", suffix=".png")
    os.close(fd)
    try:
        result = _run_worker(["frame", str(video_path), str(frame_index), tmp_name], timeout)
        if result is None:
            return None
        with Image.open(tmp_name) as image:
            image.load()
        return image
    except Exception:
        return None
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def probe_video(
    video_path: Path, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS
) -> tuple[int, int, float, Optional[float]]:
    """Returns (width, height, duration_seconds, fps_or_none) for a video file.

    Raises FileNotFoundError if the file cannot be read — including when the decode
    times out, since a file we cannot probe within the bound is treated as unreadable.
    """
    result = _run_worker(["probe", str(video_path)], timeout)
    if result is None:
        raise FileNotFoundError(f"Unable to open video at {video_path}")
    try:
        width = int(result["width"])
        height = int(result["height"])
        duration = float(result["duration"])
        fps_raw = result.get("fps")
        fps: Optional[float] = float(fps_raw) if fps_raw else None
    except (KeyError, TypeError, ValueError) as e:
        raise FileNotFoundError(f"Unable to open video at {video_path}") from e
    return width, height, duration, fps


def decoder_frame_count(video_path: Path, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS) -> Optional[int]:
    """Returns the exact decoded frame count, or None if it cannot be determined in time.

    Preferred over a ``duration * fps`` estimate, which can overshoot by one on VFR
    uploads or containers with imprecise metadata; callers fall back to that estimate
    when this returns None.
    """
    result = _run_worker(["count", str(video_path)], timeout)
    if result is None:
        return None
    count = result.get("count")
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return None
    return int(count) if count > 0 else None
