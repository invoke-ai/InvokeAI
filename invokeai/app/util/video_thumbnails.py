"""Video frame/probe helpers used by the video file store, upload router, and video nodes.

Decoding runs in a short-lived child process (``video_decode_worker.py``) with a hard
timeout. Files reaching these helpers are user uploads, and both decode backends can
hang indefinitely on crafted or malformed containers (cv2 wheels historically, ffmpeg in
degenerate cases). An in-process hang would tie up the FastAPI request worker that
called us — repeated crafted uploads could exhaust the worker pool — and a hung thread
cannot be killed from Python, so process isolation is the only reliable bound.
``subprocess.run`` kills the child when the timeout expires, so a hostile file costs at
most ``timeout`` seconds and cannot leak a stuck process.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from PIL import Image

# Generous — a healthy decode of a single frame or of container metadata takes well
# under a second even for large files. The timeout exists to bound adversarial or hung
# decodes, not to police slow ones.
VIDEO_DECODE_TIMEOUT_SECONDS = 30.0

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


def _run_worker(args: list[str], timeout: float) -> Optional[dict[str, Any]]:
    """Runs the decode worker; returns its parsed JSON output, or None on failure or timeout."""
    try:
        proc = subprocess.run(_worker_command(*args), capture_output=True, text=True, timeout=timeout)
    except Exception:
        # TimeoutExpired (child already killed by subprocess.run) or a spawn failure.
        return None
    if proc.returncode != 0:
        return None
    try:
        result = json.loads(proc.stdout)
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


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
