"""Standalone decode worker for untrusted video files.

This script is executed as a short-lived child process by
``invokeai.app.util.video_thumbnails`` — never imported by the server at runtime — so
that a decoder hang on a crafted or malformed container can be bounded by a timeout and
killed. cv2 wheels have historically hung on certain codec/container combinations, and
a hung *thread* cannot be killed from Python, so process isolation is the only reliable
bound. It is run by file path (not ``-m``) and deliberately imports only the stdlib plus
imageio/PIL/cv2, so it starts quickly without pulling in the invokeai package or torch.

Protocol: ``python video_decode_worker.py <command> <args...>``. On success a single
JSON object is written to stdout and the exit code is 0; on any failure the exit code is
non-zero with a message on stderr.

Commands:
    probe <video_path>                     -> {"width", "height", "duration", "fps"}
    frame <video_path> <index> <out_path>  -> {"ok": true}; the frame is written to out_path as PNG
    count <video_path>                     -> {"count": <int or null>}
    stream <video_path>                    -> consecutive numpy arrays on stdout
"""

import io
import json
import math
import struct
import sys
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
from PIL import Image

# Upper bound on the decoded frame size we are willing to allocate (~8K video). A small
# crafted container can claim absurd dimensions in cheap metadata while the full frame
# is only allocated at decode time, so dimensions are checked before any decode.
# Keep in sync with MAX_VIDEO_FRAME_PIXELS in video_thumbnails.py (this script must not
# import the invokeai package).
MAX_FRAME_PIXELS = 64 * 1024 * 1024
WORKER_MEMORY_HEADROOM_BYTES = 1024 * 1024 * 1024


def _limit_worker_memory() -> None:
    """Bound allocations made after worker startup on Linux."""
    if sys.platform != "linux":
        return
    try:
        import os
        import resource

        pages = int(Path("/proc/self/statm").read_text().split()[0])
        limit = pages * os.sysconf("SC_PAGE_SIZE") + WORKER_MEMORY_HEADROOM_BYTES
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (OSError, ValueError):
        pass


def _validate_decoded_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Decoded frame must be RGB; got shape {frame.shape}")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0 or height * width > MAX_FRAME_PIXELS:
        raise ValueError(f"Decoded frame dimensions {width}x{height} exceed the maximum decodable size")


def _assert_decodable_dims(video_path: Path) -> None:
    """Refuses to decode frames from a video whose reported dimensions exceed the bound.

    If the dimensions cannot be probed, decoding is refused because the parent's frame
    record bound and PIL checks run only after the decoder has allocated the frame.
    """
    try:
        width, height, _duration, _fps = _probe(video_path)
    except Exception as error:
        raise ValueError(f"Unable to validate video dimensions for {video_path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Video reports invalid dimensions {width}x{height}")
    if width * height > MAX_FRAME_PIXELS:
        raise ValueError(f"Video dimensions {width}x{height} exceed the maximum decodable size")


def _extract_frame(video_path: Path, frame_index: int) -> Optional[Image.Image]:
    """Extracts a single frame from a video file as a PIL Image. Returns None on failure.

    Tries imageio's FFMPEG plugin first since it's the same encoder we use for output,
    then falls back to cv2 — uploaded videos with unusual codecs may need that path.
    """
    try:
        # iio.imread with index=N seeks to that frame directly. Returns RGB HxWxC uint8.
        frame = iio.imread(video_path, plugin="FFMPEG", index=frame_index)
        _validate_decoded_frame(frame)
        return Image.fromarray(frame)
    except Exception:
        pass

    try:
        import cv2  # local import so the imageio-only path doesn't pay the cv2 import cost

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            if frame_index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                return None
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            _validate_decoded_frame(frame_rgb)
            return Image.fromarray(frame_rgb)
        finally:
            capture.release()
    except Exception:
        return None


def _probe(video_path: Path) -> tuple[int, int, float, Optional[float]]:
    """Returns (width, height, duration_seconds, fps_or_none) for a video file.

    Tries imageio's FFMPEG plugin first; falls back to cv2.VideoCapture. Raises if
    neither backend can read the file.
    """
    try:
        meta = iio.immeta(video_path, plugin="FFMPEG")
        fps_raw = meta.get("fps")
        duration = float(meta.get("duration", 0.0)) if meta.get("duration") is not None else 0.0
        size = meta.get("size")
        if size is None:
            # Fall through to cv2 — imageio didn't give us dimensions.
            raise ValueError("imageio probe missing 'size'")
        width, height = int(size[0]), int(size[1])
        fps: Optional[float] = float(fps_raw) if fps_raw and fps_raw > 0 else None
        return width, height, duration, fps
    except Exception:
        pass

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Unable to open video at {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_raw = capture.get(cv2.CAP_PROP_FPS)
        fps_v2: Optional[float] = float(fps_raw) if fps_raw and fps_raw > 0 else None
        duration = (frame_count / fps_v2) if (fps_v2 and frame_count > 0) else 0.0
    finally:
        capture.release()
    return width, height, duration, fps_v2


def _count(video_path: Path) -> Optional[int]:
    """Return the exact decoded frame count, or None if neither backend can determine it.

    Tries imageio's improps first (works for a handful of codecs that expose nframes in
    container metadata). For libx264 streams imageio reports ``inf``, so we fall through
    to cv2's ``CAP_PROP_FRAME_COUNT`` which reads the actual packet count. Both sources
    are preferred over a ``duration * fps`` estimate, which can overshoot by one on VFR
    uploads or containers with imprecise metadata.
    """
    try:
        props = iio.improps(video_path, plugin="FFMPEG")
    except Exception:
        props = None
    shape = getattr(props, "shape", None) if props is not None else None
    if shape:
        n = shape[0]
        if not (isinstance(n, float) and not math.isfinite(n)):
            try:
                return int(n)
            except (TypeError, ValueError, OverflowError):
                pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return count if count > 0 else None
    except Exception:
        return None


def _emit_stream_frame(frame: np.ndarray) -> None:
    _validate_decoded_frame(frame)
    record = io.BytesIO()
    np.save(record, np.ascontiguousarray(frame), allow_pickle=False)
    payload = record.getvalue()
    sys.stdout.buffer.write(struct.pack(">Q", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _stream(video_path: Path) -> None:
    """Writes decoded frames as length-prefixed, non-pickled numpy records.

    Tries imageio's FFMPEG plugin first and falls back to cv2, mirroring
    ``_extract_frame``/``_probe``/``_count`` — a file accepted at upload via the cv2
    fallback must also stream in the frame-range and concat nodes. The fallback only
    engages when no frame has been emitted yet: a mid-stream decoder failure must
    surface as an error, not restart the stream from frame 0.
    """
    emitted = False
    try:
        for frame in iio.imiter(video_path, plugin="FFMPEG"):
            _emit_stream_frame(frame)
            emitted = True
    except Exception:
        if emitted:
            raise
    if emitted:
        return

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Unable to open video at {video_path}")
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            _emit_stream_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            emitted = True
    finally:
        capture.release()
    if not emitted:
        raise ValueError(f"No frames decoded from {video_path}")


def main(argv: list[str]) -> int:
    try:
        _limit_worker_memory()
        command = argv[1]
        if command == "stream":
            _assert_decodable_dims(Path(argv[2]))
            _stream(Path(argv[2]))
        elif command == "probe":
            width, height, duration, fps = _probe(Path(argv[2]))
            print(json.dumps({"width": width, "height": height, "duration": duration, "fps": fps}))
        elif command == "frame":
            _assert_decodable_dims(Path(argv[2]))
            image = _extract_frame(Path(argv[2]), int(argv[3]))
            if image is None:
                print("no frame decoded", file=sys.stderr)
                return 1
            image.save(argv[4], format="PNG")
            print(json.dumps({"ok": True}))
        elif command == "count":
            print(json.dumps({"count": _count(Path(argv[2]))}))
        else:
            print(f"unknown command {command!r}", file=sys.stderr)
            return 1
    except Exception as e:
        print(str(e) or repr(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
