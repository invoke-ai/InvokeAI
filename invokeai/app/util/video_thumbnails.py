"""Video frame/probe helpers used by the video file store.

The primary backend is imageio's FFMPEG plugin (the same one ``wan_l2v`` uses
to *encode* output MP4s — so reading our own output is guaranteed to work).
We fall back to ``cv2.VideoCapture`` only if imageio fails; cv2 wheels have
historically hung on certain codec/container combinations, so we never rely
on it as the primary path.
"""

import os
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
from PIL import Image


def get_video_thumbnail_name(video_name: str) -> str:
    """Given a video file name (e.g. <uuid>.mp4), returns the matching thumbnail name (e.g. <uuid>.webp)."""
    return os.path.splitext(video_name)[0] + ".webp"


def extract_video_frame(video_path: Path, frame_index: int = 0) -> Optional[Image.Image]:
    """Extracts a single frame from a video file as a PIL Image. Returns None on failure.

    Tries imageio's FFMPEG plugin first since it's the same encoder we use for
    output, then falls back to cv2 (with a controlled context that can't hang
    silently — at worst it raises and we return None).
    """
    try:
        # iio.imread with index=N seeks to that frame directly. Returns RGB HxWxC uint8.
        frame = iio.imread(video_path, plugin="FFMPEG", index=frame_index)
        return Image.fromarray(frame)
    except Exception:
        pass

    # Fallback: cv2.VideoCapture. Only used if imageio couldn't decode the file
    # — uploaded videos with unusual codecs may need this path.
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
            return Image.fromarray(frame_rgb)
        finally:
            capture.release()
    except Exception:
        return None


def probe_video(video_path: Path) -> tuple[int, int, float, Optional[float]]:
    """Returns (width, height, duration_seconds, fps_or_none) for a video file.

    Tries imageio's FFMPEG plugin first; falls back to cv2.VideoCapture. Raises
    FileNotFoundError if neither backend can read the file.
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

    try:
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
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(f"Unable to open video at {video_path}") from e
