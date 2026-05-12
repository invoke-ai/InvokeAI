import os
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image


def get_video_thumbnail_name(video_name: str) -> str:
    """Given a video file name (e.g. <uuid>.mp4), returns the matching thumbnail name (e.g. <uuid>.webp)."""
    return os.path.splitext(video_name)[0] + ".webp"


def extract_video_frame(video_path: Path, frame_index: int = 0) -> Optional[Image.Image]:
    """Extracts a single frame from a video file as a PIL Image. Returns None on failure."""
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


def probe_video(video_path: Path) -> tuple[int, int, float, Optional[float]]:
    """Returns (width, height, duration_seconds, fps_or_none) for a video file.

    Raises FileNotFoundError when the file cannot be opened.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Unable to open video at {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_raw = capture.get(cv2.CAP_PROP_FPS)
        fps: Optional[float] = float(fps_raw) if fps_raw and fps_raw > 0 else None
        duration = (frame_count / fps) if (fps and frame_count > 0) else 0.0
    finally:
        capture.release()
    return width, height, duration, fps
