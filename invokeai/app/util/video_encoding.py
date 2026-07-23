"""Shared MP4 writer construction for video-producing invocations.

All video nodes encode through this helper so the encoder settings stay in one
place. libx264 + yuv420p (imageio's defaults for the FFMPEG plugin) give
broadly-compatible browser playback. ``macro_block_size=1`` is load-bearing:
imageio's default of 16 makes ffmpeg silently *rescale* frames to the next
multiple of 16 (e.g. 1920x1080 -> 1920x1088), which desynchronizes the encoded
file from the dimensions recorded in the video DTO and breaks same-dimension
checks downstream (e.g. concatenating a trimmed clip with its source).

yuv420p requires even dimensions; callers validate that before encoding.
"""

from pathlib import Path

import imageio.v2 as iio2


def make_mp4_writer(path: Path | str, fps: float):
    """Returns an imageio FFMPEG writer that preserves frame dimensions exactly."""
    return iio2.get_writer(str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264", macro_block_size=1)
