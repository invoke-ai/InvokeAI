"""Drop the FFmpeg wrapper DLL requirement from opencv-python's ``setup.py`` for a Windows ARM64 build.

OpenCV ships its ``opencv_videoio_ffmpeg*.dll`` wrapper only for x86/x64, yet ``setup.py`` lists it as a
required CMake output on every Windows build and raises ``Not found`` when it is absent. The ARM64 wheel is
built with ``-DWITH_FFMPEG=OFF`` (video decode falls back to Media Foundation), so the entry has to go.

Usage: patch_opencv_setup.py path/to/opencv-python/setup.py
"""

import sys
from pathlib import Path

FFMPEG_DLL_ENTRY = '[r"bin/opencv_videoio_ffmpeg\\d{4}%s\\.dll" % ("_64" if is64 else "")]'


def patch(source: str) -> str:
    if FFMPEG_DLL_ENTRY not in source:
        raise ValueError("opencv-python setup.py no longer contains the ffmpeg DLL entry; update this script")
    return source.replace(FFMPEG_DLL_ENTRY, "[]", 1)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1])
    path.write_text(patch(path.read_text()))
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
