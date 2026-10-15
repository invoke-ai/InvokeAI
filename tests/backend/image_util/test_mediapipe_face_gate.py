"""MediaPipe is absent on Windows ARM64 (no wheel carries its legacy `solutions` API there).

The face preprocessor and the Face Tools nodes must still import at startup -- every node module is
imported by `load_all_modules()` -- and fail with a message that names the platform when used.
"""

import platform
import sys

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.facetools import generate_face_box_mask
from invokeai.backend.image_util.mediapipe_face import MEDIAPIPE_UNAVAILABLE_MESSAGE, detect_faces


@pytest.fixture
def no_mediapipe(monkeypatch):
    """Make every `import mediapipe...` fail, including modules already cached from earlier tests."""
    for name in [m for m in sys.modules if m == "mediapipe" or m.startswith("mediapipe.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "invokeai.backend.image_util.mediapipe_face.mediapipe_face_common", raising=False)
    monkeypatch.setitem(sys.modules, "mediapipe", None)


def _image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


def _as_windows_arm64(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")


def test_detect_faces_names_the_platform_on_windows_arm64(no_mediapipe, monkeypatch):
    _as_windows_arm64(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        detect_faces(_image())
    assert str(excinfo.value) == MEDIAPIPE_UNAVAILABLE_MESSAGE


def test_face_box_mask_names_the_platform_on_windows_arm64(no_mediapipe, monkeypatch):
    _as_windows_arm64(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        generate_face_box_mask(context=None, minimum_confidence=0.5, x_offset=0, y_offset=0, pil_image=_image())  # type: ignore[arg-type]
    assert str(excinfo.value) == MEDIAPIPE_UNAVAILABLE_MESSAGE


def test_import_failure_elsewhere_carries_the_original_error(no_mediapipe, monkeypatch):
    # On platforms that do have mediapipe, an import failure is a broken install, not the platform: the
    # user must see the real cause rather than the Windows ARM64 explanation.
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match=r"MediaPipe failed to import.*mediapipe") as excinfo:
        detect_faces(_image())
    assert "Windows ARM64" not in str(excinfo.value)
