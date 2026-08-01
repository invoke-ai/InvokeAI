"""The faded mask must come out byte-for-byte the same as the original implementation.

`ExpandMaskWithFadeInvocation` feeds canvas compositing, so a one-level change in the
feather ramp shows up as a seam in a paste-back. This test pins the output against a
transcription of the previous implementation rather than against stored fixtures, so it
keeps working if the fixtures are ever regenerated.
"""

import cv2
import numpy
import pytest
from PIL import Image

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.image import ExpandMaskWithFadeInvocation


def _previous_implementation(pil_mask: Image.Image, threshold: int, fade_size_px: int) -> Image.Image:
    """The pre-optimisation body of ExpandMaskWithFadeInvocation.invoke, verbatim."""
    if fade_size_px == 0:
        return pil_mask

    np_mask = numpy.array(pil_mask)
    np_mask = numpy.where(np_mask > threshold, 255, 0).astype(numpy.uint8)
    black_mask = (np_mask == 0).astype(numpy.uint8)
    bg_mask = 1 - black_mask
    dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
    d_norm = numpy.clip(dist / fade_size_px, 0, 1)
    x_control = numpy.array([0.0, 1.0 / fade_size_px, 0.2, 0.8, 1.0])
    y_control = numpy.array([0.0, 0.0, 0.2, 0.9, 1.0])
    poly = numpy.poly1d(numpy.polyfit(x_control, y_control, 3))
    feather = poly(d_norm)
    feather = numpy.where(d_norm >= 1.0, 1.0, feather)
    feather = numpy.clip(feather, 0, 1)
    np_result = numpy.where(black_mask == 1, 0, (feather * 255).astype(numpy.uint8))
    return Image.fromarray(np_result.astype(numpy.uint8), mode="L")


class _Saved:
    def __init__(self, image: Image.Image) -> None:
        self.image_name = "mask.png"
        self.width = image.width
        self.height = image.height


class _Images:
    def __init__(self, source: Image.Image) -> None:
        self.source = source
        self.saved: Image.Image | None = None

    def get_pil(self, image_name: str, mode=None) -> Image.Image:
        image = self.source
        if mode and mode != image.mode:
            image = image.convert(mode)
        return image

    def save(self, image: Image.Image, image_category=None, **kwargs) -> _Saved:
        self.saved = image
        return _Saved(image)


class _Context:
    def __init__(self, images: _Images) -> None:
        self.images = images


def _mask(kind: str, height: int, width: int) -> Image.Image:
    rng = numpy.random.default_rng(0)
    array = numpy.zeros((height, width), dtype=numpy.uint8)
    if kind == "blobs":
        for _ in range(3):
            cx = int(rng.integers(width // 4, 3 * width // 4))
            cy = int(rng.integers(height // 4, 3 * height // 4))
            cv2.ellipse(array, (cx, cy), (width // 6, height // 6), 0, 0, 360, 255, -1)
    elif kind == "band":
        array[:, : width // 3] = 255
    elif kind == "empty":
        pass
    elif kind == "full":
        array[:] = 255
    elif kind == "hairline":
        cv2.line(array, (0, height // 2), (width - 1, height // 2), 255, 1)
    elif kind == "grey_ramp":
        array[:] = numpy.linspace(0, 255, width, dtype=numpy.uint8)[None, :]
    else:
        raise AssertionError(kind)
    return Image.fromarray(array, mode="L")


@pytest.mark.parametrize("kind", ["blobs", "band", "empty", "full", "hairline", "grey_ramp"])
@pytest.mark.parametrize("size", [(64, 64), (129, 97)])
@pytest.mark.parametrize("threshold,fade_size_px", [(0, 32), (0, 1), (127, 8), (254, 64), (0, 0)])
def test_matches_previous_implementation(kind, size, threshold, fade_size_px):
    height, width = size
    source = _mask(kind, height, width)

    images = _Images(source)
    node = ExpandMaskWithFadeInvocation(
        mask=ImageField(image_name="mask.png"), threshold=threshold, fade_size_px=fade_size_px
    )
    output = node.invoke(_Context(images))

    assert images.saved is not None
    assert images.saved.mode == "L"
    assert images.saved.size == source.size
    assert (output.width, output.height) == source.size

    expected = _previous_implementation(source, threshold, fade_size_px)
    assert numpy.array_equal(numpy.array(images.saved), numpy.array(expected))
