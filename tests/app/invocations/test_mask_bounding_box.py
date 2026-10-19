from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.mask import GetMaskBoundingBoxInvocation


def _build_context(mask: Image.Image) -> MagicMock:
    context = MagicMock()
    context.images.get_pil.return_value = mask
    return context


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        (0, (2, 1, 5, 3)),
        (1, (1, 0, 6, 4)),
        # The margin is clamped to the image bounds, which for x_max / y_max is the image size.
        (10, (0, 0, 8, 6)),
    ],
)
def test_get_mask_bounding_box_max_coords_are_exclusive(margin: int, expected: tuple[int, int, int, int]) -> None:
    mask = Image.new("RGBA", (8, 6), (0, 0, 0, 255))
    # Mask pixels span x in [2, 4] and y in [1, 2].
    for x in range(2, 5):
        for y in range(1, 3):
            mask.putpixel((x, y), (255, 255, 255, 255))

    output = GetMaskBoundingBoxInvocation(mask=ImageField(image_name="mask"), margin=margin).invoke(
        _build_context(mask)
    )

    assert output.bounding_box.tuple() == expected
    # Cropping to the bounding box (as "Crop Image to Bounding Box" does) keeps every mask pixel.
    if margin == 0:
        assert mask.crop(output.bounding_box.tuple()).getcolors() == [(6, (255, 255, 255, 255))]


def test_get_mask_bounding_box_single_pixel() -> None:
    mask = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    mask.putpixel((1, 1), (255, 255, 255, 255))

    output = GetMaskBoundingBoxInvocation(mask=ImageField(image_name="mask")).invoke(_build_context(mask))

    assert output.bounding_box.tuple() == (1, 1, 2, 2)
