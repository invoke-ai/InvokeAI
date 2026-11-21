"""PatchMatch Infill node behavior, with the native PatchMatch library replaced by a recording stand-in."""

import pytest
from PIL import Image

from invokeai.app.invocations import infill as infill_module
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.infill import InfillPatchMatchInvocation

FILL_COLOR = (200, 10, 10)
OPAQUE_COLOR = (10, 200, 10, 255)


@pytest.fixture
def patchmatch_inputs(monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    """Replaces PatchMatch with a solid fill and records the images it is asked to infill."""
    inputs: list[Image.Image] = []

    def fake_infill_patchmatch(image: Image.Image) -> Image.Image:
        inputs.append(image)
        return Image.new("RGB", image.size, FILL_COLOR)

    monkeypatch.setattr(infill_module, "infill_patchmatch", fake_infill_patchmatch)
    return inputs


def _left_opaque_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    image.paste(OPAQUE_COLOR, (0, 0, width // 2, height))
    return image


@pytest.mark.parametrize(("downscale", "expected_size"), [(2.0, (40, 24)), (4.0, (20, 12)), (1.0, (80, 48))])
def test_patchmatch_runs_on_the_downscaled_image(
    patchmatch_inputs: list[Image.Image], downscale: float, expected_size: tuple[int, int]
) -> None:
    node = InfillPatchMatchInvocation(image=ImageField(image_name="unused"), downscale=downscale)

    node.infill(_left_opaque_image(80, 48))

    assert [image.size for image in patchmatch_inputs] == [expected_size]


def test_patchmatch_result_is_restored_to_full_size_under_the_original_pixels(
    patchmatch_inputs: list[Image.Image],
) -> None:
    node = InfillPatchMatchInvocation(image=ImageField(image_name="unused"), downscale=2.0)

    result = node.infill(_left_opaque_image(80, 48))

    assert result.size == (80, 48)
    assert result.convert("RGB").getpixel((10, 20)) == OPAQUE_COLOR[:3]
    assert result.convert("RGB").getpixel((70, 20)) == FILL_COLOR
