import numpy as np
import pytest
import torch

from invokeai.backend.image_util.raster import AlphaMode, ColorEncoding, RasterChannels, ValueRange
from invokeai.backend.image_util.raster.convert import decoded_unit_raster


def _batch(dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5) / 16 - 1.5).to(dtype)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.float16, np.float16), (torch.float32, np.float32), (torch.bfloat16, np.float32)],
)
def test_the_first_sample_is_kept_unclamped_at_the_decode_precision(dtype, expected) -> None:
    batch = _batch(dtype)

    for tensor in (batch, batch[0]):  # NCHW and CHW
        raster = decoded_unit_raster(tensor)

        assert raster is not None
        assert raster.dtype == expected
        assert (raster.height, raster.width) == (4, 5)
        np.testing.assert_array_equal(raster.view(), batch[0].float().permute(1, 2, 0).numpy().astype(expected))
        assert raster.view().min() < 0.0 and raster.view().max() > 1.0
        descriptor = raster.descriptor
        assert (descriptor.channels, descriptor.color, descriptor.value_range, descriptor.alpha) == (
            RasterChannels.RGB,
            ColorEncoding.SRGB,
            ValueRange.UNIT,
            AlphaMode.NONE,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_sample_with_any_non_finite_value_gives_none(bad: float) -> None:
    batch = _batch(torch.float32)
    batch[0, 2, 3, 4] = bad  # one value in the last channel

    assert decoded_unit_raster(batch) is None
    # Only the first sample counts: a bad second sample does not stop the first from being saved.
    other = _batch(torch.float32)
    other[1, 0, 0, 0] = bad
    assert decoded_unit_raster(other) is not None
