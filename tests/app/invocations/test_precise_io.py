"""Shared precise-image policy of the VAE nodes: decode rasters and unit tensors of stored assets."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from invokeai.app.invocations.vae.precise_io import (
    float_output_fits,
    float_output_raster,
    load_precise_unit_tensor,
    precise_signed_decode_raster,
)
from invokeai.backend.image_util.raster import AlphaMode, RasterChannels


def _context(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(images=SimpleNamespace(precise_storage_enabled=enabled), logger=MagicMock())


def test_only_the_first_sample_is_mapped_and_the_input_is_left_alone() -> None:
    decoded = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(0)) * 3  # beyond [-1, 1]
    before = decoded.clone()

    raster = precise_signed_decode_raster(_context(), decoded)

    expected = ((before[0].double() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(raster.view(), expected, rtol=0, atol=1e-6)
    assert raster.view().min() < 0.0 and raster.view().max() > 1.0
    assert torch.equal(decoded, before)  # the legacy quantizer still reads it


def test_a_bf16_output_is_widened_before_it_is_mapped() -> None:
    decoded = (torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(1)) * 2 - 1).to(torch.bfloat16)

    raster = precise_signed_decode_raster(_context(), decoded)

    expected = ((decoded[0].double() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(raster.view(), expected.astype(np.float32))
    in_bf16 = ((decoded[0] + 1.0) / 2).float().permute(1, 2, 0).numpy()
    assert (raster.view() != in_bf16).any()  # otherwise the case could not tell the two orders apart


def test_disabled_returns_none_without_reading_the_tensor() -> None:
    # A meta tensor cannot be copied to the host, so any work before the gate would raise.
    assert precise_signed_decode_raster(_context(enabled=False), torch.empty(1, 3, 2, 2, device="meta")) is None


def test_alpha_maps_a_four_channel_sample_to_a_straight_rgba_raster() -> None:
    decoded = torch.rand(1, 4, 3, 5, generator=torch.Generator().manual_seed(2)) * 3 - 1.5

    raster = precise_signed_decode_raster(_context(), decoded, alpha=True)

    expected = ((decoded[0].double() + 1.0) / 2).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(raster.view(), expected, rtol=0, atol=1e-6)
    assert (raster.descriptor.channels, raster.descriptor.alpha) == (RasterChannels.RGBA, AlphaMode.STRAIGHT)


def test_no_precise_asset_gives_no_unit_tensor(precise_world) -> None:
    name = precise_world.store_png(np.zeros((8, 8, 3), np.uint8), "RGB")

    assert load_precise_unit_tensor(precise_world.context, name) is None


@pytest.mark.parametrize("alpha", [False, True])
def test_a_precise_asset_gives_its_unclamped_unit_tensor(precise_world, alpha: bool) -> None:
    array = np.random.default_rng(3).uniform(-0.4, 1.4, size=(8, 12, 4)).astype(np.float32)
    name = precise_world.store_precise(array, RasterChannels.RGBA)

    tensor = load_precise_unit_tensor(precise_world.context, name, alpha=alpha)

    expected = array.transpose(2, 0, 1)
    np.testing.assert_array_equal(tensor.numpy(), expected if alpha else expected[:3])


@pytest.mark.parametrize(
    ("channels", "expected", "alpha"),
    [
        (1, RasterChannels.L, AlphaMode.NONE),
        (3, RasterChannels.RGB, AlphaMode.NONE),
        (4, RasterChannels.RGBA, AlphaMode.STRAIGHT),
    ],
)
def test_a_float_output_becomes_a_zero_copy_raster(channels, expected, alpha) -> None:
    buffer = np.random.default_rng(0).random((600, 7, channels), dtype=np.float32) * 3 - 1  # past 0..1, > 1 row block

    raster = float_output_raster(_context(), buffer)

    assert (raster.descriptor.channels, raster.descriptor.alpha) == (expected, alpha)
    assert np.shares_memory(raster.view(), buffer)
    np.testing.assert_array_equal(raster.view(), buffer)


@pytest.mark.parametrize("bad_row", [0, 599])
def test_a_non_finite_float_output_gives_no_raster_and_a_warning(bad_row) -> None:
    buffer = np.zeros((600, 4, 3), dtype=np.float32)
    buffer[bad_row, 2, 1] = np.inf
    context = _context()

    assert float_output_raster(context, buffer) is None
    assert "not finite" in context.logger.warning.call_args.args[0]


def test_a_float_output_beyond_the_asset_bounds_is_refused_before_allocation(monkeypatch) -> None:
    monkeypatch.setattr("invokeai.backend.image_util.raster.precise_asset.MAX_PIXEL_BYTES", 4 * 3 * 100)
    context = _context()

    assert float_output_fits(context, 10, 10, 3)
    context.logger.warning.assert_not_called()
    assert not float_output_fits(context, 10, 11, 3)
    context.logger.warning.assert_called_once()
    assert not float_output_fits(_context(enabled=False), 1, 1, 3)
