"""`precise_signed_decode_raster`: the unit-range raster every [-1, 1] decoder node stores."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from invokeai.app.invocations.vae.precise_io import precise_signed_decode_raster


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
