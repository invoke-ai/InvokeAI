"""Tests for `get_model_compute_dtype`.

Regression coverage for the SDXL + fp8_storage crash:

    NotImplementedError: "pow_cuda" not implemented for 'Float8_e4m3fn'

The legacy SD/SDXL denoise path derived every tensor dtype (latents, noise, conditioning,
control images, LoRA patch weights) from `unet.dtype`. With fp8 layerwise casting the UNet's
first parameter is `float8_e4m3fn`, so the latents were created in a storage-only dtype and the
first bit of scheduler math (`sigma ** 2` in `add_noise`) blew up before the UNet was ever
called.
"""

from logging import getLogger
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.fp8 import (
    FP8_COMPUTE_DTYPE_ATTR,
    get_model_compute_dtype,
    set_fp8_compute_dtype,
)


def _fp8_supported() -> bool:
    return hasattr(torch, "float8_e4m3fn")


class _MiniUNet(torch.nn.Module):
    """Stand-in for a diffusers model: castable layers plus a skipped norm, and a `dtype`
    property that reports the first parameter's dtype (what `ModelMixin.dtype` does)."""

    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.norm1 = torch.nn.LayerNorm(4)
        self.linear = torch.nn.Linear(4, 4)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def test_returns_model_dtype_for_normal_model():
    model = _MiniUNet().to(torch.float16)
    assert get_model_compute_dtype(model) == torch.float16


def test_returns_marked_compute_dtype():
    model = _MiniUNet().to(torch.float16)
    set_fp8_compute_dtype(model, torch.bfloat16)
    assert get_model_compute_dtype(model) == torch.bfloat16


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_returns_compute_dtype_after_real_fp8_cast():
    """End-to-end over the loader's real casting path: `model.dtype` reports the float8 storage
    dtype, but the resolver must hand back the compute dtype so downstream tensors stay usable."""
    loader = ModelLoader.__new__(ModelLoader)
    loader._torch_device = torch.device("cuda")
    loader._torch_dtype = torch.float16
    loader._logger = getLogger("test")
    config = SimpleNamespace(
        type=ModelType.Main,
        base=BaseModelType.StableDiffusionXL,
        name="test",
        default_settings=SimpleNamespace(fp8_storage=True),
    )

    model = loader._apply_fp8_layerwise_casting(_MiniUNet().to(torch.float16), config)

    # Precondition: the naive `model.dtype` read is the storage dtype — this is what crashed.
    assert model.dtype == torch.float8_e4m3fn
    assert get_model_compute_dtype(model) == torch.float16

    # And the resolved dtype supports the arithmetic the scheduler does on it.
    sigma = torch.tensor([1.5], dtype=get_model_compute_dtype(model))
    assert torch.isfinite(1 / ((sigma**2 + 1) ** 0.5)).all()


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_falls_back_to_scan_when_marker_missing():
    """A model that reached us without the marker (e.g. an older cache entry) must still resolve:
    the fp8 cast skips norm layers, so their dtype reveals the compute dtype."""
    model = _MiniUNet().to(torch.bfloat16)
    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    assert not hasattr(model, FP8_COMPUTE_DTYPE_ATTR)

    assert model.dtype == torch.float8_e4m3fn
    assert get_model_compute_dtype(model) == torch.bfloat16
