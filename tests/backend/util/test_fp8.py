"""Tests for the FP8 storage helpers: `get_model_compute_dtype`, and `peak_fp8_cast_transient_bytes` at the end.

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
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.util.fp8 import (
    FP8_COMPUTE_DTYPE_ATTR,
    get_model_compute_dtype,
    peak_fp8_cast_transient_bytes,
    set_fp8_compute_dtype,
)


@pytest.fixture(autouse=True)
def _device_holds_fp8():
    """Answer the storage probe True for every test here.

    These all drive the real cast through `_apply_fp8_layerwise_casting` on a `cuda` loader, and what
    they assert is what `get_model_compute_dtype` reports afterwards -- not whether a device can hold
    float8. The CUDA branch used to be answered True without asking, so this never had to be said;
    the probe is real now, and a runner with no driver rightly fails it.
    """
    with patch("invokeai.backend.model_manager.load.load_default._device_supports_fp8_storage", return_value=True):
        yield


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
        # A real config always declares a format, and the gate reads it: whether the loader for this
        # `(base, type, format)` implements the cast is part of the answer (`load/fp8_capability.py`).
        format=ModelFormat.Diffusers,
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
    """A model that reached us without the marker (e.g. an older cache entry, cast before the
    marker existed) must still resolve: the fp8 cast skips norm layers, so their dtype reveals the
    compute dtype."""
    model = _MiniUNet().to(torch.bfloat16)
    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    # Simulate the legacy, marker-less model.
    delattr(model, FP8_COMPUTE_DTYPE_ATTR)

    assert model.dtype == torch.float8_e4m3fn
    assert get_model_compute_dtype(model) == torch.bfloat16


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_cast_records_the_marker_itself():
    """The marker must be set by the cast itself, not by its callers — a caller that forgets it
    would fall through to the scan, and a caller that derives the dtype from an already-cast model
    would record float8."""
    model = _MiniUNet().to(torch.bfloat16)
    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

    assert getattr(model, FP8_COMPUTE_DTYPE_ATTR) == torch.bfloat16


@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_rejects_float8_as_compute_dtype(storage_dtype: torch.dtype):
    """A float8 compute dtype would be handed straight back by `get_model_compute_dtype`, silently
    reintroducing the crash. Recording one must fail loudly instead."""
    model = _MiniUNet().to(torch.float16)
    with pytest.raises(ValueError, match="storage-only dtype"):
        set_fp8_compute_dtype(model, storage_dtype)

    assert not hasattr(model, FP8_COMPUTE_DTYPE_ATTR)


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_double_cast_is_a_noop():
    """The cast is not idempotent: a second pass would derive the compute dtype from the already-fp8
    first parameter. An already-marked model must be left alone."""
    loader = ModelLoader.__new__(ModelLoader)
    loader._torch_device = torch.device("cuda")
    loader._torch_dtype = torch.float16
    loader._logger = getLogger("test")
    config = SimpleNamespace(
        type=ModelType.Main,
        base=BaseModelType.StableDiffusionXL,
        # A real config always declares a format, and the gate reads it: whether the loader for this
        # `(base, type, format)` implements the cast is part of the answer (`load/fp8_capability.py`).
        format=ModelFormat.Diffusers,
        name="test",
        default_settings=SimpleNamespace(fp8_storage=True),
    )

    model = loader._apply_fp8_layerwise_casting(_MiniUNet().to(torch.bfloat16), config)
    model = loader._apply_fp8_layerwise_casting(model, config)

    assert get_model_compute_dtype(model) == torch.bfloat16
    # The skipped norm layer is still in the compute dtype — a second cast would have taken it and
    # the hooks would have been registered twice.
    assert model.norm1.weight.dtype == torch.bfloat16
    assert len(model.linear._forward_pre_hooks) == 1


class _TwoLayers(torch.nn.Module):
    """Two castable layers of different sizes, run one after the other, and a norm FP8 Storage leaves alone."""

    def __init__(self):
        super().__init__()
        self.large = torch.nn.Linear(64, 64)
        self.small = torch.nn.Linear(64, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.small(self.large(x)))


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_the_fp8_cast_transient_is_the_largest_layer_fp8_storage_widens_in_a_forward():
    """Measured on the loader's real cast: each layer's params are widened while its own forward runs, and put back
    before the next layer widens, so what is widened at once peaks at the largest layer's, weight and bias. In fp32,
    which CPU tests run in."""
    model = _TwoLayers()
    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float32)
    widened_at_once = []

    def record(module: torch.nn.Module, args: object) -> None:
        widened_at_once.append(
            sum(
                p.nbytes for layer in (model.large, model.small) for p in layer.parameters() if p.dtype == torch.float32
            )
        )

    # Registered after the cast's hooks, so they see each layer's params as its forward runs on them.
    for layer in (model.large, model.small):
        layer.register_forward_pre_hook(record)
    model(torch.randn(1, 64))

    assert widened_at_once == [(64 * 64 + 64) * 4, (64 * 8 + 8) * 4]
    assert peak_fp8_cast_transient_bytes(model, torch.float32) == max(widened_at_once)


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
@pytest.mark.parametrize("scaled, copies", [(False, 1), (True, 2)], ids=["raw", "scaled"])
def test_a_kept_fp8_checkpoint_weight_is_widened_per_forward_and_a_scaled_one_twice(scaled: bool, copies: int):
    """`CustomLinear` dequantizes a kept fp8 weight per forward; `dequantize_weight` holds the widened weight and its
    product with the `weight_scale` at once. Only the weight is kept in fp8, so the bias costs nothing."""
    layer = torch.nn.Linear(64, 32, device="meta")
    layer.weight = torch.nn.Parameter(
        torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta"), requires_grad=False
    )
    if scaled:
        layer.register_buffer("weight_scale", torch.ones((), device="meta"), persistent=False)

    assert peak_fp8_cast_transient_bytes(torch.nn.Sequential(layer), torch.bfloat16) == copies * 32 * 64 * 2


class _PackedTensor(torch.Tensor):
    """A quantized payload kept in a tensor subclass, as SDNQ and GGUF keep theirs."""


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_a_model_without_plain_fp8_weights_has_no_fp8_cast_transient():
    """Dense weights are what the forward runs on, and a packed payload is its own scheme's to dequantize."""
    dense = torch.nn.Linear(64, 64, device="meta", dtype=torch.bfloat16)
    packed = torch.nn.Linear(64, 64, device="meta")
    packed.weight = torch.nn.Parameter(
        torch.empty(64, 64, dtype=torch.float8_e4m3fn, device="meta").as_subclass(_PackedTensor), requires_grad=False
    )

    assert peak_fp8_cast_transient_bytes(torch.nn.Sequential(dense, packed), torch.bfloat16) == 0
