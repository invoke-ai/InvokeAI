"""Tests for `ModelLoader` FP8 helpers.

Covers:
- `_should_use_fp8` excludes ControlLoRA (the LoRA loader never runs the layerwise
  casting helper, and a LoRA isn't a standalone forward module — so a persisted
  `fp8_storage=true` must be a no-op).
- `_wrap_forward_with_fp8_cast` is exception-safe: if forward raises, the storage-dtype
  cast still runs and parameters end up in fp8 (previously the post-hook path silently
  left params in compute dtype, defeating the storage savings).
"""

from logging import getLogger
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _make_loader(device: str = "cuda") -> ModelLoader:
    """Build a ModelLoader without going through dependency injection.

    `_should_use_fp8` and `_wrap_forward_with_fp8_cast` only depend on `_torch_device`,
    so we instantiate via __new__ and set the minimum state directly.
    """
    loader = ModelLoader.__new__(ModelLoader)
    loader._torch_device = torch.device(device)
    loader._torch_dtype = torch.float16
    loader._logger = getLogger("test")
    return loader


def _make_config(model_type: ModelType, fp8: bool, base: BaseModelType = BaseModelType.Flux):
    return SimpleNamespace(
        type=model_type,
        base=base,
        name="test",
        default_settings=SimpleNamespace(fp8_storage=fp8),
    )


def test_should_use_fp8_excludes_control_lora():
    """ControlLoRA gets the FP8 toggle in the UI history but the LoRA loader never applies
    layerwise casting (the model isn't run as a standalone forward pass — it patches into a
    base model). The loader must silently ignore a persisted `fp8_storage=true` to avoid
    misleading users who toggled it under a prior version.
    """
    loader = _make_loader(device="cuda")
    with patch("torch.cuda.is_available", return_value=True):
        assert loader._should_use_fp8(_make_config(ModelType.ControlLoRa, fp8=True)) is False


def test_should_use_fp8_excludes_lora():
    loader = _make_loader(device="cuda")
    assert loader._should_use_fp8(_make_config(ModelType.LoRA, fp8=True)) is False


def test_should_use_fp8_returns_true_for_main_with_fp8():
    loader = _make_loader(device="cuda")
    assert loader._should_use_fp8(_make_config(ModelType.Main, fp8=True)) is True


def test_should_use_fp8_returns_false_for_main_without_fp8():
    loader = _make_loader(device="cuda")
    assert loader._should_use_fp8(_make_config(ModelType.Main, fp8=False)) is False


def test_should_use_fp8_returns_false_on_cpu():
    loader = _make_loader(device="cpu")
    assert loader._should_use_fp8(_make_config(ModelType.Main, fp8=True)) is False


class _RaisingModule(torch.nn.Module):
    """A module whose forward unconditionally raises — used to test that the FP8 wrapper's
    storage-dtype cleanup runs even when forward fails."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")


def _fp8_supported() -> bool:
    return hasattr(torch, "float8_e4m3fn")


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_wrap_forward_restores_storage_dtype_on_exception():
    """When forward raises, params must be returned to storage dtype. Otherwise FP8 storage
    savings silently revert to fp16/bf16 and the cache's size accounting becomes stale.
    """
    storage_dtype = torch.float8_e4m3fn
    compute_dtype = torch.bfloat16

    module = _RaisingModule()
    for p in module.parameters(recurse=False):
        p.data = p.data.to(storage_dtype)

    ModelLoader._wrap_forward_with_fp8_cast(module, storage_dtype, compute_dtype)

    # Sanity: params start in storage dtype.
    assert module.weight.dtype == storage_dtype
    assert module.bias.dtype == storage_dtype

    with pytest.raises(RuntimeError, match="boom"):
        module(torch.zeros(4, dtype=compute_dtype))

    # Critical assertion: cleanup ran despite the exception.
    assert module.weight.dtype == storage_dtype
    assert module.bias.dtype == storage_dtype


@pytest.mark.skipif(not _fp8_supported(), reason="torch.float8_e4m3fn not available")
def test_wrap_forward_casts_to_compute_then_back_on_success():
    """Happy-path sanity check: params are in compute dtype during forward, storage dtype after."""
    storage_dtype = torch.float8_e4m3fn
    compute_dtype = torch.bfloat16

    seen_dtypes: list[torch.dtype] = []

    class _CaptureModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen_dtypes.append(self.weight.dtype)
            return x + self.weight

    module = _CaptureModule()
    for p in module.parameters(recurse=False):
        p.data = p.data.to(storage_dtype)

    ModelLoader._wrap_forward_with_fp8_cast(module, storage_dtype, compute_dtype)

    module(torch.zeros(4, dtype=compute_dtype))

    assert seen_dtypes == [compute_dtype]
    assert module.weight.dtype == storage_dtype


def test_apply_fp8_to_nn_module_uses_wrapper():
    """`_apply_fp8_to_nn_module` should delegate cleanup to `_wrap_forward_with_fp8_cast`
    rather than rely on the pre-hook/post-hook pair (which is not exception-safe).
    """
    module = torch.nn.Linear(4, 4)
    with patch.object(ModelLoader, "_wrap_forward_with_fp8_cast") as mock_wrap:
        ModelLoader._apply_fp8_to_nn_module(module, torch.float16, torch.float32)
    mock_wrap.assert_called_once_with(module, torch.float16, torch.float32)
