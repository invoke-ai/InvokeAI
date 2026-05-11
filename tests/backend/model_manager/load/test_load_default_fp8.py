"""Tests for `ModelLoader` FP8 helpers.

Covers:
- `_should_use_fp8` excludes ControlLoRA (the LoRA loader never runs the layerwise
  casting helper, and a LoRA isn't a standalone forward module — so a persisted
  `fp8_storage=true` must be a no-op).
- `_wrap_forward_with_fp8_cast` uses pre/post hooks with `always_call=True`, so it is
  exception-safe AND survives `apply_custom_layers_to_model`'s instance swap. Without
  hooks, an instance-level `forward` override would be carried into the new CustomLinear
  via the shared `__dict__` and silently bypass `CustomLinear.forward` — breaking LoRA
  patch dispatch for FP8 checkpoint models.
- `_apply_fp8_to_nn_module` skips precision-sensitive layers (norm, pos_embed, etc.)
  so FLUX RMSNorm.scale and friends aren't crushed to FP8.
"""

from logging import getLogger
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
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
    """`_apply_fp8_to_nn_module` should delegate per-module wrapping to
    `_wrap_forward_with_fp8_cast`, which encapsulates the hook registration.
    """
    module = torch.nn.Linear(4, 4)
    with patch.object(ModelLoader, "_wrap_forward_with_fp8_cast") as mock_wrap:
        ModelLoader._apply_fp8_to_nn_module(module, torch.float16, torch.float32)
    mock_wrap.assert_called_once_with(module, torch.float16, torch.float32)


def test_apply_fp8_to_nn_module_skips_norm_modules():
    """Modules whose path matches `norm` must not be cast — diffusers' `enable_layerwise_casting`
    does the same. FLUX RMSNorm.scale is the canonical example: a tiny learned scalar that
    breaks badly in FP8.
    """

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(4)
            self.linear = torch.nn.Linear(4, 4)

    storage_dtype = torch.float16
    compute_dtype = torch.float32
    model = _Model()
    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype, compute_dtype)

    # Linear params get cast to storage dtype.
    assert model.linear.weight.dtype == storage_dtype
    # Norm params stay in compute dtype — they must not be cast.
    assert model.norm1.weight.dtype == compute_dtype
    assert model.norm1.bias.dtype == compute_dtype


def test_apply_fp8_to_nn_module_skips_pos_embed_and_proj_in_out():
    """Position embeddings and the in/out projection of transformer blocks are also on the
    diffusers default skip list — they're precision-sensitive.
    """

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = torch.nn.Linear(4, 4)
            self.proj_in = torch.nn.Linear(4, 4)
            self.proj_out = torch.nn.Linear(4, 4)
            self.attn = torch.nn.Linear(4, 4)

    storage_dtype = torch.float16
    compute_dtype = torch.float32
    model = _Model()
    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype, compute_dtype)

    assert model.attn.weight.dtype == storage_dtype
    assert model.pos_embed.weight.dtype == compute_dtype
    assert model.proj_in.weight.dtype == compute_dtype
    assert model.proj_out.weight.dtype == compute_dtype


def test_apply_fp8_to_nn_module_skips_unsupported_layer_types():
    """Only the layer classes in `_FP8_SUPPORTED_PYTORCH_LAYERS` are cast — matches diffusers'
    behavior. A custom RMSNorm-style module with a raw Parameter must be left alone, otherwise
    its learned scalar gets clobbered.
    """

    class _ScaleModule(torch.nn.Module):
        """Mimics FLUX RMSNorm — a tiny learned scalar that must not be cast to FP8."""

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = _ScaleModule()
            self.linear = torch.nn.Linear(4, 4)

    storage_dtype = torch.float16
    compute_dtype = torch.float32
    model = _Model()
    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype, compute_dtype)

    assert model.linear.weight.dtype == storage_dtype
    # Critical: the RMS-style scalar lives on a custom module type, not in the supported list.
    assert model.rms.scale.dtype == compute_dtype


def test_wrap_forward_reaches_custom_linear_after_apply_custom_layers():
    """Production order: `_load_model` applies FP8 wrapping, THEN `ModelCache.put()` calls
    `apply_custom_layers_to_model` which constructs a NEW `CustomLinear` object via
    `CustomLinear.__new__` and points its `__dict__` at the original `Linear.__dict__`
    (see `wrap_custom_layer`). The new object is installed on the parent in place of the
    original Linear.

    An instance-level `forward` override would be carried into the new CustomLinear via the
    shared dict but would close over the OLD Linear instance — so calls to the new
    CustomLinear would silently route to `Linear.forward(old_instance, ...)` and bypass
    `CustomLinear.forward`, where LoRA/ControlLoRA patches are applied. This is the bug a
    reviewer reproduced on a fresh worktree.

    Hooks fix this because `nn.Module._call_impl` dispatches them with the *actual* called
    instance, and `self.forward(...)` is resolved by normal class lookup — reaching
    `CustomLinear.forward`. This test exercises the production wrapping path (real
    `apply_custom_layers_to_model`) and asserts CustomLinear.forward is reached by attaching
    a sentinel patch list and observing that the patch-aware branch runs.
    """

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(4, 4, bias=False)

    parent = Parent()
    original_linear = parent.child

    ModelLoader._wrap_forward_with_fp8_cast(original_linear, torch.float16, torch.float32)

    apply_custom_layers_to_model(parent)
    new_child = parent.child

    # Sanity: production wrapping replaced the child with a NEW CustomLinear instance.
    assert isinstance(new_child, CustomLinear)
    assert new_child is not original_linear

    # Attach a sentinel patch so CustomLinear.forward routes through the LoRA-aware branch
    # (see custom_linear.py: `if len(self._patches_and_weights) > 0`). If that branch fires,
    # our FP8 wrapping is correctly dispatched through CustomLinear.forward.
    patch_was_invoked = {"hit": False}

    class _SentinelPatch:
        def __init__(self):
            self.hit = patch_was_invoked

        def __call__(self, *_args, **_kwargs):  # not actually called
            pass

    # Patch the CustomLinear's patch-handling branch to record that it was reached.
    original_patch_branch = CustomLinear._autocast_forward_with_patches

    def tracked_patch_branch(self, input):
        patch_was_invoked["hit"] = True
        # Return a same-shape tensor so the outer caller doesn't choke.
        return torch.zeros_like(input @ self.weight.t())

    new_child._patches_and_weights = [(_SentinelPatch(), 1.0)]
    try:
        CustomLinear._autocast_forward_with_patches = tracked_patch_branch
        _ = new_child(torch.zeros(1, 4, dtype=torch.float32))
    finally:
        CustomLinear._autocast_forward_with_patches = original_patch_branch
        new_child._patches_and_weights = []

    assert patch_was_invoked["hit"] is True, (
        "FP8-wrapped forward did not reach CustomLinear.forward — LoRA/ControlLoRA patches "
        "would be silently bypassed on FP8 checkpoint models."
    )
