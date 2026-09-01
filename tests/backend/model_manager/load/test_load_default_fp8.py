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

from invokeai.backend.model_manager.load.load_default import (
    _FP8_PROBE_FAILURE_REPORTED,
    _FP8_STORAGE_SUPPORTED,
    _QUANTIZED_MODEL_FORMATS,
    ModelLoader,
    _device_supports_fp8_storage,
    _model_declared_skip_patterns,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


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


def _make_quantized_config(fmt: ModelFormat = ModelFormat.GGUFQuantized):
    """A config carrying a quantized `format`, which `_make_config` deliberately omits."""
    config = _make_config(ModelType.Main, fp8=True)
    config.format = fmt
    return config


@pytest.mark.parametrize(
    "config,submodel",
    [
        (_make_config(ModelType.VAE, fp8=True), None),
        (_make_config(ModelType.LoRA, fp8=True), None),
        # Z-Image used to be listed here. It is no longer excluded — see
        # `test_should_use_fp8_allows_z_image` for why the exclusion became obsolete.
        # A quantized model takes its place: its guard must also sit ahead of the device probe.
        (_make_quantized_config(), None),
        (_make_config(ModelType.Main, fp8=True), SubModelType.Tokenizer),
        (_make_config(ModelType.Main, fp8=False), None),
    ],
)
def test_should_use_fp8_does_not_probe_the_device_for_excluded_models(config, submodel):
    """The device probe must run only for a model that actually wants FP8.

    It allocates on the GPU, so probing before the exclusions fires it on the very first load of
    any kind -- a tokenizer, a VAE, a scheduler -- and on API/install threads it forces XPU lazy
    SYCL init on a thread that never generates.
    """
    loader = _make_loader("xpu")
    probe_path = "invokeai.backend.model_manager.load.load_default._device_supports_fp8_storage"
    with patch(probe_path) as mock_probe:
        assert loader._should_use_fp8(config, submodel) is False
    mock_probe.assert_not_called()


def test_should_use_fp8_probes_the_device_when_fp8_is_requested():
    loader = _make_loader("xpu")
    probe_path = "invokeai.backend.model_manager.load.load_default._device_supports_fp8_storage"
    config = _make_config(ModelType.Main, fp8=True)
    with patch(probe_path, return_value=True) as mock_probe:
        assert loader._should_use_fp8(config, None) is True
    mock_probe.assert_called_once()
    # An unsupported device still vetoes, just without probing on every unrelated load.
    with patch(probe_path, return_value=False):
        assert loader._should_use_fp8(config, None) is False


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


@pytest.mark.parametrize(
    "submodel_type",
    [SubModelType.PromptEnhancer, SubModelType.PromptEnhancerTokenizer],
)
def test_should_use_fp8_excludes_prompt_enhancer(submodel_type: SubModelType):
    """The ERNIE-Image prompt enhancer is a causal LM driven by `generate()` — one full forward per
    generated token, so layerwise casting pays the bf16<->fp8 round trip on every token of the
    rewritten prompt, on top of fp8 rounding a model whose whole job is text quality. It must be
    excluded like the text encoders, even though the parent Main config has `fp8_storage=true`.
    """
    loader = _make_loader(device="cuda")
    config = _make_config(ModelType.Main, fp8=True, base=BaseModelType.ErnieImage)
    assert loader._should_use_fp8(config, submodel_type) is False
    # Sanity: the same config *does* opt the transformer in, so the assertion above is about the
    # submodel exclusion and not about the config failing to enable fp8 at all.
    assert loader._should_use_fp8(config, SubModelType.Transformer) is True


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


def test_apply_fp8_to_nn_module_honors_extra_skip_patterns():
    """A model's own `_skip_layerwise_casting_patterns` must be applied on top of our defaults."""

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t_embedder = torch.nn.Linear(4, 4)
            self.attn = torch.nn.Linear(4, 4)

    storage_dtype = torch.float16
    compute_dtype = torch.float32
    model = _Model()
    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    ModelLoader._apply_fp8_to_nn_module(
        model, storage_dtype, compute_dtype, extra_skip_patterns=("t_embedder", "cap_embedder")
    )

    assert model.attn.weight.dtype == storage_dtype
    assert model.t_embedder.weight.dtype == compute_dtype


def test_apply_fp8_layerwise_casting_passes_model_declared_skip_patterns():
    """Regression test for Z-Image + fp8 crashing with
    `RuntimeError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`.

    Diffusers models declare precision-sensitive modules in `_skip_layerwise_casting_patterns`, and
    `enable_layerwise_casting()` honors them. Our hook-based replacement must read that list too —
    it is not redundant with `_FP8_DEFAULT_SKIP_PATTERNS`. `ZImageTransformer2DModel` declares
    `['t_embedder', 'cap_embedder']` because `TimestepEmbedder.forward` reads
    `self.mlp[0].weight.dtype` and casts its *input* to it: with an fp8 weight the input becomes
    float8 before the pre-hook restores the weight, and `F.linear` has no float8 kernel.
    """

    class _FakeZImage(torch.nn.Module):
        _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]

        def __init__(self):
            super().__init__()
            self.t_embedder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
            self.cap_embedder = torch.nn.Linear(4, 4)
            self.layers = torch.nn.Linear(4, 4)

    loader = _make_loader(device="cuda")
    model = _FakeZImage().to(torch.bfloat16)

    with patch.object(ModelLoader, "_should_use_fp8", return_value=True):
        loader._apply_fp8_layerwise_casting(model, _make_config(ModelType.Main, fp8=True, base=BaseModelType.ZImage))

    # The declared modules keep their compute dtype...
    assert model.t_embedder[0].weight.dtype == torch.bfloat16
    assert model.cap_embedder.weight.dtype == torch.bfloat16
    # ...while everything else is stored in fp8, so the toggle still saves VRAM.
    assert model.layers.weight.dtype == torch.float8_e4m3fn


def test_anima_transformer_declares_t_embedder_skip():
    """Regression guard for Anima + FP8 rendering a heavily dithered image.

    `AnimaTransformer.t_embedder` produces the `adaln_lora` conditioning consumed by every block,
    so casting it to FP8 corrupts every token of every block — verified against a bf16 run at the
    same seed/steps/CFG. None of the generic `_FP8_DEFAULT_SKIP_PATTERNS` match it (this
    architecture doesn't use diffusers' module names), so the model has to declare it itself.
    """
    from invokeai.backend.anima.anima_transformer import AnimaTransformer

    assert "t_embedder" in AnimaTransformer._skip_layerwise_casting_patterns

    # And the declared patterns actually reach the cast, matched against dotted module paths.
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t_embedder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
            self.blocks = torch.nn.Linear(4, 4)

    model = _Model().to(torch.float32)
    ModelLoader._apply_fp8_to_nn_module(
        model,
        storage_dtype=torch.float16,
        compute_dtype=torch.float32,
        extra_skip_patterns=tuple(AnimaTransformer._skip_layerwise_casting_patterns),
    )

    assert model.t_embedder[0].weight.dtype == torch.float32
    assert model.blocks.weight.dtype == torch.float16


@pytest.mark.parametrize(
    "fmt",
    [
        ModelFormat.GGUFQuantized,
        ModelFormat.BnbQuantizednf4b,
        ModelFormat.BnbQuantizedLlmInt8b,
        ModelFormat.SDNQQuantized,
    ],
)
def test_should_use_fp8_excludes_quantized_formats(fmt: ModelFormat):
    """Already-quantized weights must never be re-encoded as FP8.

    Casting them is not a no-op: GGUF raises `Operation changed the dtype of GGMLTensor
    unexpectedly`, and bnb NF4 corrupts silently (`bnb.nn.LinearNF4` subclasses `nn.Linear`, so its
    packed uint8 payload is cast to float8 and inference then returns finite garbage).

    Parametrized over `ModelFormat` members rather than raw strings: `_QUANTIZED_MODEL_FORMATS`
    holds strings, so testing it with strings would pass even if the enum values drifted.
    """
    loader = _make_loader(device="cuda")
    config = _make_config(ModelType.Main, fp8=True)
    config.format = fmt
    assert loader._should_use_fp8(config) is False


def test_quantized_format_set_matches_the_taxonomy():
    """Every entry in `_QUANTIZED_MODEL_FORMATS` must still name a real `ModelFormat` value.

    The set is declared as raw strings to keep `load_default` free of a taxonomy import at module
    scope, so nothing else stops a rename in `ModelFormat` from silently disabling the check —
    `config.format` would simply never match again, and FP8 would be re-enabled for that format.
    """
    assert _QUANTIZED_MODEL_FORMATS <= {fmt.value for fmt in ModelFormat}


def test_apply_fp8_skips_quantized_params_regardless_of_format():
    """Backstop behind the format check, for quantization the model's format does not reveal
    (e.g. a `diffusers`-format checkpoint quantized by an external tool).

    Both signals are covered: a non-floating-point payload (bnb's packed uint8) and a
    `torch.Tensor` subclass (GGUF's `GGMLTensor`).
    """

    class _FakeQuantTensor(torch.Tensor):
        """Stands in for GGMLTensor: a Tensor subclass carrying a quantized payload."""

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.packed = torch.nn.Linear(4, 4, bias=False)  # bnb-style uint8 payload
            self.subclassed = torch.nn.Linear(4, 4, bias=False)  # GGUF-style tensor subclass
            # NB: not `normal` — that would be caught by the `norm` skip pattern.
            self.attn = torch.nn.Linear(4, 4, bias=False)

    model = _Model().to(torch.bfloat16)
    model.packed.weight = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.uint8), requires_grad=False)
    model.subclassed.weight = torch.nn.Parameter(
        torch.zeros(4, 4, dtype=torch.bfloat16).as_subclass(_FakeQuantTensor), requires_grad=False
    )

    ModelLoader._apply_fp8_to_nn_module(model, torch.float8_e4m3fn, torch.bfloat16)

    assert model.packed.weight.dtype == torch.uint8
    assert not model.packed._forward_pre_hooks, "a quantized layer must not get cast hooks either"
    assert model.subclassed.weight.dtype == torch.bfloat16
    assert not model.subclassed._forward_pre_hooks
    # Control: an ordinary layer in the same model is still cast.
    assert model.attn.weight.dtype == torch.float8_e4m3fn


def test_should_use_fp8_allows_z_image():
    """Z-Image was excluded while we used diffusers' `enable_layerwise_casting()` with the global
    torch dtype (fp16) as compute dtype, which clashed with the model's bf16 weights. The compute
    dtype now comes from the model itself, so the exclusion is obsolete.
    """
    loader = _make_loader(device="cuda")
    assert loader._should_use_fp8(_make_config(ModelType.Main, fp8=True, base=BaseModelType.ZImage)) is True


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


def test_apply_fp8_layerwise_casting_uses_hook_path_for_model_mixin():
    """Regression test for the FLUX.2 Klein 9B partial-load device-mismatch crash.

    Diffusers' `enable_layerwise_casting()` registers a `LayerwiseCastingHook` whose
    `pre_forward` only casts dtype (not device) and whose hook system replaces
    `Linear.forward` with a wrapper that calls the *original* `Linear.forward` captured
    before the hook was installed. `ModelCache.put()` later wraps Linear as CustomLinear
    sharing `__dict__`, so the diffusers wrapper is carried into the new CustomLinear and
    routes calls to the captured original Linear.forward — bypassing
    `CustomLinear.forward`'s `cast_to_device`. On partial load (some weights on CPU,
    input on cuda), this raises a device-mismatch error.

    The fix routes ModelMixin through `_apply_fp8_to_nn_module` (hook-based,
    `forward`-preserving). This test asserts that path is taken even when the model
    inherits from ModelMixin.
    """
    from diffusers.models.modeling_utils import ModelMixin

    class _FakeModelMixin(ModelMixin):
        # ModelMixin requires a config_name class attribute and a config dict for serialization.
        # We never serialize, so we only need to satisfy isinstance() checks.
        config_name = "config.json"

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.linear(x)

    loader = _make_loader(device="cuda")
    config = _make_config(ModelType.Main, fp8=True)

    model = _FakeModelMixin()

    with (
        patch.object(ModelLoader, "_should_use_fp8", return_value=True),
        patch.object(ModelLoader, "_apply_fp8_to_nn_module") as mock_to_nn,
        patch.object(_FakeModelMixin, "enable_layerwise_casting") as mock_enable,
    ):
        loader._apply_fp8_layerwise_casting(model, config)

    mock_to_nn.assert_called_once()
    mock_enable.assert_not_called()


# ===== _device_supports_fp8_storage probe ===================================
# The probe gates FP8 storage in two places: the generic layerwise-casting path and the
# Krea 2 Qwen3-VL encoder. It must never regress CUDA, and must not claim support on CPU.


@pytest.fixture(autouse=True)
def _clear_fp8_probe_cache():
    _FP8_STORAGE_SUPPORTED.clear()
    _FP8_PROBE_FAILURE_REPORTED.clear()
    yield
    _FP8_STORAGE_SUPPORTED.clear()
    _FP8_PROBE_FAILURE_REPORTED.clear()


def test_device_supports_fp8_storage_cuda_is_unconditional():
    """CUDA is answered without probing, so the result holds on machines with no GPU."""
    assert _device_supports_fp8_storage(torch.device("cuda")) is True


def test_device_supports_fp8_storage_rejects_cpu():
    assert _device_supports_fp8_storage(torch.device("cpu")) is False


def test_device_supports_fp8_storage_xpu_probes_and_survives_failure():
    """XPU float8 support is build/driver dependent, so a failing probe must return False
    rather than propagate."""
    with patch("torch.zeros", side_effect=RuntimeError("no float8 on this build")):
        assert _device_supports_fp8_storage(torch.device("xpu")) is False


class _RecordingTensor:
    """Stands in for a tensor so the probe's cast sequence can be observed without a real GPU."""

    def __init__(self, log: list, fail_on=None):
        self._log = log
        self._fail_on = fail_on

    def to(self, target):
        if self._fail_on is not None and target == self._fail_on:
            raise RuntimeError(f"unsupported: {target}")
        self._log.append(target)
        return self


def _probe_with_recorder(device: torch.device, fail_on=None) -> tuple[bool, list]:
    log: list = []
    with patch("torch.zeros", return_value=_RecordingTensor(log, fail_on)) as mock_zeros:
        result = _device_supports_fp8_storage(device)
    # The storage cast happens on CPU at runtime, so the probe must not allocate on the device.
    assert mock_zeros.call_args.kwargs.get("device") is None
    return result, log


def test_device_supports_fp8_storage_mirrors_the_runtime_cast_sequence():
    """At runtime the storage cast is CPU-side, the fp8 tensor is copied to the device, and the
    pre-hook upcasts there. A probe that did all three on the device would pass on a build where
    the fp8 host->device copy or one upcast target fails, then break at forward time."""
    ok, log = _probe_with_recorder(torch.device("xpu", 1))
    assert ok is True
    assert log == [torch.float8_e4m3fn, torch.device("xpu", 1), torch.bfloat16, torch.float16]


def test_device_supports_fp8_storage_rejects_a_build_missing_bf16_upcast():
    """compute_dtype is bf16 for Krea-2/FLUX; fp16-only support must not report True."""
    ok, _ = _probe_with_recorder(torch.device("xpu"), fail_on=torch.bfloat16)
    assert ok is False


def test_device_supports_fp8_storage_does_not_cache_failures():
    """The probe runs during a model load, so a transient failure (e.g. the device is
    momentarily full) must not disable FP8 for the lifetime of the process."""
    with patch("torch.zeros", side_effect=torch.OutOfMemoryError("transient")):
        assert _device_supports_fp8_storage(torch.device("xpu")) is False
    ok, _ = _probe_with_recorder(torch.device("xpu"))
    assert ok is True


def test_device_supports_fp8_storage_is_cached_per_device():
    """float8 support is a per-device property; one device's answer must not decide for another."""
    ok, _ = _probe_with_recorder(torch.device("xpu", 0))
    assert ok is True
    # xpu:1 has not been probed, so a failing probe there must be observed, not short-circuited.
    with patch("torch.zeros", side_effect=RuntimeError("no float8 on this device")):
        assert _device_supports_fp8_storage(torch.device("xpu", 1)) is False


def test_model_declared_skip_patterns_unions_both_diffusers_attributes():
    """`enable_layerwise_casting()` unions `_skip_layerwise_casting_patterns` with
    `_keep_in_fp32_modules`. We replaced that call with our own hook-based path, so we have to read
    both — otherwise a model that declares only the latter loses its exclusions silently.
    """

    class _Model(torch.nn.Module):
        _skip_layerwise_casting_patterns = ["t_embedder"]
        _keep_in_fp32_modules = ["time_embedder"]

    assert _model_declared_skip_patterns(_Model()) == ("t_embedder", "time_embedder")


def test_model_declared_skip_patterns_tolerates_missing_and_odd_declarations():
    """Most models declare neither attribute; diffusers sets them to `None` on some. A bare string
    is accepted too, so a subclass that writes one instead of a list isn't silently expanded into
    per-character patterns."""

    class _Bare(torch.nn.Module):
        pass

    class _Nulls(torch.nn.Module):
        _skip_layerwise_casting_patterns = None
        _keep_in_fp32_modules = None

    class _Strings(torch.nn.Module):
        _keep_in_fp32_modules = "time_embedder"

    assert _model_declared_skip_patterns(_Bare()) == ()
    assert _model_declared_skip_patterns(_Nulls()) == ()
    assert _model_declared_skip_patterns(_Strings()) == ("time_embedder",)


def test_keep_in_fp32_modules_are_not_cast():
    """End-to-end through the cast: a module named only by `_keep_in_fp32_modules` keeps its
    compute dtype."""

    class _Model(torch.nn.Module):
        _keep_in_fp32_modules = ["time_embedder"]

        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(4, 4)
            self.attn = torch.nn.Linear(4, 4)

    loader = _make_loader(device="cuda")
    model = _Model().to(torch.bfloat16)

    with patch.object(ModelLoader, "_should_use_fp8", return_value=True):
        loader._apply_fp8_layerwise_casting(model, _make_config(ModelType.Main, fp8=True))

    assert model.time_embedder.weight.dtype == torch.bfloat16
    assert model.attn.weight.dtype == torch.float8_e4m3fn
