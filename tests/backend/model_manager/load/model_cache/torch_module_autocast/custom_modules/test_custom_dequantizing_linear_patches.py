"""Sidecar patches that read the original weight values (DoRA) on dequantizing Linears.

The weight lives in packed buffers, so a patch cannot be handed it for free: the module dequantizes it once per
forward, and only when a patch in its list reads values. Patches that read only the weight shape must not pay
for that.
"""

import gc
import weakref
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_dequantizing_linear import (
    CustomDequantizingLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.fp8_rowwise import FP8_ROWWISE_WEIGHT_DTYPE, Fp8Linear
from invokeai.backend.quantization.int8_convrot import Int4ConvrotLinear, Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.quantized_payloads import nvfp4_signed_tensors, quantize_convrot, quantize_convrot_int4

RANK = 3


def _int8(convrot: bool) -> Callable[[], tuple[DequantizingLinear, torch.Tensor]]:
    def build():
        weight = torch.randn(12, 16)
        if convrot:
            payload = quantize_convrot(weight, group_size=4)
            module = Int8ConvrotLinear(payload.codes, payload.scale, convrot=True, group_size=4)
            return module, payload.dequantized
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        codes = (weight / scale).round().to(torch.int8)
        return Int8ConvrotLinear(codes, scale, convrot=False), codes.float() * scale

    return build


def _int4():
    payload = quantize_convrot_int4(torch.randn(24, 256))
    return Int4ConvrotLinear(payload.packed, payload.scale, convrot=True), payload.dequantized


def _nvfp4():
    tensors, dense = nvfp4_signed_tensors("p", torch.rand(128, 64) > 0.5)
    return NVFP4Linear(tensors["p.weight"], tensors["p.weight_scale"], tensors["p.weight_scale_2"]), dense


def _fp8():
    weight = torch.randn(12, 16)
    scale = weight.abs().amax(dim=1) / 448.0
    module = Fp8Linear(16, 12, bias=False, compute_dtype=torch.float32)
    module.weight = (weight / scale.unsqueeze(1)).to(FP8_ROWWISE_WEIGHT_DTYPE)
    module.weight_scale = scale
    return module, module.weight.float() * scale.unsqueeze(1)


BUILDERS = {
    "int8_convrot": _int8(True),
    "int8_plain": _int8(False),
    "int4_convrot": _int4,
    "nvfp4": _nvfp4,
    "fp8_rowwise": _fp8,
}


def _wrapped(name: str) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    module, dense = BUILDERS[name]()
    model = torch.nn.Module()
    model.proj = module
    apply_custom_layers_to_model(model)
    assert isinstance(model.proj, CustomDequantizingLinear)
    return model, dense


def _dora(out_features: int, in_features: int, *, out_axis: bool) -> DoRALayer:
    magnitude = torch.rand(out_features) + 0.5 if out_axis else torch.rand(1, in_features) + 0.5
    return DoRALayer(
        up=torch.randn(out_features, RANK) * 0.3,
        down=torch.randn(RANK, in_features) * 0.3,
        dora_scale=magnitude * 3,
        alpha=None,
        bias=None,
        magnitude_is_out_dim=out_axis,
    )


def _dora_delta(w: torch.Tensor, layer: DoRALayer, out_axis: bool) -> torch.Tensor:
    """DoRA update written from the paper, independently of DoRALayer: m * (W + dW) / ||W + dW|| - W."""
    direction = w + layer.up @ layer.down
    if out_axis:
        norm = torch.sqrt((direction**2).sum(dim=1, keepdim=True))
        return layer.dora_scale.reshape(-1, 1) * direction / norm - w
    norm = torch.sqrt((direction**2).sum(dim=0, keepdim=True))
    return layer.dora_scale * direction / norm - w


def _lokr(out_features: int, in_features: int) -> LoKRLayer:
    return LoKRLayer(
        w1=torch.randn(2, 2) * 0.2,
        w1_a=None,
        w1_b=None,
        w2=torch.randn(out_features // 2, in_features // 2) * 0.2,
        w2_a=None,
        w2_b=None,
        t2=None,
        alpha=None,
        bias=None,
    )


def _loha(out_features: int, in_features: int) -> LoHALayer:
    return LoHALayer(
        w1_a=torch.randn(out_features, RANK) * 0.3,
        w1_b=torch.randn(RANK, in_features) * 0.3,
        w2_a=torch.randn(out_features, RANK) * 0.3,
        w2_b=torch.randn(RANK, in_features) * 0.3,
        t1=None,
        t2=None,
        alpha=None,
        bias=None,
    )


def _lora(out_features: int, in_features: int) -> LoRALayer:
    return LoRALayer(
        up=torch.randn(out_features, RANK) * 0.3,
        mid=None,
        down=torch.randn(RANK, in_features) * 0.3,
        alpha=None,
        bias=None,
    )


def _apply(model: torch.nn.Module, layers_and_weights: list[tuple[BaseLayerPatch, float]]):
    patches = [(ModelPatchRaw(layers={"p-proj": layer}), weight) for layer, weight in layers_and_weights]
    return LayerPatcher.apply_smart_model_patches(
        model=model, patches=patches, prefix="p-", dtype=torch.float32, force_sidecar_patching=True
    )


@pytest.mark.parametrize("out_axis", [True, False], ids=["out_axis", "in_axis"])
@pytest.mark.parametrize("name", BUILDERS)
def test_dora_on_a_quantized_linear_equals_dora_on_its_dequantized_weight(name: str, out_axis: bool) -> None:
    model, dense = _wrapped(name)
    out_features, in_features = dense.shape
    layer = _dora(out_features, in_features, out_axis=out_axis)
    x = torch.randn(5, in_features)

    with _apply(model, [(layer, 0.8)]):
        patched = model.proj(x)

    expected = F.linear(x, dense + 0.8 * _dora_delta(dense, layer, out_axis))
    assert torch.allclose(patched, expected, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(patched, F.linear(x, dense), atol=1e-2)


@pytest.mark.parametrize("name", BUILDERS)
def test_dora_beside_lora_and_lokr_patches_each_see_the_original_weight(name: str) -> None:
    model, dense = _wrapped(name)
    out_features, in_features = dense.shape
    dora = _dora(out_features, in_features, out_axis=True)
    lora, lokr = _lora(out_features, in_features), _lokr(out_features, in_features)
    x = torch.randn(5, in_features)

    with _apply(model, [(dora, 0.7), (lora, 0.5), (lokr, 0.6)]):
        patched = model.proj(x)

    assert lokr.w1 is not None and lokr.w2 is not None
    expected_weight = (
        dense + 0.7 * _dora_delta(dense, dora, True) + 0.5 * (lora.up @ lora.down) + 0.6 * torch.kron(lokr.w1, lokr.w2)
    )
    assert torch.allclose(patched, F.linear(x, expected_weight), atol=1e-4, rtol=1e-4)


def test_dora_inside_swapped_and_merged_wrappers_gets_the_weight_too() -> None:
    model, dense = _wrapped("int8_convrot")
    out_features, in_features = dense.shape
    half = out_features // 2
    x = torch.randn(5, in_features)

    inner = _dora(out_features, in_features, out_axis=True)
    with _apply(model, [(SwappedHalvesLayerPatch(inner), 1.0)]):
        swapped = model.proj(x)
    swapped_delta = _dora_delta(dense.roll(half, dims=0), inner, True).roll(half, dims=0)
    assert torch.allclose(swapped, F.linear(x, dense + swapped_delta), atol=1e-4, rtol=1e-4)

    top, bottom = _dora(half, in_features, out_axis=True), _dora(half, in_features, out_axis=True)
    merged = MergedLayerPatch([top, bottom], [Range(0, half), Range(half, out_features)])
    with _apply(model, [(merged, 1.0)]):
        patched = model.proj(x)
    delta = torch.cat([_dora_delta(dense[:half], top, True), _dora_delta(dense[half:], bottom, True)])
    assert torch.allclose(patched, F.linear(x, dense + delta), atol=1e-4, rtol=1e-4)


class _DequantSpy:
    """Counts a module's `_dequantized_weight` calls, calling through, and weakly tracks what it returns."""

    def __init__(self, module: DequantizingLinear):
        self.calls = 0
        self.returned: list[weakref.ref] = []
        self.requests: list[tuple[torch.device, torch.dtype]] = []
        original = module._dequantized_weight

        def spy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            self.calls += 1
            self.requests.append((device, dtype))
            weight = original(device, dtype)
            self.returned.append(weakref.ref(weight))
            return weight

        module._dequantized_weight = spy  # type: ignore[method-assign]


@pytest.mark.parametrize("name", ["int8_convrot", "nvfp4"])
def test_patches_that_only_need_the_weight_shape_do_not_dequantize_it(name: str) -> None:
    model, dense = _wrapped(name)
    out_features, in_features = dense.shape
    x = torch.randn(5, in_features)
    spy = _DequantSpy(model.proj)
    model.proj(x)
    forward_calls = spy.calls

    shape_only: list[tuple[BaseLayerPatch, float]] = [
        (_lora(out_features, in_features), 0.5),
        (_loha(out_features, in_features), 0.5),
        (_lokr(out_features, in_features), 0.5),
        (FullLayer(weight=torch.randn(out_features, in_features), bias=None), 0.5),
    ]
    spy.calls = 0
    with _apply(model, shape_only):
        model.proj(x)
    assert spy.calls == forward_calls

    spy.calls = 0
    with _apply(model, [*shape_only, (_dora(out_features, in_features, out_axis=True), 0.5)]):
        model.proj(x)
    assert spy.calls == forward_calls + 1


def test_leaving_the_patch_scope_keeps_no_dequantized_weight_and_changes_nothing() -> None:
    model, dense = _wrapped("nvfp4")
    out_features, in_features = dense.shape
    module = model.proj
    layer = _dora(out_features, in_features, out_axis=False)
    patch_tensors = {k: v.clone() for k, v in vars(layer).items() if isinstance(v, torch.Tensor)}
    buffers = {k: v.clone() for k, v in module.named_buffers()}
    x = torch.randn(5, in_features)
    unpatched = module(x)
    attributes = set(vars(module))
    spy = _DequantSpy(module)

    with _apply(model, [(layer, 1.0)]):
        model.proj(x)
    del module._dequantized_weight
    gc.collect()

    assert spy.calls == 2  # the module's own forward, and the one DoRA needs
    assert all(ref() is None for ref in spy.returned)
    assert set(vars(module)) == attributes
    for key, value in module.named_buffers():
        assert value.dtype == buffers[key].dtype and value.device == buffers[key].device
        assert torch.equal(value, buffers[key])
    for key, value in patch_tensors.items():
        assert torch.equal(getattr(layer, key), value)
    assert module.get_num_patches() == 0
    assert torch.equal(module(x), unpatched)


def test_a_value_reading_patch_gets_the_weight_in_the_inputs_dtype_and_device() -> None:
    model, dense = _wrapped("int8_convrot")
    out_features, in_features = dense.shape
    x = torch.randn(5, in_features, dtype=torch.float64)
    spy = _DequantSpy(model.proj)

    with _apply(model, [(_dora(out_features, in_features, out_axis=True), 1.0)]):
        model.proj(x)

    assert spy.requests and all(request == (x.device, torch.float64) for request in spy.requests)
