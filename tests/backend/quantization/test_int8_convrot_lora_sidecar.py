"""Sidecar LoRA patching over Int8ConvrotLinear modules.

The int8 weight is a buffer (not a Parameter), so this exercises three pieces at once:
the ``CustomInt8ConvrotLinear`` wrapper the model cache installs, the ``LayerPatcher``'s
buffers-only device fallback, and the numerical claim that the sidecar residual equals
patching the dequantized weights directly.
"""

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    AUTOCAST_MODULE_TYPE_MAPPING,
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.int8_convrot import (
    Int8ConvrotLinear,
    build_regular_hadamard,
    dequantize_convrot_weight,
)

IN_FEATURES = 16
OUT_FEATURES = 12
GROUP_SIZE = 4
RANK = 3


def _quantize_per_channel(weight: torch.Tensor, convrot: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization (optionally after convrot rotation)."""
    if convrot:
        h = build_regular_hadamard(GROUP_SIZE)
        out, in_ = weight.shape
        # Rotation is its own inverse-transpose here (H symmetric orthonormal): W_rot = grouped(W) @ H.
        weight = (weight.view(out, in_ // GROUP_SIZE, GROUP_SIZE) @ h).view(out, in_)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _make_module(convrot: bool) -> tuple[Int8ConvrotLinear, torch.Tensor]:
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    q, scale = _quantize_per_channel(weight, convrot)
    module = Int8ConvrotLinear(weight=q, weight_scale=scale, convrot=convrot, group_size=GROUP_SIZE)
    dequant = dequantize_convrot_weight(q, scale, convrot, torch.float32, group_size=GROUP_SIZE)
    return module, dequant


def test_int8_convrot_linear_is_registered_for_custom_wrapping():
    assert Int8ConvrotLinear in AUTOCAST_MODULE_TYPE_MAPPING


def test_sidecar_lora_matches_dequantized_direct_patch():
    torch.manual_seed(1)
    model = torch.nn.Module()
    module, dequant = _make_module(convrot=True)
    model.add_module("proj", module)
    apply_custom_layers_to_model(model)

    down = torch.randn(RANK, IN_FEATURES)
    up = torch.randn(OUT_FEATURES, RANK)
    layer = LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)
    patch = ModelPatchRaw(layers={"lora_transformer-proj": layer})

    x = torch.randn(5, IN_FEATURES)
    baseline = model.proj(x)
    assert torch.allclose(baseline, torch.nn.functional.linear(x, dequant), atol=1e-5)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 0.75)],
        prefix="lora_transformer-",
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        patched = model.proj(x)
        expected = torch.nn.functional.linear(x, dequant + 0.75 * (up @ down))
        assert torch.allclose(patched, expected, atol=1e-4)
        # The residual rides in activation space: the quantized storage is untouched.
        assert model.proj.weight.dtype == torch.int8

    # Unpatched again after the context exits.
    assert torch.allclose(model.proj(x), baseline)
    assert model.proj.get_num_patches() == 0


def test_sidecar_lora_without_convrot():
    model = torch.nn.Module()
    module, dequant = _make_module(convrot=False)
    model.add_module("proj", module)
    apply_custom_layers_to_model(model)

    down = torch.randn(RANK, IN_FEATURES)
    up = torch.randn(OUT_FEATURES, RANK)
    patch = ModelPatchRaw(
        layers={"lora_transformer-proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)}
    )

    x = torch.randn(3, IN_FEATURES)
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 1.0)],
        prefix="lora_transformer-",
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        patched = model.proj(x)
    expected = torch.nn.functional.linear(x, dequant + up @ down)
    assert torch.allclose(patched, expected, atol=1e-4)
