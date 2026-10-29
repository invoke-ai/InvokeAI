"""Ideogram 4's row-wise fp8 Linear under the model cache: LoRA as a sidecar, weights kept on the device.

The weight is a float8 buffer with a per-row float32 scale, so a LoRA cannot be written into it. The cache wraps the
module for sidecar patches (``CustomFp8Linear``); the residual has to match patching the dequantized weight directly.
The wrapper must not also make the weights streamable: the fp8 pair stays fully resident, as it did unwrapped.
"""

import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching
from invokeai.backend.quantization.fp8_rowwise import FP8_ROWWISE_WEIGHT_DTYPE, Fp8Linear

IN_FEATURES = 16
OUT_FEATURES = 12
RANK = 3


def _fp8_linear(generator: torch.Generator) -> tuple[Fp8Linear, torch.Tensor]:
    """A row-wise fp8 Linear and the float32 weight it dequantizes to."""
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator)
    scale = weight.abs().amax(dim=1) / 448.0
    module = Fp8Linear(IN_FEATURES, OUT_FEATURES, bias=False, compute_dtype=torch.float32)
    module.weight = (weight / scale.unsqueeze(1)).to(FP8_ROWWISE_WEIGHT_DTYPE)
    module.weight_scale = scale
    return module, module.weight.float() * scale.unsqueeze(1)


def test_a_lora_on_an_fp8_linear_is_applied_as_a_sidecar_and_removed_after() -> None:
    generator = torch.Generator().manual_seed(0)
    model = torch.nn.Module()
    module, dequantized = _fp8_linear(generator)
    model.add_module("proj", module)
    apply_custom_layers_to_model(model)
    up = torch.randn(OUT_FEATURES, RANK, generator=generator)
    down = torch.randn(RANK, IN_FEATURES, generator=generator)
    patch = ModelPatchRaw(layers={"proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)})
    x = torch.randn(2, IN_FEATURES, generator=generator)

    assert requires_sidecar_patching(model, ModelFormat.Diffusers)
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(patch, 0.5)], prefix="", dtype=torch.float32, force_sidecar_patching=True
    ):
        patched = model.proj(x)

    torch.testing.assert_close(patched, x @ (dequantized + 0.5 * up @ down).T)
    torch.testing.assert_close(model.proj(x), x @ dequantized.T)


def test_the_cache_keeps_fp8_weights_on_the_device_when_it_offloads() -> None:
    """Wrapped for its patches only: offloading everything it may leaves the fp8 weights resident, while a plain
    Linear beside them streams. Runs against the meta device, so it covers machines without an accelerator."""
    model = torch.nn.Module()
    model.add_module("fp8", _fp8_linear(torch.Generator().manual_seed(0))[0])
    model.add_module("dense", torch.nn.Linear(IN_FEATURES, OUT_FEATURES))
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=True)
    cached_model.full_load_to_vram()

    cached_model.partial_unload_from_vram(cached_model.total_bytes(), keep_required_weights_in_vram=True)

    assert model.fp8.weight.is_meta and model.fp8.weight_scale.is_meta
    assert not model.dense.weight.is_meta
