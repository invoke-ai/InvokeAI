from contextlib import contextmanager
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions import flux_bfl_peft_lora_conversion_utils as conversion
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor


def quantized_linear(kind: str) -> torch.nn.Module:
    if kind == "int8":
        return Int8ConvrotLinear(torch.ones(128, 64, dtype=torch.int8), torch.ones(128, 1), convrot=False)
    if kind == "nvfp4":
        return NVFP4Linear(
            torch.full((128, 32), 0x22, dtype=torch.uint8),
            torch.ones(128, 4).to(torch.float8_e4m3fn),
            torch.tensor(1.0),
        )
    linear = torch.nn.Linear(64, 128, bias=False)
    if kind == "gguf":
        weight = quantize_tensor(torch.ones(128, 64), gguf.GGMLQuantizationType.Q8_0)
    else:
        assert kind == "sdnq"
        weight = SDNQTensor(
            data=torch.ones(128, 64, dtype=torch.int8),
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size((128, 64)),
            compute_dtype=torch.float32,
            scale=torch.ones(1),
        )
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    return linear


def model_and_state(kind: str, mixed: bool):
    attention = torch.nn.ModuleDict({name: quantized_linear(kind) for name in ("to_q", "to_k", "to_v")})
    if mixed:
        attention["to_k"] = torch.nn.Linear(64, 128, bias=False)
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"attn": attention})])
    model.proj = quantized_linear(kind)
    apply_custom_layers_to_model(model)
    state = {}
    for key, outputs in (("double_blocks.0.img_attn.qkv", 384), ("proj", 128)):
        state[f"diffusion_model.{key}.lora_A.weight"] = torch.ones(2, 64) * 0.1
        state[f"diffusion_model.{key}.lora_B.weight"] = torch.ones(outputs, 2) * 0.2
        state[f"diffusion_model.{key}.dora_scale"] = torch.ones(1, 64) * 3
    return model, state


@pytest.mark.parametrize("kind", ["gguf", "sdnq", "int8", "nvfp4"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
@torch.no_grad()
def test_quantized_fused_group_warns_and_skips_but_unfused_dora_applies(kind, mixed, interrupt, monkeypatch):
    model, state = model_and_state(kind, mixed)
    attention = model.transformer_blocks[0]["attn"]
    logger = MagicMock()
    monkeypatch.setattr(conversion, "logger", logger)
    monkeypatch.setattr("invokeai.backend.patches.layer_patcher.InvokeAILogger.get_logger", lambda _: logger)
    patch = conversion.lora_model_from_flux2_bfl_peft_state_dict(state)
    x = torch.ones(1, 64)
    baseline = [attention[name](x).clone() for name in ("to_q", "to_k", "to_v")]
    baseline_proj = model.proj(x).clone()
    active = []

    @contextmanager
    def pin():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    model_format = {"gguf": ModelFormat.GGUFQuantized, "sdnq": ModelFormat.SDNQQuantized}.get(
        kind, ModelFormat.Checkpoint
    )
    assert requires_sidecar_patching(model, model_format)
    with pytest.raises(RuntimeError, match="cancelled") if interrupt else _no_exception():
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(patch, 0.7, pin())],
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
            force_sidecar_patching=True,
        ):
            assert active
            for index, name in enumerate(("to_q", "to_k", "to_v")):
                assert attention[name].get_num_patches() == 0
                torch.testing.assert_close(attention[name](x), baseline[index], rtol=0, atol=0)
            assert model.proj.get_num_patches() == 1
            assert not torch.allclose(model.proj(x), baseline_proj)
            if interrupt:
                raise RuntimeError("cancelled")
    assert not active
    assert logger.warning.call_count == 1
    assert "skip" in str(logger.warning.call_args).lower()
    assert all(attention[name].get_num_patches() == 0 for name in ("to_q", "to_k", "to_v"))
    assert model.proj.get_num_patches() == 0
    torch.testing.assert_close(model.proj(x), baseline_proj, rtol=0, atol=0)


@contextmanager
def _no_exception():
    yield


@pytest.mark.parametrize("kind", ["gguf", "sdnq", "int8", "nvfp4"])
def test_quantized_fused_group_does_not_hide_malformed_dimensions(kind):
    model, state = model_and_state(kind, False)
    patch = conversion.lora_model_from_flux2_bfl_peft_state_dict(state)
    fused = next(layer for key, layer in patch.layers.items() if "to_q" in key)
    fused.layer.down = torch.ones(2, 63)
    with (
        pytest.raises(RuntimeError, match="dimensions"),
        LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(patch, 1)],
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
            force_sidecar_patching=True,
        ),
    ):
        pytest.fail("malformed adapter must fail")
