import copy

import pytest
import torch

from invokeai.backend.flux2.denoise_working_memory import flux2_dora_patch_bytes
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer, FactorizedDoRALayer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from tests.backend.patches.lora_conversions.test_flux2_dora_application import SplitAttention, load_adapter, load_state
from tests.backend.patches.lora_conversions.test_flux2_dora_qkv_conversion import _SplitQKV
from tests.backend.patches.test_sidecar_working_memory import _PeakAllocation


@pytest.mark.parametrize("fp8", [False, True])
@torch.no_grad()
def test_patcher_binds_norm_to_sidecar_compute_dtype(fp8):
    model = _SplitQKV("img", 7)
    attention = model.transformer_blocks[0]["attn"]
    for i, module in enumerate(attention.values()):
        module.weight.fill_(1.003 + i * 0.0001 if not fp8 else 1)
        if fp8:
            module.weight = torch.nn.Parameter(module.weight.to(torch.float8_e4m3fn), requires_grad=False)
            module.register_buffer("weight_scale", torch.tensor(1.003 + i * 0.0001))
    apply_custom_layers_to_model(model)
    layer = DoRALayer(torch.zeros(12, 1), torch.zeros(1, 7), torch.ones(1, 7), None, None)
    patch = convert_bfl_lora_patch_to_diffusers(
        ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": layer})
    )
    bases = [
        module._cast_weight_bias_for_input(torch.ones(1, 7, dtype=torch.bfloat16))[0] for module in attention.values()
    ]
    base = torch.cat(bases).float()
    expected = base / base.norm(dim=0, keepdim=True) - base
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 1.0)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.bfloat16,
        force_sidecar_patching=True,
    ):
        actual = torch.cat(
            [
                module._patches_and_weights[0][0].get_weight(base)
                for module, base in zip(attention.values(), bases, strict=True)
            ]
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_fused_input_norm_matches_actual_sidecar_base_dtype():
    model = _SplitQKV("img", 7)
    attention = model.transformer_blocks[0]["attn"]
    for i, module in enumerate(attention.values()):
        module.weight.fill_(1.003 + i * 0.0001)
    layer = DoRALayer(torch.zeros(12, 1), torch.zeros(1, 7), torch.ones(1, 7), None, None)
    patch = convert_bfl_lora_patch_to_diffusers(
        ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": layer})
    )
    bound = next(iter(patch.layers.values())).bind(model, (torch.bfloat16,) * 3)
    base = torch.cat([module.weight for module in attention.values()]).to(torch.bfloat16).float()
    expected = base / base.norm(dim=0, keepdim=True) - base
    residual = torch.cat(
        [
            part.get_weight(module.weight.to(torch.bfloat16))
            for (_, part), module in zip(bound, attention.values(), strict=True)
        ]
    )
    torch.testing.assert_close(residual.float(), expected, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_output_axis_bfloat16_small_direction_residual_survives():
    base = torch.ones(4, 7, dtype=torch.bfloat16)
    up = torch.tensor([[0.001953125], [-0.001953125], [0.001953125], [-0.001953125]], dtype=torch.bfloat16)
    down = torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.bfloat16)
    magnitude = torch.full((4,), 2.640625, dtype=torch.bfloat16)
    layer = DoRALayer(up, down, magnitude, None, None, magnitude_is_out_dim=True)
    direction = base.float() + up.float() @ down.float()
    expected = magnitude.float().reshape(-1, 1) * direction / direction.norm(dim=1, keepdim=True) - base.float()
    torch.testing.assert_close(layer.get_weight(base).float(), expected, atol=1e-6, rtol=1e-6)


def test_split_factors_do_not_retain_duplicate_promotions():
    model = _SplitQKV("img", 7)
    layer = DoRALayer(
        torch.ones(12, 2, dtype=torch.bfloat16),
        torch.ones(2, 7, dtype=torch.bfloat16),
        torch.ones(1, 7, dtype=torch.bfloat16),
        None,
        None,
    )
    patch = convert_bfl_lora_patch_to_diffusers(
        ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": layer})
    )
    parts = [part for _, part in next(iter(patch.layers.values())).bind(model)]
    for part in parts:
        part.to(device=torch.device("cpu"), dtype=torch.bfloat16)
    assert all(part.down.dtype == torch.bfloat16 for part in parts)
    assert len({part.down.untyped_storage().data_ptr() for part in parts}) == 1


@pytest.mark.parametrize("suffix", ["lora_magnitude_vector.weight", "lora_magnitude_vector", "magnitude"])
def test_bfl_magnitude_loads_from_safetensors(tmp_path, suffix):
    prefix = "diffusion_model.double_blocks.0.img_attn.qkv"
    state = {
        f"{prefix}.lora_A.weight": torch.ones(2, 7),
        f"{prefix}.lora_B.weight": torch.ones(12, 2),
        f"{prefix}.{suffix}": torch.arange(1, 13, dtype=torch.float32),
    }
    patch = convert_bfl_lora_patch_to_diffusers(load_state(tmp_path, state))
    layers = list(patch.layers.values())
    assert len(layers) == 3
    assert all(isinstance(layer, DoRALayer) and layer.magnitude_is_out_dim for layer in layers)
    torch.testing.assert_close(torch.cat([layer.dora_scale for layer in layers]), state[f"{prefix}.{suffix}"])


@pytest.mark.parametrize("sidecar", [False, True])
@torch.no_grad()
def test_dora_budget_covers_live_cpu_tensor_allocations(tmp_path, sidecar):
    adapter, _ = load_adapter(tmp_path, "input", "kohya", torch.bfloat16)
    patch = convert_bfl_lora_patch_to_diffusers(adapter)
    model = SplitAttention(14)
    attention = model.transformer_blocks[0]["attn"]
    for module in attention.values():
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn if sidecar else torch.bfloat16)
        if sidecar:
            module.register_buffer("weight_scale", torch.linspace(0.6, 1.0, 4))
    patches = [
        (ModelPatchRaw({key: copy.copy(layer) for key, layer in patch.layers.items()}), strength)
        for strength in (0.5, -0.6)
    ]
    x = torch.randn(5, 7)
    budget = flux2_dora_patch_bytes(model, patches, dtype=x.dtype, rows=x.shape[0])
    allocation = _PeakAllocation()
    with (
        allocation,
        LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=patches,
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=x.dtype,
            force_direct_patching=not sidecar,
        ),
    ):
        if sidecar:
            attention["to_q"](x)
    assert 0 < allocation.peak <= budget


def _factorized_dora_layer(kind: str, dtype: torch.dtype) -> FactorizedDoRALayer:
    """A LoKr or LoHa with an output-axis magnitude for a Linear(7, 4)."""
    generator = torch.Generator().manual_seed(11)

    def randn(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.3).to(dtype)

    if kind == "lokr":
        factors: LoKRLayer | LoHALayer = LoKRLayer(
            w1=randn(2, 1), w1_a=None, w1_b=None, w2=randn(2, 7), w2_a=None, w2_b=None, t2=None, alpha=None, bias=None
        )
    else:
        factors = LoHALayer(
            w1_a=randn(4, 2),
            w1_b=randn(2, 7),
            w2_a=randn(4, 2),
            w2_b=randn(2, 7),
            t1=None,
            t2=None,
            alpha=None,
            bias=None,
        )
    return FactorizedDoRALayer(factors, dora_scale=randn(4, 1).abs() + 0.5, magnitude_is_out_dim=True)


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("kind", ["lokr", "loha"])
@torch.no_grad()
def test_factorized_dora_budget_covers_live_cpu_tensor_allocations(kind, sidecar):
    model = SplitAttention(14)
    attention = model.transformer_blocks[0]["attn"]
    for module in attention.values():
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn if sidecar else torch.bfloat16)
        if sidecar:
            module.register_buffer("weight_scale", torch.linspace(0.6, 1.0, 4))
    patches = [
        (
            ModelPatchRaw(
                {
                    f"{FLUX_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.{name}": _factorized_dora_layer(
                        kind, torch.bfloat16
                    )
                    for name in ("to_q", "to_k")
                }
            ),
            strength,
        )
        for strength in (0.5, -0.6)
    ]
    x = torch.randn(5, 7)
    budget = flux2_dora_patch_bytes(model, patches, dtype=x.dtype, rows=x.shape[0])
    allocation = _PeakAllocation()
    with (
        allocation,
        LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=patches,
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=x.dtype,
            force_direct_patching=not sidecar,
        ),
    ):
        if sidecar:
            attention["to_q"](x)
            attention["to_k"](x)
    assert 0 < allocation.peak <= budget


def test_dev_width_dora_reserves_more_than_one_flat_adapter_allowance(tmp_path):
    width = 6144
    model = SplitAttention(0)
    attention = model.transformer_blocks[0]["attn"]
    for name in model.names[:3]:
        attention[name] = torch.nn.Linear(width, width, bias=False, device="meta", dtype=torch.bfloat16)
    key = "lora_unet_double_blocks_0_img_attn_qkv"
    adapter = load_state(
        tmp_path,
        {
            f"{key}.lora_up.weight": torch.ones(3 * width, 2),
            f"{key}.lora_down.weight": torch.ones(2, width),
            f"{key}.dora_scale": torch.ones(1, width),
        },
        torch.bfloat16,
    )
    patch = convert_bfl_lora_patch_to_diffusers(adapter)
    budget = flux2_dora_patch_bytes(model, [(patch, 0.5)], dtype=torch.bfloat16, rows=1)
    # Four concurrently live FP32 matrices in the direction/residual calculation alone.
    assert budget >= 4 * width * width * torch.float32.itemsize > 512 * 2**20
    assert flux2_dora_patch_bytes(model, [(patch, 0.0)], dtype=torch.bfloat16, rows=1) == 0


def test_klein_width_packed_dora_prices_the_unpacked_projection_shape(tmp_path):
    in_features, out_features = 4096, 36864
    model = torch.nn.Module()
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_qkv_mlp_proj = Int8ConvrotLinear(
        torch.empty(out_features, in_features, device="meta", dtype=torch.int8),
        torch.empty(out_features, 1, device="meta"),
        convrot=True,
    )
    model.single_transformer_blocks = torch.nn.ModuleList([block])
    key = "diffusion_model.single_blocks.0.linear1"
    patch = load_state(
        tmp_path,
        {
            f"{key}.lora_A.weight": torch.ones(2, in_features),
            f"{key}.lora_B.weight": torch.ones(out_features, 2),
            f"{key}.dora_scale": torch.ones(out_features, 1),
        },
        torch.bfloat16,
    )
    budget = flux2_dora_patch_bytes(
        model, [(convert_bfl_lora_patch_to_diffusers(patch), 0.5)], dtype=torch.bfloat16, rows=1
    )
    assert budget >= 4 * in_features * out_features * torch.float32.itemsize
