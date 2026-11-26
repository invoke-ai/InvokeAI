import re
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from tests.backend.patches.lora_conversions.test_flux2_dora_application import (
    SplitAttention,
    fused_expected,
    load_adapter,
    load_state,
)
from tests.backend.patches.lora_conversions.test_flux2_dora_qkv_conversion import _SplitQKV


@pytest.mark.parametrize("variant", ["aitoolkit", "peft-bfl", "kohya", "loha", "lokr"])
@torch.no_grad()
def test_loader_keeps_other_flux_adapter_routes_numerically_unchanged(tmp_path, variant):
    up = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
    down = torch.arange(14, dtype=torch.float32).reshape(2, 7) / 14
    delta = up @ down
    values = {"lora_A.weight": down, "lora_B.weight": up}
    key = "diffusion_model.double_blocks.0.img_attn.qkv"
    if variant == "peft-bfl":
        key = "base_model.model.double_blocks.0.img_attn.qkv"
    elif variant == "kohya":
        key = "lora_unet_double_blocks_0_img_attn_qkv"
        values = {"lora_down.weight": down, "lora_up.weight": up}
    elif variant == "loha":
        values = {"hada_w1_a": up, "hada_w1_b": down, "hada_w2_a": up + 0.2, "hada_w2_b": down - 0.1}
        delta = delta * ((up + 0.2) @ (down - 0.1))
    elif variant == "lokr":
        w1 = torch.tensor([[0.3], [-0.4], [0.5]])
        w2 = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 28
        values = {"lokr_w1": w1, "lokr_w2": w2}
        delta = torch.kron(w1, w2)
    patch = load_state(tmp_path, {f"{key}.{suffix}": value for suffix, value in values.items()})
    model = _SplitQKV("img", 7)
    attention = model.transformer_blocks[0]["attn"]
    base = torch.cat([attention[name].weight for name in model.names])
    bias = torch.cat([attention[name].bias for name in model.names])
    x = torch.ones(2, 7)
    expected = torch.nn.functional.linear(x, base + 0.6 * delta, bias)
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(convert_bfl_lora_patch_to_diffusers(patch), 0.6)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        torch.testing.assert_close(model(x), expected, atol=4e-6, rtol=4e-6)


@torch.no_grad()
def test_vector_input_magnitude_and_sparse_bias_on_square_fused_weight():
    generator = torch.Generator().manual_seed(31)
    model = _SplitQKV("txt", 12)
    apply = model.transformer_blocks[0]["attn"]
    base = torch.randn(12, 12, generator=generator)
    for index, name in enumerate(model.names):
        apply[name].weight.copy_(base[index * 4 : (index + 1) * 4])
    bias = torch.cat([apply[name].bias for name in model.names])
    up = torch.randn(12, 2, generator=generator)
    down = torch.randn(2, 12, generator=generator)
    magnitude = torch.linspace(0.3, 1.5, 12)
    sparse_bias = torch.sparse_coo_tensor([[0, 5, 11]], [0.2, -0.4, 0.6], (12,))
    layer = DoRALayer(up, down, magnitude, 3.0, sparse_bias)
    patch = convert_bfl_lora_patch_to_diffusers(
        ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.txt_attn.qkv": layer})
    )
    direction = base + 1.5 * (up @ down)
    normalized = direction * magnitude / direction.norm(dim=0, keepdim=True)
    x = torch.randn(2, 12, generator=generator)
    expected = torch.nn.functional.linear(
        x, base + 0.6 * (normalized - base), bias + 0.6 * 1.5 * sparse_bias.to_dense()
    )
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 0.6)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        torch.testing.assert_close(model(x), expected, atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize("sidecar", [False, True])
@torch.no_grad()
def test_multiple_adapters_preserve_direct_order_and_sidecar_base_semantics(tmp_path, sidecar):
    patches, all_factors = [], []
    for axis in ("input", "output"):
        directory = tmp_path / axis
        directory.mkdir()
        patch, factors = load_adapter(directory, axis, "diffusion_model.", torch.bfloat16)
        patches.append(patch)
        all_factors.append(factors)
    model = SplitAttention(924)
    attention = model.transformer_blocks[0]["attn"]
    original = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(2, 7, generator=torch.Generator().manual_seed(10))
    for order in ((0, 1), (1, 0)):
        expected = []
        for stream, names in enumerate((model.names[:3], model.names[3:])):
            base = torch.cat([attention[name].weight for name in names])
            bias = torch.cat([attention[name].bias for name in names])
            updated = base.clone()
            for index in order:
                strength = (-0.25, 0.7)[index]
                up, down, magnitude = all_factors[index][stream]
                current = base if sidecar else updated
                direction = current + 1.5 * (up @ down)
                normalized = direction * magnitude / direction.norm(dim=index, keepdim=True)
                updated = updated + strength * (normalized - current)
            expected.append(torch.nn.functional.linear(x, updated, bias))
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(convert_bfl_lora_patch_to_diffusers(patches[index]), (-0.25, 0.7)[index]) for index in order],
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
            force_direct_patching=not sidecar,
            force_sidecar_patching=sidecar,
        ):
            torch.testing.assert_close(model(x), torch.cat(expected, dim=-1), atol=4e-6, rtol=4e-6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)


@pytest.mark.parametrize("axis,scaled", [("input", False), ("input", True), ("output", False)])
@torch.no_grad()
def test_native_fp8_base_uses_dense_values_and_keeps_storage(tmp_path, axis, scaled):
    patch, factors = load_adapter(tmp_path, axis, "kohya", torch.bfloat16)
    model = SplitAttention(63)
    reference = SplitAttention(63)
    for module, dense in zip(
        model.transformer_blocks[0]["attn"].values(), reference.transformer_blocks[0]["attn"].values(), strict=True
    ):
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        dense.weight.copy_(module.weight.float())
        if scaled:
            scale = torch.linspace(0.4, 0.8, 4)
            module.register_buffer("weight_scale", scale)
            dense.weight.mul_(scale.reshape(-1, 1))
    x = torch.randn(2, 7, generator=torch.Generator().manual_seed(12))
    expected = fused_expected(reference, factors, x, -0.3, axis)
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(convert_bfl_lora_patch_to_diffusers(patch), -0.3)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        # Native FP8 must select sidecars even when direct patching is requested.
        force_direct_patching=True,
    ):
        torch.testing.assert_close(model(x), expected, atol=4e-6, rtol=4e-6)
    for module in model.transformer_blocks[0]["attn"].values():
        assert module.weight.dtype == torch.float8_e4m3fn
        assert module.get_num_patches() == 0


@torch.no_grad()
def test_shared_norm_uses_bounded_row_blocks_without_modifying_cached_patch(tmp_path, monkeypatch):
    import invokeai.backend.patches.layers.fused_input_dora_layer as fused

    patch, factors = load_adapter(tmp_path, "input", "base_model.model.", torch.bfloat16)
    model = SplitAttention(92)
    block_shapes = []
    addmm = torch.addmm

    def record_addmm(base, up, down, **kwargs):
        block_shapes.append(base.shape)
        return addmm(base, up, down, **kwargs)

    monkeypatch.setattr(fused, "_NORM_BLOCK_ELEMENTS", 14)
    monkeypatch.setattr(torch, "addmm", record_addmm)
    layer = next(iter(patch.layers.values()))
    split = layer.bind(model)
    up, down, _ = factors[0]
    attention = model.transformer_blocks[0]["attn"]
    direction = torch.cat([attention[name].weight for name in model.names[:3]]) + 1.5 * (up @ down)
    assert len(block_shapes) == 6 and all(rows * columns <= 14 for rows, columns in block_shapes)
    assert len(split) == 3
    assert all(part.direction_norm is split[0][1].direction_norm for _, part in split)
    torch.testing.assert_close(split[0][1].direction_norm, direction.norm(dim=0, keepdim=True))
    assert layer.layer.up.dtype == torch.bfloat16
    assert layer.layer.direction_norm is None


@pytest.mark.parametrize("sidecar", [False, True])
@torch.no_grad()
def test_binding_failure_restores_prior_adapter_and_releases_cache_pins(tmp_path, sidecar):
    patch, _ = load_adapter(tmp_path, "input", "kohya", torch.bfloat16)
    # The second adapter has an incompatible input width, after the first was already applied.
    model = SplitAttention(35)
    attention = model.transformer_blocks[0]["attn"]
    original = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.ones(1, 7)
    baseline = model(x)
    active = []

    @contextmanager
    def model_in_ram():
        active.append(True)
        try:
            yield patch
        finally:
            active.pop()

    invocation = SimpleNamespace(transformer=SimpleNamespace(loras=[SimpleNamespace(lora="adapter", weight=0.5)]))
    context = SimpleNamespace(
        models=SimpleNamespace(load=lambda _: SimpleNamespace(model=patch, model_in_ram=model_in_ram))
    )
    bad, _ = load_adapter(tmp_path, "input", "base_model.model.", torch.bfloat16)
    next(iter(bad.layers.values())).layer.down = torch.ones(2, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError), ExitStack() as stack:
        good = Flux2DenoiseInvocation._materialize_lora_patches(invocation, context, stack)[0]
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[good, (bad, 0.7, model_in_ram())],
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
            force_direct_patching=not sidecar,
            force_sidecar_patching=sidecar,
        ):
            pytest.fail("incompatible base must fail during binding")
    assert not active
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    assert all(module.get_num_patches() == 0 for module in attention.values())
    torch.testing.assert_close(model(x), baseline, atol=0, rtol=0)


@pytest.mark.parametrize("missing", ["to_q", "to_k", "to_v"])
@pytest.mark.parametrize("suppress", [False, True])
@torch.no_grad()
def test_missing_fused_projection_skips_group_and_applies_other_layers(tmp_path, monkeypatch, missing, suppress):
    patch, _ = load_adapter(tmp_path, "input", "kohya", torch.float32)
    patch = convert_bfl_lora_patch_to_diffusers(patch)
    model = SplitAttention(35)
    attention = model.transformer_blocks[0]["attn"]
    del attention[missing]
    logger = MagicMock()
    monkeypatch.setattr("invokeai.backend.patches.layer_patcher.InvokeAILogger.get_logger", lambda _: logger)
    original = {name: module.weight.clone() for name, module in attention.items()}
    pattern = re.compile("transformer_blocks") if suppress else None

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 0.7)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
        suppress_warning_layers=pattern,
    ):
        for name in set(model.names[:3]) - {missing}:
            torch.testing.assert_close(attention[name].weight, original[name], rtol=0, atol=0)
        assert not torch.equal(attention["add_q_proj"].weight, original["add_q_proj"])
    assert logger.warning.call_count == (0 if suppress else 1)
    for name, module in attention.items():
        torch.testing.assert_close(module.weight, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_bfloat16_fused_input_sidecar_retains_small_direction_residual(storage):
    # The low-precision forward is the behavior under test; the reference constructs the fused residual in float32.
    model = _SplitQKV("img", 7).to(dtype=torch.bfloat16)
    attention = model.transformer_blocks[0]["attn"]
    for module in attention.values():
        module.weight.fill_(1)
        module.bias.zero_()
    apply_custom_layers_to_model(model)
    up = torch.full((12, 1), 0.001953125 if storage == torch.bfloat16 else 0.251, dtype=storage)
    up[::2] *= -1
    down = torch.zeros(1, 7, dtype=storage)
    down[0, 0] = 1
    magnitude = torch.full((1, 7), 3.46875 if storage == torch.bfloat16 else 3.47, dtype=storage)
    patch = convert_bfl_lora_patch_to_diffusers(
        ModelPatchRaw(
            {f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": DoRALayer(up, down, magnitude, None, None)}
        )
    )
    base = torch.ones(12, 7)
    direction = base + up.float() @ down.float()
    residual = magnitude.float() * direction / direction.norm(dim=0, keepdim=True) - base
    x = torch.tensor([[1, -1, 0, 0, 0, 0, 0]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, residual.to(torch.bfloat16))
    assert expected.abs().min() > 0
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 1.0)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.bfloat16,
        force_sidecar_patching=True,
    ):
        torch.testing.assert_close(model(x), expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("wrapper", ["none", "swapped", "merged"])
@torch.no_grad()
def test_flux2_overlapping_applications_do_not_mutate_cached_adapter(tmp_path, sidecar, wrapper):
    patch, _ = load_adapter(tmp_path, "input", "kohya", torch.bfloat16)
    # A diffusers-named layer passes through the converter unchanged, as do simple BFL renames.
    shared = DoRALayer(
        torch.ones(4, 2, dtype=torch.bfloat16),
        torch.ones(2, 7, dtype=torch.bfloat16),
        torch.ones(4, 1, dtype=torch.bfloat16),
        None,
        None,
        magnitude_is_out_dim=True,
    )
    layer = shared
    if wrapper == "swapped":
        layer = SwappedHalvesLayerPatch(shared)
    elif wrapper == "merged":
        layer = MergedLayerPatch([shared], [Range(0, 4)])
    key = f"{FLUX_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q"
    patch.layers[key] = layer
    original_up = shared.up
    original_size = patch.calc_size()
    active = []

    @contextmanager
    def pin():
        active.append(True)
        try:
            yield patch
        finally:
            active.pop()

    context = SimpleNamespace(models=SimpleNamespace(load=lambda _: SimpleNamespace(model=patch, model_in_ram=pin)))
    invocation = SimpleNamespace(transformer=SimpleNamespace(loras=[SimpleNamespace(lora="adapter", weight=0.5)]))
    first, second = SplitAttention(1), SplitAttention(2)
    args = {
        "prefix": FLUX_LORA_TRANSFORMER_PREFIX,
        "dtype": torch.float32,
        "force_sidecar_patching": sidecar,
        "force_direct_patching": not sidecar,
    }
    with ExitStack() as first_stack:
        patches = Flux2DenoiseInvocation._materialize_lora_patches(invocation, context, first_stack)
        [(leaf, _, _)] = LayerPatcher._leaf_placements(patches[0][0].layers[key])
        assert leaf is not shared and leaf.up is original_up
        first_stack.enter_context(LayerPatcher.apply_smart_model_patches(model=first, patches=patches, **args))
        if not sidecar:
            assert leaf.up is original_up and leaf.down is shared.down
        assert shared.up is original_up and shared.up.dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match="cancelled"), ExitStack() as second_stack:
            patches = Flux2DenoiseInvocation._materialize_lora_patches(invocation, context, second_stack)
            second_stack.enter_context(LayerPatcher.apply_smart_model_patches(model=second, patches=patches, **args))
            assert len(active) == 2
            assert shared.up is original_up and shared.up.dtype == torch.bfloat16
            assert patch.calc_size() == original_size
            raise RuntimeError("cancelled")
        assert len(active) == 1
        assert shared.up is original_up
    assert not active
    assert shared.up is original_up
    assert all(
        module.get_num_patches() == 0
        for model in (first, second)
        for module in model.transformer_blocks[0]["attn"].values()
    )
