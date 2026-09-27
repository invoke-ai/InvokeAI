"""nvfp4: byte layout, block-scale layout, the per-layer state-dict passes and the packed module.

No expected value here comes from `nvfp4.py` itself. Nibble order and the E2M1 table are checked
against literal bytes; the block-scale layout against the index formula measured on real checkpoints,
written out as arithmetic; and the whole decode against a slice of a real Comfy-Org checkpoint and its
bf16 build. That slice is one tile row high: it pins the layout inside a row of tiles against real data,
while the order of the tile rows rests on the formula test.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.quantization.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4_weight,
    install_nvfp4_layers,
    pop_nvfp4_layers,
    predict_nvfp4_install_size,
    split_nvfp4_rows,
)
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_tensors, stored_layout

FIXTURE = Path(__file__).parent / "data" / "z_image_turbo_nvfp4_slices.safetensors"


def _nvfp4_tensors(path: str, codes: torch.Tensor, block_scale: float, global_scale: float) -> dict[str, torch.Tensor]:
    """One layer in checkpoint layout, carrying the per-tensor marker this module's subject reads."""
    layer = nvfp4_tensors(path, codes, block_scale=block_scale, global_scale=global_scale)
    layer[f"{path}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
    return layer


def _zero_layer(path: str, rows: int = 128) -> dict[str, torch.Tensor]:
    return _nvfp4_tensors(path, torch.zeros(rows, 64, dtype=torch.uint8), 1.0, 1.0)


def _signed_layer(path: str, positive: torch.Tensor) -> dict[str, torch.Tensor]:
    """Codes 2 and 10 are +1.0 and -1.0; with a block scale of 2 and a global scale of 0.25 the weight is +-0.5."""
    return _nvfp4_tensors(path, torch.where(positive, 2, 10).to(torch.uint8), block_scale=2.0, global_scale=0.25)


def _packed_linear(tensors: dict[str, torch.Tensor], path: str, bias: torch.Tensor | None = None) -> NVFP4Linear:
    return NVFP4Linear(
        tensors[f"{path}.weight"], tensors[f"{path}.weight_scale"], tensors[f"{path}.weight_scale_2"], bias=bias
    )


def test_the_upper_nibble_is_the_first_element_and_both_scales_multiply() -> None:
    weight = torch.zeros(128, 32, dtype=torch.uint8)
    # Every code once: 0x7F is 7 then 15, 0x19 is 1 then 9, and so on.
    weight[0, :8] = torch.tensor([0x7F, 0x19, 0x2C, 0x35, 0x46, 0xAB, 0xDE, 0x08], dtype=torch.uint8)
    scale = torch.full((128, 4), 4.0).to(torch.float8_e4m3fn)

    decoded = dequantize_nvfp4_weight(weight, scale, torch.tensor(0.125), torch.float32)

    codes = [6.0, -6.0, 0.5, -0.5, 1.0, -2.0, 1.5, 3.0, 2.0, 4.0, -1.0, -1.5, -3.0, -4.0, 0.0, -0.0]
    assert torch.equal(decoded[0, :16], torch.tensor(codes) * 0.5)
    assert decoded.shape == (128, 64)


@pytest.mark.parametrize("layer", ["layers.0.attention.out", "layers.5.feed_forward.w2"])
def test_a_real_checkpoint_slice_decodes_to_its_bf16_build(layer: str) -> None:
    """Comfy-Org/z_image_turbo at revision 08d04455279082882deaabc8d0d09fc914c071e1 (Apache-2.0): the
    first 128 rows and input columns of `split_files/diffusion_models/z_image_turbo_nvfp4.safetensors`
    and of `z_image_turbo_bf16.safetensors` beside it. The block scales are the first 8x128 entries of
    the stored grid, which is exactly the tiled layout of that 128x8 sub-grid.

    Measured when cut: cosine 0.9953 and 0.9955 decoded, 0.9102 and 0.8801 with the grid read row by
    row. The latter loads without an error and quietly degrades the model -- the failure pinned here.
    The norm check is what catches a global scale applied the wrong way, which leaves the cosine alone.
    """
    fixture = load_file(FIXTURE)
    weight, scale = fixture[f"{layer}.weight"], fixture[f"{layer}.weight_scale"]
    scale_2 = fixture[f"{layer}.weight_scale_2"]
    reference = fixture[f"{layer}.bf16_reference"].float().flatten()

    decoded = dequantize_nvfp4_weight(weight, scale, scale_2, torch.float32).flatten()
    assert torch.cosine_similarity(decoded, reference, dim=0).item() > 0.99
    assert (decoded.norm() / reference.norm()).item() == pytest.approx(1.0, abs=0.01)

    # Laying the stored grid out once more makes the unblock hand back the raw grid: a row-major read.
    row_major = stored_layout(scale.float()).to(torch.float8_e4m3fn)
    naive = dequantize_nvfp4_weight(weight, row_major, scale_2, torch.float32).flatten()
    assert torch.cosine_similarity(naive, reference, dim=0).item() < 0.95

    # The packed module keeps the block scales as bytes and has to compute with exactly this weight.
    x = torch.randn(2, 128)
    module = NVFP4Linear(weight, scale, scale_2)
    assert torch.equal(module(x), torch.nn.functional.linear(x, decoded.view(128, 128)))


def test_popped_layers_keep_their_tensors_as_stored_and_leave_the_rest_for_the_fp8_path() -> None:
    fp8_weight = torch.randn(32, 32).to(torch.float8_e4m3fn)
    fp8_scale = torch.tensor(0.5)
    fp8_marker = comfy_quant_marker({"format": "float8_e4m3fn"})
    tekken = torch.randint(0, 256, (64,), dtype=torch.uint8)
    sd = {
        **_zero_layer("blocks.0.mlp"),
        "blocks.0.mlp.input_scale": torch.tensor(1.0),
        "blocks.0.mlp.bias": torch.zeros(128),
        "blocks.0.attn.weight": fp8_weight,
        "blocks.0.attn.weight_scale": fp8_scale,
        "blocks.0.attn.comfy_quant": fp8_marker,
        "tekken_model": tekken,
    }
    weight, scale = sd["blocks.0.mlp.weight"], sd["blocks.0.mlp.weight_scale"]

    payloads = pop_nvfp4_layers(sd)

    assert list(payloads) == ["blocks.0.mlp"]
    assert payloads["blocks.0.mlp"].weight is weight
    assert payloads["blocks.0.mlp"].weight_scale is scale
    # The bias is an ordinary tensor the loader casts; everything else of the layer is gone.
    assert sorted(sd) == [
        "blocks.0.attn.comfy_quant",
        "blocks.0.attn.weight",
        "blocks.0.attn.weight_scale",
        "blocks.0.mlp.bias",
        "tekken_model",
    ]
    # The scaled-fp8 layer, its scale and its marker belong to the fp8 path.
    assert sd["blocks.0.attn.weight"] is fp8_weight
    assert sd["blocks.0.attn.weight_scale"] is fp8_scale
    assert sd["blocks.0.attn.comfy_quant"] is fp8_marker
    assert sd["tekken_model"] is tekken


def test_a_layer_only_the_safetensors_header_names_is_read() -> None:
    """Comfy's Z-Image build names its nvfp4 layers in `_quantization_metadata` and writes no per-tensor marker."""
    sd = _zero_layer("layer")
    del sd["layer.comfy_quant"]

    assert list(pop_nvfp4_layers(sd, header_layers={"layer": {"format": "nvfp4"}})) == ["layer"]


def test_a_layer_nothing_names_as_nvfp4_is_refused_rather_than_read_by_comfy_conventions() -> None:
    """Other producers write the same key names, and nothing guarantees they share ComfyUI's nibble order or
    block-scale layout; a wrong guess loads a model that runs and generates degraded images."""
    sd = _zero_layer("layer")
    del sd["layer.comfy_quant"]

    with pytest.raises(ValueError, match="laid out like nvfp4 that no ComfyUI"):
        pop_nvfp4_layers(sd, header_layers={"other": {"format": "nvfp4"}})
    assert "layer.weight" in sd


def test_an_awq_checkpoint_is_refused_before_anything_is_taken_out() -> None:
    sd = {**_zero_layer("layer"), "layer.pre_quant_scale": torch.ones(64)}

    with pytest.raises(ValueError, match="AWQ"):
        pop_nvfp4_layers(sd)
    assert "layer.weight" in sd


def test_a_packed_weight_without_its_global_scale_is_refused() -> None:
    sd = _zero_layer("layer")
    del sd["layer.weight_scale_2"], sd["layer.comfy_quant"]

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        pop_nvfp4_layers(sd)


@pytest.mark.parametrize("missing", ["weight", "weight_scale"])
def test_a_global_scale_without_its_layer_is_refused_naming_it(missing: str) -> None:
    sd = _zero_layer("layer")
    del sd[f"layer.{missing}"]

    with pytest.raises(ValueError, match=f"nvfp4 layer 'layer': has a weight_scale_2 but no {missing}$"):
        pop_nvfp4_layers(sd)


def test_a_global_scale_on_a_dense_weight_is_refused() -> None:
    sd = _zero_layer("layer")
    sd["layer.weight"] = torch.zeros(128, 64)

    with pytest.raises(ValueError, match="nvfp4 layer 'layer': expected a packed uint8"):
        pop_nvfp4_layers(sd)


def test_a_malformed_layer_is_refused_before_any_layer_is_taken_out() -> None:
    sd = {**_zero_layer("a"), **_zero_layer("b")}
    sd["b.weight_scale"] = torch.ones(128, 8).to(torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="nvfp4 layer 'b': .* does not describe"):
        pop_nvfp4_layers(sd)
    assert "a.weight" in sd


@pytest.mark.parametrize(
    ("tensors", "marker", "message"),
    [
        ("dense", {"format": "nvfp4"}, "nvfp4 layer 'layer' has no weight_scale_2"),
        ("nvfp4", {"format": "float8_e4m3fn"}, "marked 'float8_e4m3fn'"),
    ],
)
def test_a_marker_that_contradicts_its_tensors_is_refused(tensors: str, marker: dict, message: str) -> None:
    sd = {"layer.weight": torch.zeros(128, 64)} if tensors == "dense" else _zero_layer("layer")
    sd["layer.comfy_quant"] = comfy_quant_marker(marker)

    with pytest.raises(ValueError, match=message):
        pop_nvfp4_layers(sd)


@pytest.mark.parametrize(
    ("module", "extra", "message"),
    [
        (None, {}, "names no module"),
        (torch.nn.Linear(64, 256, bias=False), {}, "holds a 128x64 weight"),
        (torch.nn.Linear(64, 128, bias=True), {}, "has no bias"),
        (torch.nn.Linear(64, 128, bias=False), {"proj.bias": torch.zeros(128)}, "has a bias"),
    ],
    ids=["missing_module", "other_shape", "missing_bias", "surplus_bias"],
)
def test_a_layer_the_built_model_cannot_hold_is_refused_at_load(
    module: torch.nn.Module | None, extra: dict[str, torch.Tensor], message: str
) -> None:
    """Once a packed module replaces the model's Linear, a strict load checks the new module's keys and assigns
    its buffers unchecked, so none of these mismatches would stop it; they would surface in the forward, if at
    all."""
    model = torch.nn.Module()
    if module is not None:
        model.proj = module
    sd = {**_zero_layer("proj"), **extra}
    payloads = pop_nvfp4_layers(sd)

    with pytest.raises(ValueError, match=message):
        predict_nvfp4_install_size(model, payloads, torch.float32)
        install_nvfp4_layers(model, sd, payloads, torch.float32)


def test_a_fused_layer_splits_on_whole_tile_rows_and_each_part_keeps_its_own_scales() -> None:
    sd = _nvfp4_tensors("fused", torch.full((256, 64), 2, dtype=torch.uint8), block_scale=1.0, global_scale=0.5)
    # One scale per tile row reads the same tiled as row by row, so each part's scale is known without the layout.
    tile_row_scales = torch.tensor([[1.0], [4.0]]).repeat_interleave(128, dim=0).repeat(1, 4)
    sd["fused.weight_scale"] = tile_row_scales.to(torch.float8_e4m3fn)
    (payload,) = pop_nvfp4_layers(sd).values()

    first, second = split_nvfp4_rows("fused", payload, 2)

    for part, value in ((first, 0.5), (second, 2.0)):
        decoded = dequantize_nvfp4_weight(part.weight, part.weight_scale, part.weight_scale_2, torch.float32)
        assert torch.equal(decoded, torch.full((128, 64), value))
    with pytest.raises(ValueError, match="whole 128-row scale tiles"):
        split_nvfp4_rows("fused", payload, 4)


def test_a_lora_rides_as_a_sidecar_over_the_packed_weight() -> None:
    """Patches cannot be written into packed buffers, so the model has to be recognised as needing sidecars, and
    the wrapper has to hand patches that are not LoRAs the unpacked weight shape to be reshaped to."""
    torch.manual_seed(2)
    positive = torch.randint(0, 2, (128, 64), dtype=torch.bool)
    bias = torch.randn(128)
    model = torch.nn.Module()
    model.proj = _packed_linear(_signed_layer("proj", positive), "proj", bias=bias)
    apply_custom_layers_to_model(model)
    up, down, full = torch.randn(128, 4), torch.randn(4, 64), torch.randn(128, 64)
    lora = ModelPatchRaw(layers={"lora-proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)})
    full_diff = ModelPatchRaw(layers={"lora-proj": FullLayer(weight=full, bias=None)})
    x = torch.randn(3, 64)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(lora, 0.75), (full_diff, 0.5)],
        prefix="lora-",
        dtype=torch.float32,
        force_sidecar_patching=requires_sidecar_patching(model, ModelFormat.Checkpoint),
    ):
        patched = model.proj(x)

    dense = torch.where(positive, 0.5, -0.5)
    expected = torch.nn.functional.linear(x, dense + 0.75 * (up @ down) + 0.5 * full, bias)
    assert torch.allclose(patched, expected, atol=1e-4)
    assert model.proj.get_num_patches() == 0
    assert torch.equal(model.proj(x), torch.nn.functional.linear(x, dense, bias))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures CUDA allocations")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_transient_estimate_covers_what_the_per_forward_decode_allocates(dtype: torch.dtype) -> None:
    """The node reserves this much working memory for the decode, and the model's resident size does not include
    it, so an estimate that falls short lets the first forward compete with weights the cache just placed.

    Measured in requested bytes: peak *allocated* bytes count whole cached blocks, which blocks left over by
    earlier tests in the same process can inflate."""
    device = torch.device("cuda")
    torch.manual_seed(3)
    tensors = _nvfp4_tensors("proj", torch.randint(0, 16, (1024, 4096), dtype=torch.uint8), 2.0, 0.25)
    model = torch.nn.Module()
    model.proj = _packed_linear({k: v.to(device) for k, v in tensors.items()}, "proj")
    model.proj._dequantized_weight(device, dtype)  # Mints the shared byte tables.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_stats()["requested_bytes.all.current"]

    weight = model.proj._dequantized_weight(device, dtype)
    torch.cuda.synchronize()

    measured = torch.cuda.memory_stats()["requested_bytes.all.peak"] - baseline
    estimate = peak_dequant_transient_bytes(model, dtype)
    assert weight.dtype is dtype
    assert 0.95 * estimate <= measured <= estimate


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_buffers_left_in_ram_stream_per_call_and_a_resident_forward_never_waits_on_the_host() -> None:
    """Partial loading leaves some packed layers in RAM, and their forward has to stream them to the input's
    device. A resident forward runs 240 times per Z-Image step, so it must not synchronize with the host."""
    device = torch.device("cuda")
    torch.manual_seed(4)
    tensors = _nvfp4_tensors("proj", torch.randint(0, 16, (128, 64), dtype=torch.uint8), 2.0, 0.25)
    model = torch.nn.Module()
    model.proj = _packed_linear(tensors, "proj", bias=torch.randn(128))
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)
    x = torch.randn(3, 64, device=device, dtype=torch.bfloat16)

    streamed = model.proj(x)
    model.to(device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        resident = model.proj(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert streamed.device == x.device
    assert torch.equal(streamed, resident)
