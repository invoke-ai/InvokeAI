"""Native FLUX.3 checkpoints -> diffusers' FLUX.3 transformer.

The real-layout tests read the key names and shapes captured from the released Klein and Klein Base files and
from Klein's quantized builds. Converted onto diffusers' ``Flux3Transformer2DModel`` (on the meta device), every
parameter must be filled and nothing may be left over, and every quantization scale must land beside its weight --
a renamed key in a re-upload shows up here, not as a block silently missing from a generation.
"""

import accelerate
import pytest
import torch
from diffusers import Flux3Transformer2DModel

from invokeai.backend.flux3.conversion import (
    convert_flux3_layer_entries,
    convert_flux3_native_to_diffusers,
    flux3_native_key_to_diffusers,
    flux3_native_module_to_diffusers,
    flux3_transformer_config,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.flux3.flux3_fixture import TINY_CONFIG, native_flux3_keys
from tests.backend.model_manager.load.state_dicts import (
    flux3_klein_base_keys,
    flux3_klein_fp8r_keys,
    flux3_klein_int8_convrot_keys,
    flux3_klein_keys,
    flux3_klein_nvfp4_bfl_keys,
    flux3_klein_nvfp4_comfy_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import keys_to_meta_state_dict
from tests.fixtures.quantized_payloads import comfy_quant_marker

# The released transformer config (transformer/config.json of both repositories, 2026-10-15).
_RELEASED = {
    "in_channels": 128,
    "cond_in_channels": 128,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 20480,
    "timestep_channels": 256,
    "mlp_ratio": 3.0,
}


@pytest.mark.parametrize(
    ("fixture", "flow_map"), [(flux3_klein_keys, True), (flux3_klein_base_keys, False)], ids=["klein", "base"]
)
def test_the_released_layout_fills_the_diffusers_model_exactly(fixture, flow_map: bool) -> None:
    native = keys_to_meta_state_dict(fixture.state_dict_keys)

    config = flux3_transformer_config(native)
    assert {key: config[key] for key in _RELEASED} == _RELEASED
    assert config["flow_map"] is flow_map
    # The fixture keeps block 0 of each stack, so the depth read from it is 1 everywhere.
    assert (config["num_layers"], config["num_single_layers"]) == (1, 1)

    with accelerate.init_empty_weights():
        model = Flux3Transformer2DModel(**config)
    converted = convert_flux3_native_to_diffusers(dict(native))
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    assert {name: tuple(t.shape) for name, t in converted.items()} == expected


def test_the_fixture_names_are_the_released_ones() -> None:
    """The tiny fixture enumerates the layout itself; it must be the one the released file uses."""
    assert set(native_flux3_keys(1, 1, flow_map=True)) == set(flux3_klein_keys.state_dict_keys)
    assert set(native_flux3_keys(1, 1, flow_map=False)) == set(flux3_klein_base_keys.state_dict_keys)


def test_the_output_adaln_is_reordered_from_shift_scale_to_scale_shift() -> None:
    shift, scale = torch.full((4, 3), 1.0), torch.full((4, 3), 2.0)
    native = {"final_layer.image.adaLN_modulation.1.weight": torch.cat([shift, scale])}

    converted = convert_flux3_native_to_diffusers(native)

    assert torch.equal(converted["norm_out.linear.weight"], torch.cat([scale, shift]))
    assert native == {}, "the input dict is emptied so it holds no second reference to the weights"


def test_a_packed_gguf_adaln_is_unpacked_before_the_halves_swap() -> None:
    import gguf

    shift, scale = torch.full((4, 32), 1.0), torch.full((4, 32), 2.0)
    raw = torch.cat([shift, scale])
    packed = GGMLTensor(
        raw, ggml_quantization_type=gguf.GGMLQuantizationType.F32, tensor_shape=raw.shape, compute_dtype=torch.float32
    )

    converted = convert_flux3_native_to_diffusers({"final_layer.image.adaLN_modulation.1.weight": packed})

    weight = converted["norm_out.linear.weight"]
    assert not isinstance(weight, GGMLTensor)
    assert torch.equal(weight, torch.cat([scale, shift]))


def test_training_only_heads_are_dropped() -> None:
    converted = convert_flux3_native_to_diffusers(
        {
            "repa_projectors.0.weight": torch.zeros(1),
            "repa_projectors.0.weight_scale": torch.ones(1),
            "txt_in.weight": torch.zeros(1),
        }
    )
    assert set(converted) == {"context_embedder.weight"}
    # The quantization entries read under native names go with them, and a fused `qkv`'s describes all three parts.
    marker = {"format": "int8_tensorwise"}
    assert convert_flux3_layer_entries({"repa_projectors.0": marker, "single_blocks.2.qkv": marker}) == {
        f"single_transformer_blocks.2.attn.{projection}": marker for projection in ("to_q", "to_k", "to_v")
    }


@pytest.mark.parametrize(
    ("native", "diffusers"),
    [
        ("content_mode_blocks.image_cond.0.attn_out.weight", "mode_blocks.image_cond.0.attn.to_out.0.weight"),
        # q, k and v have identical shapes, so only their names keep a swap from loading silently.
        ("content_mode_blocks.image.2.q_proj.weight", "mode_blocks.image.2.attn.to_q.weight"),
        ("txt_mode_blocks.1.k_proj.weight", "txt_mode_blocks.1.attn.to_k.weight"),
        ("single_blocks.4.v_proj.weight", "single_transformer_blocks.4.attn.to_v.weight"),
        ("txt_mode_blocks.3.norm.query_norm.scale", "txt_mode_blocks.3.attn.norm_q.weight"),
        ("single_blocks.27.mlp_in.weight", "single_transformer_blocks.27.ff.linear_in.weight"),
        ("early_stream_modulations.txt.lin.weight", "early_stream_modulations.txt.linear.weight"),
        ("ref_time_in.out_layer.weight", "target_timestep_embedder.linear_2.weight"),
        ("emb_in.image_cond.weight", "x_embedders.image_cond.weight"),
    ],
)
def test_key_renames(native: str, diffusers: str) -> None:
    assert flux3_native_key_to_diffusers(native) == diffusers


def test_the_structure_is_read_from_the_weights_not_assumed() -> None:
    """A deeper finetune must load as itself, not as the first N blocks of the released depth."""
    native = {name: torch.empty(1, device="meta") for name in native_flux3_keys(3, 5, flow_map=False)}
    native["emb_in.image.weight"] = torch.empty(128, 32, device="meta")
    native["emb_in.image_cond.weight"] = torch.empty(128, 32, device="meta")
    native["txt_in.weight"] = torch.empty(128, 64, device="meta")
    native["time_in.in_layer.weight"] = torch.empty(128, 32, device="meta")
    native["single_blocks.0.norm.query_norm.scale"] = torch.empty(128, device="meta")
    native["single_blocks.0.mlp_in.weight"] = torch.empty(768, 128, device="meta")

    config = flux3_transformer_config(native)

    assert (config["num_layers"], config["num_single_layers"]) == (3, 5)
    assert (config["num_attention_heads"], config["mlp_ratio"]) == (1, 3.0)


def test_a_gap_in_a_block_stack_is_refused() -> None:
    native = {name: torch.empty(1, device="meta") for name in native_flux3_keys(2, 3, flow_map=False)}
    native = {name: t for name, t in native.items() if not name.startswith("single_blocks.1.")}
    native.update(_shapes_of_tiny())
    with pytest.raises(ValueError, match="missing 'single_blocks.' blocks"):
        flux3_transformer_config(native)


def test_per_mode_stacks_of_different_depth_are_refused() -> None:
    native = {name: torch.empty(1, device="meta") for name in native_flux3_keys(2, 1, flow_map=False)}
    native = {name: t for name, t in native.items() if not name.startswith("content_mode_blocks.image_cond.1.")}
    native.update(_shapes_of_tiny())
    with pytest.raises(ValueError, match="per-mode"):
        flux3_transformer_config(native)


_QUANTIZED_RELEASES = {
    "int8-convrot": flux3_klein_int8_convrot_keys,
    "nvfp4-comfy": flux3_klein_nvfp4_comfy_keys,
    "nvfp4-bfl": flux3_klein_nvfp4_bfl_keys,
    "fp8r": flux3_klein_fp8r_keys,
}
_SIDE_CHANNEL = (".weight_scale_2", ".weight_scale", ".input_scale", ".comfy_quant")


@pytest.mark.parametrize("fixture", _QUANTIZED_RELEASES.values(), ids=_QUANTIZED_RELEASES.keys())
def test_every_side_channel_of_a_quantized_release_lands_beside_its_weight(fixture) -> None:
    """A scale or marker left under a name the model does not have is dropped, and its weight loads off by
    `1/weight_scale` without a word. So each must land beside its own weight, on a Linear of the release geometry,
    and a per-row one must have that weight's rows: the int8 build's fused `qkv` splits three ways, scales included.
    The weights themselves fill the model exactly, and ComfyUI's attention-kernel hints are gone."""
    native = keys_to_meta_state_dict(fixture.state_dict_keys)
    with accelerate.init_empty_weights():
        model = Flux3Transformer2DModel(**flux3_transformer_config(native))
    linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

    converted = convert_flux3_native_to_diffusers(dict(native))

    side_channels = {key: value for key, value in converted.items() if key.endswith(_SIDE_CHANNEL)}
    assert side_channels
    assert set(converted) - set(side_channels) == set(model.state_dict())
    for key, value in side_channels.items():
        module = key.rpartition(".")[0]
        assert module in linears, key
        if value.numel() > 1 and not key.endswith(".comfy_quant"):
            assert value.shape[0] == converted[f"{module}.weight"].shape[0] == linears[module].out_features, key


@pytest.mark.parametrize("scale_shape", [(768, 1), (768,), ()], ids=["per-row", "per-row-1d", "per-tensor"])
def test_a_fused_qkv_splits_into_query_key_and_value_with_its_side_channel(scale_shape: tuple[int, ...]) -> None:
    """Rows in query, key, value order, as Comfy-Org's int8 build stores them; a per-row scale splits with its rows,
    and whatever describes the whole layer -- a per-tensor scale, the activation scale, the marker -- goes to all
    three."""
    rows = torch.arange(768, dtype=torch.float32)
    marker = comfy_quant_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    scale = rows.reshape(scale_shape) if scale_shape else torch.tensor(0.5)
    native = {
        "single_blocks.0.qkv.weight": rows.unsqueeze(1).expand(768, 4).clone(),
        "single_blocks.0.qkv.weight_scale": scale,
        "single_blocks.0.qkv.input_scale": torch.tensor(2.0),
        "single_blocks.0.qkv.comfy_quant": marker,
    }

    converted = convert_flux3_native_to_diffusers(native)

    for index, projection in enumerate(("to_q", "to_k", "to_v")):
        prefix = f"single_transformer_blocks.0.attn.{projection}"
        own_rows = rows[256 * index : 256 * (index + 1)]
        assert torch.equal(converted[f"{prefix}.weight"], own_rows.unsqueeze(1).expand(256, 4))
        expected_scale = own_rows.reshape(256, *scale_shape[1:]) if scale_shape else torch.tensor(0.5)
        assert torch.equal(converted[f"{prefix}.weight_scale"], expected_scale)
        assert torch.equal(converted[f"{prefix}.input_scale"], torch.tensor(2.0))
        assert torch.equal(converted[f"{prefix}.comfy_quant"], marker)
    assert not any(".qkv." in key for key in converted)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("scale_shape", [(4,), (4, 1), ()], ids=["per-row", "per-row-2d", "per-tensor"])
def test_the_output_adalns_weight_scale_moves_with_its_rows(dtype: torch.dtype, scale_shape: tuple[int, ...]) -> None:
    """The codes are swapped as stored, and a per-row scale with them; folded or dequantized later, a scale left
    unswapped would scale every row of the scale half by a row of the shift half."""
    shift, scale = torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)
    row_scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    native = {
        "final_layer.image.adaLN_modulation.1.weight": torch.cat([shift, scale]).to(dtype),
        "final_layer.image.adaLN_modulation.1.weight_scale": row_scales.reshape(scale_shape)
        if scale_shape
        else row_scales[0],
        "final_layer.image.adaLN_modulation.1.input_scale": torch.tensor(0.5),
    }

    converted = convert_flux3_native_to_diffusers(native)

    # Compared as bytes: `torch.equal` is not implemented for float8 on the CPU.
    weight = converted["norm_out.linear.weight"]
    assert weight.dtype is dtype
    assert torch.equal(weight.view(torch.uint8), torch.cat([scale, shift]).to(dtype).view(torch.uint8))
    swapped = torch.tensor([3.0, 4.0, 1.0, 2.0])
    expected = swapped.reshape(scale_shape) if scale_shape else row_scales[0]
    assert torch.equal(converted["norm_out.linear.weight_scale"], expected)
    assert torch.equal(converted["norm_out.linear.input_scale"], torch.tensor(0.5))


@pytest.mark.parametrize("packed", ["nvfp4", "gguf"])
def test_a_packed_fused_qkv_is_refused(packed: str) -> None:
    """nvfp4 packs two codes per byte and GGML whole blocks of a row; neither is split into projections here."""
    import gguf

    if packed == "nvfp4":
        weight = torch.zeros(768, 128, dtype=torch.uint8)
    else:
        weight = GGMLTensor(
            torch.zeros(768, 8 * 34, dtype=torch.uint8),
            ggml_quantization_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=torch.Size([768, 256]),
            compute_dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="packed"):
        convert_flux3_native_to_diffusers({"single_blocks.0.qkv.weight": weight})


@pytest.mark.parametrize(
    ("native", "diffusers"),
    [
        ("content_mode_blocks.image_cond.1.qkv", [f"mode_blocks.image_cond.1.attn.to_{p}" for p in "qkv"]),
        ("txt_mode_blocks.2.qkv", [f"txt_mode_blocks.2.attn.to_{p}" for p in "qkv"]),
        # A norm stores its parameter as `scale`, so its module is found through that name.
        ("single_blocks.3.norm.key_norm", ["single_transformer_blocks.3.attn.norm_k"]),
        ("single_stream_modulations.image.lin", ["single_stream_modulations.image.linear"]),
        ("final_layer.image.adaLN_modulation.1", ["norm_out.linear"]),
    ],
)
def test_module_renames(native: str, diffusers: list[str]) -> None:
    assert flux3_native_module_to_diffusers(native) == diffusers


def _shapes_of_tiny() -> dict[str, torch.Tensor]:
    hidden = TINY_CONFIG["attention_head_dim"] * TINY_CONFIG["num_attention_heads"]
    return {
        "emb_in.image.weight": torch.empty(hidden, 32, device="meta"),
        "emb_in.image_cond.weight": torch.empty(hidden, 32, device="meta"),
        "txt_in.weight": torch.empty(hidden, 64, device="meta"),
        "time_in.in_layer.weight": torch.empty(hidden, 32, device="meta"),
        "single_blocks.0.norm.query_norm.scale": torch.empty(128, device="meta"),
        "single_blocks.0.mlp_in.weight": torch.empty(6 * hidden, hidden, device="meta"),
    }
