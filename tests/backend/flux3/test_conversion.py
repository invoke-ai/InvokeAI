"""Native FLUX.3 checkpoints -> the vendored diffusers transformer.

The real-layout tests read the key names and shapes captured from the released Klein and Klein Base files.
Converted onto the vendored module (on the meta device), every parameter must be filled and nothing may be
left over -- a renamed key in a re-upload shows up here, not as a block silently missing from a generation.
"""

import accelerate
import pytest
import torch

from invokeai.backend.flux3 import Flux3Transformer2DModel
from invokeai.backend.flux3.conversion import (
    convert_flux3_native_to_diffusers,
    flux3_native_key_to_diffusers,
    flux3_transformer_config,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.flux3.flux3_fixture import TINY_CONFIG, native_flux3_keys
from tests.backend.model_manager.load.state_dicts import flux3_klein_base_keys, flux3_klein_keys

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


def _meta_state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    return {name: torch.empty(shape, device="meta") for name, (shape, _dtype) in keys.items()}


@pytest.mark.parametrize(
    ("fixture", "flow_map"), [(flux3_klein_keys, True), (flux3_klein_base_keys, False)], ids=["klein", "base"]
)
def test_the_released_layout_fills_the_vendored_model_exactly(fixture, flow_map: bool) -> None:
    native = _meta_state_dict(fixture.state_dict_keys)

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
        {"repa_projectors.0.weight": torch.zeros(1), "txt_in.weight": torch.zeros(1)}
    )
    assert set(converted) == {"context_embedder.weight"}


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
