"""Qwen-Image-2.1's single-file layouts reach diffusers' modules: the fused MLP split and the ComfyUI VAE rename."""

import accelerate
import gguf
import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.qwen_image_2_1.checkpoint_layout import (
    convert_comfy_vae_to_diffusers,
    count_transformer_blocks,
    fit_to_module_shapes,
    is_comfy_vae_layout,
    split_fused_gate_up,
    split_fused_layer_names,
)
from tests.backend.model_manager.load.state_dicts import (
    qwen_image_2_1_int8_convrot_keys,
    qwen_image_2_1_vae_comfyui_keys,
)


def _meta_state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    # Full extents on the meta device: the split reads row counts, so the shapes have to be the real ones.
    return {name: torch.empty(shape, device="meta") for name, (shape, _) in keys.items()}


def test_comfyui_vae_lands_on_every_diffusers_parameter() -> None:
    sd = _meta_state_dict(qwen_image_2_1_vae_comfyui_keys.state_dict_keys)
    assert is_comfy_vae_layout(sd)

    converted = convert_comfy_vae_to_diffusers(sd)
    with accelerate.init_empty_weights():
        model = AutoencoderKLQwenImage21()
    fit_to_module_shapes(converted, model)

    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    assert {name: tuple(t.shape) for name, t in converted.items()} == expected


def test_an_unknown_comfyui_vae_key_is_refused_rather_than_dropped() -> None:
    sd = _meta_state_dict(qwen_image_2_1_vae_comfyui_keys.state_dict_keys)
    sd["decoder.upsamples.0.extra.weight"] = torch.empty(1, device="meta")
    with pytest.raises(ValueError, match="decoder.upsamples.0.extra.weight"):
        convert_comfy_vae_to_diffusers(sd)


def test_the_int8_fused_mlp_splits_into_diffusers_linears_with_scale_and_marker() -> None:
    sd = split_fused_gate_up(_meta_state_dict(qwen_image_2_1_int8_convrot_keys.state_dict_keys))
    block = "transformer_blocks.0.img_mlp"

    assert not [k for k in sd if ".gate_up." in k]
    for half in ("gate_layer", "proj"):
        assert tuple(sd[f"{block}.{half}.weight"].shape) == (12288, 4096)
        assert tuple(sd[f"{block}.{half}.weight_scale"].shape) == (12288, 1)
        assert tuple(sd[f"{block}.{half}.comfy_quant"].shape) == (72,)

    # Every weight now names a parameter of a diffusers transformer as deep as the fixture.
    with accelerate.init_empty_weights():
        model = QwenImage21Transformer2DModel(num_layers=count_transformer_blocks(sd))
    weights = {k for k in sd if not k.endswith((".weight_scale", ".comfy_quant"))}
    assert weights == set(model.state_dict())


def test_the_split_keeps_rows_and_per_row_scales_and_shares_per_tensor_sidecars() -> None:
    rows, cols = 6, 4
    weight = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    sd = {
        "transformer_blocks.0.img_mlp.gate_up.weight": weight,
        "transformer_blocks.0.img_mlp.gate_up.weight_scale": torch.arange(rows, dtype=torch.float32).reshape(rows, 1),
        "transformer_blocks.0.img_mlp.gate_up.input_scale": torch.tensor(0.5),
        "transformer_blocks.0.img_mlp.gate_up.comfy_quant": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "transformer_blocks.0.img_mlp.out.weight": torch.ones(cols, rows // 2),
    }
    out = split_fused_gate_up(sd)

    assert torch.equal(out["transformer_blocks.0.img_mlp.gate_layer.weight"], weight[:3])
    assert torch.equal(out["transformer_blocks.0.img_mlp.proj.weight"], weight[3:])
    assert out["transformer_blocks.0.img_mlp.gate_layer.weight_scale"].flatten().tolist() == [0, 1, 2]
    assert out["transformer_blocks.0.img_mlp.proj.weight_scale"].flatten().tolist() == [3, 4, 5]
    for half in ("gate_layer", "proj"):
        assert out[f"transformer_blocks.0.img_mlp.{half}.input_scale"].item() == 0.5
        assert out[f"transformer_blocks.0.img_mlp.{half}.comfy_quant"].tolist() == [1, 2, 3]
    assert out["transformer_blocks.0.img_mlp.out.weight"] is sd["transformer_blocks.0.img_mlp.out.weight"]


def test_a_ggml_split_dequantizes_to_the_halves_of_the_whole() -> None:
    rows, cols = 8, 64
    values = np.random.default_rng(0).standard_normal((rows, cols)).astype(np.float32)
    qtype = gguf.GGMLQuantizationType.Q8_0
    packed = torch.from_numpy(gguf.quants.quantize(values, qtype))
    fused = GGMLTensor(packed, qtype, torch.Size((rows, cols)), torch.float32)

    out = split_fused_gate_up({"transformer_blocks.0.img_mlp.gate_up.weight": fused})
    gate = out["transformer_blocks.0.img_mlp.gate_layer.weight"]
    up = out["transformer_blocks.0.img_mlp.proj.weight"]

    whole = fused.get_dequantized_tensor()
    assert isinstance(gate, GGMLTensor) and gate.shape == (rows // 2, cols)
    assert torch.equal(gate.get_dequantized_tensor(), whole[: rows // 2])
    assert torch.equal(up.get_dequantized_tensor(), whole[rows // 2 :])


def test_header_hints_for_the_fused_layer_apply_to_both_halves() -> None:
    hints = split_fused_layer_names({"transformer_blocks.3.img_mlp.gate_up": "hint", "img_in": "other"})
    assert hints == {
        "transformer_blocks.3.img_mlp.gate_layer": "hint",
        "transformer_blocks.3.img_mlp.proj": "hint",
        "img_in": "other",
    }


def test_a_gap_in_the_blocks_is_refused() -> None:
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        count_transformer_blocks(
            {"transformer_blocks.0.attn.to_q.weight": 0, "transformer_blocks.2.attn.to_q.weight": 0}
        )
