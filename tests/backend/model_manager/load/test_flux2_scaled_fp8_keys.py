"""A scaled-fp8 FLUX.2 checkpoint must survive the BFL -> diffusers rename with its scales.

FLUX.2 is the hard case of the scaled-fp8 loaders: unlike FLUX.1 the keys are renamed, and the
fused `qkv` is split into three projections. Both steps can lose a scale silently -- the weights
then stay quantized but unscaled, off by `1/weight_scale`, with nothing logged.
"""

import torch

from invokeai.backend.model_manager.load.model_loaders.flux2_state_dict_utils import (
    convert_flux2_bfl_to_diffusers,
    remap_flux2_layer_paths,
)
from invokeai.backend.quantization.fp8_scaled import FP8_DTYPE, extract_fp8_scaled_layers
from tests.backend.model_manager.load.state_dicts.flux2_klein_4b_scaled_fp8_keys import (
    state_dict_keys as klein_keys,
)

_DTYPES = {"F8_E4M3": FP8_DTYPE, "F32": torch.float32, "BF16": torch.bfloat16}


def _build_state_dict() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in klein_keys.items():
        torch_dtype = _DTYPES[dtype]
        if torch_dtype is FP8_DTYPE:
            sd[key] = torch.zeros(shape, dtype=torch.float32).to(FP8_DTYPE)
        elif key.endswith((".weight_scale", ".input_scale")):
            # 1.0 is the placeholder `_usable_input_scale` rejects, so it must not be used here.
            sd[key] = torch.full(shape, 2.5, dtype=torch_dtype)
        else:
            sd[key] = torch.zeros(shape, dtype=torch_dtype)
    return sd


def test_no_scale_is_left_on_a_bfl_path() -> None:
    """Every scale must land next to the weight it belongs to, under its diffusers name."""
    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())

    scales = [k for k in converted if k.endswith((".weight_scale", ".input_scale"))]
    assert scales, "fixture carries no scales"
    orphans = [k for k in scales if k.rsplit(".", 1)[0] + ".weight" not in converted]
    assert orphans == []
    assert [k for k in converted if k.startswith(("double_blocks.", "single_blocks."))] == []


def test_the_fused_qkv_scalar_is_copied_to_all_three_projections() -> None:
    """A per-tensor scale describes the whole fused tensor, so each third inherits it unchanged."""
    fused = [k for k in klein_keys if k.endswith(".qkv.weight")]
    assert fused, "fixture carries no fused qkv weight"

    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())

    for group in (("to_q", "to_k", "to_v"), ("add_q_proj", "add_k_proj", "add_v_proj")):
        for name in group:
            key = f"transformer_blocks.0.attn.{name}.weight_scale"
            assert key in converted, f"missing {key}"
            assert converted[key].shape == torch.Size([]), "a scalar scale must not be split"
            assert torch.equal(converted[key], torch.tensor(2.5))


def test_a_scale_never_overwrites_its_own_weight() -> None:
    """The block renames are substring tests, so `...proj.weight_scale` matches `...proj.weight`.

    Routed through the weight converter, the scale would be written to the weight's destination key
    and replace it. The weight must still be the fp8 tensor afterwards, not an f32 scalar.
    """
    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())

    weight = converted["transformer_blocks.0.attn.to_out.0.weight"]
    assert weight.dtype is FP8_DTYPE
    assert weight.dim() == 2


def test_every_quantized_linear_is_recognized_after_the_rename() -> None:
    sd = convert_flux2_bfl_to_diffusers(_build_state_dict())
    fp8_weights = {k for k, v in sd.items() if v.dtype is FP8_DTYPE and k.endswith(".weight")}

    layers = extract_fp8_scaled_layers(sd)

    assert {f"{path}.weight" for path in layers} == fp8_weights
    assert all(layer.input_scale is not None for layer in layers.values())


def test_layer_hints_are_renamed_one_to_many_for_the_fused_qkv() -> None:
    """Hints name layers in the BFL scheme; the scales are read after the rename.

    Without remapping, `full_precision_matrix_mult` matches nothing and is silently ignored.
    """
    mapping = remap_flux2_layer_paths(["double_blocks.0.img_attn.qkv", "double_blocks.0.img_attn.proj"])

    assert mapping["double_blocks.0.img_attn.qkv"] == [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    ]
    assert mapping["double_blocks.0.img_attn.proj"] == ["transformer_blocks.0.attn.to_out.0"]


def test_a_per_channel_scale_is_split_like_the_weight() -> None:
    """Not present in this checkpoint, but the split must not assume every scale is a scalar."""
    sd = {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(9, 2, dtype=torch.float32).to(FP8_DTYPE),
        "double_blocks.0.img_attn.qkv.weight_scale": torch.arange(9, dtype=torch.float32),
    }

    converted = convert_flux2_bfl_to_diffusers(sd)

    assert torch.equal(converted["transformer_blocks.0.attn.to_q.weight_scale"], torch.arange(0, 3.0))
    assert torch.equal(converted["transformer_blocks.0.attn.to_k.weight_scale"], torch.arange(3, 6.0))
    assert torch.equal(converted["transformer_blocks.0.attn.to_v.weight_scale"], torch.arange(6, 9.0))
