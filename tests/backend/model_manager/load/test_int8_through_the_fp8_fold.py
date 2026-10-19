"""An int8_tensorwise layer reaching a fold that only applies the scale.

The two schemes share a key layout -- a `.weight` beside a per-output-channel `.weight_scale` -- so
a loader with no int8 branch folds one as if it were the other. The scale even goes down the right
axis, so the result has the weight's shape, its dtype and roughly its magnitude. What is missing is
the inverse Hadamard rotation `convrot` applies along the input dim, and without it the weight bears
no relation to the one the checkpoint encodes.

The distinction that decides the guard: with `convrot` *off*, `codes * scale` is the complete
decode, and the fold produces exactly what the int8 path would. Refusing on the int8 dtype alone
would break that build. So the marker is what is read, and the two cases are pinned apart here.
"""

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.comfyui_state_dict_utils import _dequantize_comfyui_fp8
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

CPU = torch.device("cpu")
ROTATED = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}
UNROTATED = {"format": "int8_tensorwise", "convrot": False}


def _int8_layer(marker: dict | None, *, rotate: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One int8 layer, and the weight a correct loader recovers from it."""
    torch.manual_seed(0)
    original = torch.randn(64, CONVROT_GROUP_SIZE)
    if rotate:
        payload = quantize_convrot(original)
        codes, scale = payload.codes, payload.scale
    else:
        scale = original.abs().amax(dim=1, keepdim=True) / 127.0
        codes = torch.clamp(torch.round(original / scale), -128, 127).to(torch.int8)
        scale = scale.float()

    state_dict = {"lin.weight": codes, "lin.weight_scale": scale}
    if marker is not None:
        state_dict["lin.comfy_quant"] = comfy_quant_marker(marker)
    expected = Int8ConvrotLinear(codes, scale, convrot=rotate)._dequantized_weight(CPU, torch.float32)
    return state_dict, expected


def test_a_rotated_layer_is_refused_rather_than_folded() -> None:
    """Measured on this layer: folded without the derotation it correlates with the stored weight at
    0.07, against 0.9999 for the real decode -- at the same shape, dtype and magnitude."""
    state_dict, _ = _int8_layer(ROTATED, rotate=True)

    with pytest.raises(ValueError, match="quantized with convrot"):
        _dequantize_comfyui_fp8(state_dict, torch.float32)


def test_what_the_fold_produces_on_a_rotated_layer_is_not_the_stored_weight() -> None:
    """The measurement the refusal rests on, taken through the fold itself rather than recomputed.

    The payload is rotated but its marker says `convrot: false`, which is the one way to get a
    rotated layer past the guard and see what the fold does with it. A checkpoint cannot lie like
    that by accident -- the point is to hold the number the guard exists for against the real code
    path, so that a fold which silently stopped applying the scale, or started skipping int8, is
    caught here too.
    """
    state_dict, correct = _int8_layer(UNROTATED, rotate=True)

    assert _dequantize_comfyui_fp8(state_dict, torch.float32) == 1
    folded = state_dict["lin.weight"]

    assert folded.shape == correct.shape and folded.dtype is correct.dtype
    similarity = torch.nn.functional.cosine_similarity(folded.flatten(), correct.flatten(), dim=0)
    assert similarity < 0.2, f"the rotation would have to matter for the guard to be worth having ({similarity})"


def test_an_unrotated_layer_still_folds_bit_identically() -> None:
    """The half that must keep working. `convrot: false` makes `codes * scale` the whole decode, so
    a build like that loads correctly through this fold and a dtype-only guard would break it."""
    state_dict, expected = _int8_layer(UNROTATED, rotate=False)

    assert _dequantize_comfyui_fp8(state_dict, torch.float32) == 1
    assert torch.equal(state_dict["lin.weight"], expected)


def test_an_int8_weight_with_no_marker_is_refused() -> None:
    """Nothing says whether it was rotated. Read unrotated, a rotated weight loads cleanly and
    generates noise -- and every published int8 build marks every one of its int8 weights, so an
    unmarked one is not a build this can reason about."""
    state_dict, _ = _int8_layer(None, rotate=True)

    with pytest.raises(ValueError, match="no int8_tensorwise marker"):
        _dequantize_comfyui_fp8(state_dict, torch.float32)


def test_a_scaled_fp8_layer_is_untouched_by_the_check() -> None:
    """The population this fold exists for. The guard keys on the int8 dtype, so an fp8 layer never
    reaches it -- asserted, because a guard that fired here would break every Wan and Qwen-Image
    scaled-fp8 checkpoint."""
    payload = quantize_scaled_fp8(torch.randn(8, 16))
    state_dict = {"lin.weight": payload.codes, "lin.weight_scale": payload.scale}

    assert _dequantize_comfyui_fp8(state_dict, torch.float32) == 1
    assert torch.equal(state_dict["lin.weight"], payload.dequantized)


def test_the_other_scale_spelling_is_refused_too() -> None:
    """Comfy's Qwen2.5-VL producer writes `.scale_weight`. The weight key is derived from whichever
    suffix matched, so a guard that only recovered the path for one spelling would let the other
    through -- into the fold, unrotated."""
    state_dict, _ = _int8_layer(ROTATED, rotate=True)
    state_dict["lin.scale_weight"] = state_dict.pop("lin.weight_scale")

    with pytest.raises(ValueError, match="quantized with convrot"):
        _dequantize_comfyui_fp8(state_dict, torch.float32)


def test_a_header_entry_is_not_taken_as_evidence_that_a_layer_is_safe() -> None:
    """The safetensors header names formats too, but it cannot carry `convrot` or the group size --
    which is why `flux.py` gives the per-tensor marker precedence. An earlier version of this guard
    let a header hint override the marker, so a header-named rotated build walked straight through
    into the unrotated fold. The guard reads the marker and nothing else; the header is simulated
    here by the only thing that could stand in for it, a marker of the wrong format.
    """
    state_dict, _ = _int8_layer({"format": "float8_e4m3fn"}, rotate=True)

    with pytest.raises(ValueError, match="no int8_tensorwise marker"):
        _dequantize_comfyui_fp8(state_dict, torch.float32)
