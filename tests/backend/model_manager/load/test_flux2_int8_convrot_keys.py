"""An `int8_tensorwise` FLUX.2 checkpoint must survive the BFL -> diffusers rename with its markers.

The scaled-fp8 sibling of this file covers the scales. What is new here is the `comfy_quant` marker,
and it is load-bearing in a way a scale is not: a scale that goes missing leaves a weight quantized
but unscaled, while a *marker* that goes missing means `extract_int8_convrot_markers` claims nothing
at all and the int8 codes are cast to bf16 as raw integers -- a model of small signed integers.

Both of FLUX.2's fused layers are exercised, and they pull in opposite directions. The per-tensor
scale and the marker each describe a whole fused tensor, so `double_blocks.*.img_attn.qkv` -- which
diffusers splits three ways -- has to *copy* both to every projection, while
`single_blocks.*.linear1` stays fused as `to_qkv_mlp_proj` and has to carry both to exactly one.
"""

import json

import torch

from invokeai.backend.model_manager.load.model_loaders.flux2_state_dict_utils import (
    convert_flux2_bfl_to_diffusers,
)
from invokeai.backend.quantization.int8_convrot import (
    extract_int8_convrot_markers,
    reject_unmarked_int8_weights,
)
from tests.backend.model_manager.load.state_dicts.flux2_klein_9b_int8_convrot_keys import (
    state_dict_keys as klein_keys,
)

_DTYPES = {"I8": torch.int8, "U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16}

# What the repack writes, read from the file: no `convrot`, so no rotation to undo.
MARKER = {"format": "int8_tensorwise"}


def _build_state_dict() -> dict[str, torch.Tensor]:
    """The captured layout at exact extents but almost no storage.

    The extents matter -- the fused tensors are split by row -- but the values do not, so each
    tensor is one element expanded to its full shape. The markers are the exception: they carry
    the JSON the loader parses, so they are built for real (and are 29 bytes each).
    """
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in klein_keys.items():
        if key.endswith(".comfy_quant"):
            sd[key] = torch.frombuffer(bytearray(json.dumps(MARKER).encode("utf-8")), dtype=torch.uint8).clone()
            continue
        sd[key] = torch.full((), 2.5 if key.endswith(".weight_scale") else 0.0, dtype=_DTYPES[dtype]).expand(shape)
    return sd


def test_the_marker_this_suite_builds_matches_the_captured_blob_length() -> None:
    """A consistency guard between `MARKER` and the fixture, not a check on production code.

    The captured blobs are 29 bytes, which is exactly `{"format": "int8_tensorwise"}` -- no
    `convrot` flag. If someone re-captures the fixture from a repack that added one, the recorded
    length moves and this fails, which is the prompt to update `MARKER` rather than keep testing
    the unrotated case against a rotated file.
    """
    lengths = {tuple(shape) for key, (shape, _) in klein_keys.items() if key.endswith(".comfy_quant")}

    assert lengths == {(len(json.dumps(MARKER).encode("utf-8")),)}


def test_every_int8_weight_is_still_claimed_after_the_rename() -> None:
    """The end state the loader depends on: no orphaned codes anywhere in the converted dict."""
    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())
    markers = extract_int8_convrot_markers(converted)

    assert markers, "no marker survived the conversion"
    # Raises if any int8 weight is left without one. Checked through the guard the loader uses
    # rather than by recounting, because that guard is what has to stay satisfied.
    reject_unmarked_int8_weights(converted, markers, "FLUX.2")
    assert [k for k in converted if k.startswith(("double_blocks.", "single_blocks."))] == []


def test_a_fused_layer_hands_its_marker_and_scale_to_every_projection() -> None:
    """A per-tensor scale and a marker blob describe the whole fused tensor.

    Splitting either would be wrong and leaving either on the fused path is worse: the projections
    the fusion becomes then carry int8 codes that nothing claims.
    """
    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())
    markers = extract_int8_convrot_markers(converted)

    for fused, parts in (
        ("double_blocks.0.img_attn.qkv", ("to_q", "to_k", "to_v")),
        ("double_blocks.0.txt_attn.qkv", ("add_q_proj", "add_k_proj", "add_v_proj")),
    ):
        source_scale = _build_state_dict()[f"{fused}.weight_scale"]
        destinations = [path for path in markers if path.rsplit(".", 1)[-1] in parts]
        assert len(destinations) == len(parts), f"{fused} -> {destinations}"
        for path in destinations:
            assert markers[path] == MARKER
            assert torch.equal(converted[f"{path}.weight_scale"], source_scale)
            assert converted[f"{path}.weight"].dtype is torch.int8


def test_the_single_block_fusion_is_renamed_rather_than_split() -> None:
    """`single_blocks.linear1` fuses attention *and* MLP, and diffusers keeps it fused.

    So it is the opposite case from the qkv, and worth pinning for that reason: its marker must be
    carried to exactly one destination. It is also the largest quantized layer in the file
    (36864 x 4096), i.e. the one that sets the per-forward dequantization transient the denoise
    node reserves for.
    """
    converted = convert_flux2_bfl_to_diffusers(_build_state_dict())
    markers = extract_int8_convrot_markers(converted)

    fused = "single_transformer_blocks.0.attn.to_qkv_mlp_proj"
    from_single_block = sorted(path for path in markers if path.startswith("single_transformer_blocks.0."))

    # Exactly one destination from `linear1` and one from `linear2` -- no split, no duplicate.
    assert from_single_block == ["single_transformer_blocks.0.attn.to_out", fused]
    assert markers[fused] == MARKER
    assert converted[f"{fused}.weight"].shape == torch.Size(klein_keys["single_blocks.0.linear1.weight"][0])
    assert converted[f"{fused}.weight"].dtype is torch.int8
