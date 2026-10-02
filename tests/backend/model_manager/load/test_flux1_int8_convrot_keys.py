"""What the released `int8_tensorwise` FLUX.1 repack actually contains.

The loader suite beside this one runs on a synthetic tiny geometry, which cannot say whether the
*real* file still looks the way the loader assumes. This one reads the captured key layout and
pins the three properties the decode depends on. Each is a thing a re-upload could change without
any loader code moving:

- the marker body, which is where `convrot` and the group size live;
- the group alignment, which is what makes a rotated weight decodable at all;
- the scale layout, which decides whether the scale multiplies rows or columns.
"""

import json

import torch

from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT8_TENSORWISE_FORMAT,
    as_column_scale,
    check_int8_scale_layout,
)
from tests.backend.model_manager.load.state_dicts.flux1_dev_int8_convrot_keys import (
    state_dict_keys as flux1_keys,
)

MARKER = {"format": INT8_TENSORWISE_FORMAT, "per_row": True, "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}

QUANTIZED = sorted(key[: -len(".weight")] for key, (_, dtype) in flux1_keys.items() if dtype == "I8")


def test_the_fixture_is_the_marker_this_suite_assumes() -> None:
    """The recorded blob length is 89 bytes, which is exactly the marker spelled out above.

    A re-capture from a repack that dropped `convrot`, or changed the group size, moves that length
    -- and the loader tests would otherwise keep exercising the rotated case against a file that no
    longer rotates, which is the one mistake that loads cleanly and generates noise.
    """
    lengths = {tuple(shape) for key, (shape, _) in flux1_keys.items() if key.endswith(".comfy_quant")}

    assert lengths == {(len(json.dumps(MARKER).encode("utf-8")),)}


def test_every_quantized_layer_is_marked_and_scaled() -> None:
    # No orphan codes and no orphan scales: the two halves `reject_unmarked_int8_weights` and
    # `reject_foreign_quantization_scales` exist to keep matched.
    assert QUANTIZED, "fixture carries no int8 weights"
    for path in QUANTIZED:
        assert flux1_keys.get(f"{path}.comfy_quant"), path
        assert flux1_keys.get(f"{path}.weight_scale"), path
    scaled = {key[: -len(".weight_scale")] for key in flux1_keys if key.endswith(".weight_scale")}
    assert scaled == set(QUANTIZED)


def test_every_quantized_weight_is_a_whole_number_of_rotation_groups_wide() -> None:
    """`convrot` rotates in groups of 256 along the input dim.

    A layer that is not a multiple of that cannot be derotated at all -- `build_regular_hadamard`
    refuses it -- so this is the property that decides whether the released file is loadable. It
    also explains why the repack leaves `img_in` (64 wide) alone.
    """
    for path in QUANTIZED:
        in_features = flux1_keys[f"{path}.weight"][0][1]
        assert in_features % CONVROT_GROUP_SIZE == 0, f"{path} is {in_features} wide"


def test_the_scales_are_per_output_channel() -> None:
    """`[out, 1]`, not `[out]` and not a block grid.

    Broadcasting aligns trailing dimensions, so an `[out]` scale would multiply along the *input*
    axis instead -- for a square weight that is silent, and this repack has plenty of square ones.
    """
    for path in QUANTIZED:
        weight_shape, _ = flux1_keys[f"{path}.weight"]
        scale_shape, scale_dtype = flux1_keys[f"{path}.weight_scale"]
        assert scale_dtype == "F32", path
        assert scale_shape == [weight_shape[0], 1], path

        # Through the helpers the loader uses, so a layout change is caught as the loader sees it.
        weight = torch.zeros(weight_shape, dtype=torch.int8)
        scale = torch.zeros(scale_shape, dtype=torch.float32)
        check_int8_scale_layout(path, weight, scale)
        assert as_column_scale(weight, scale).shape == (weight_shape[0], 1)


def test_the_repack_leaves_the_embedders_and_the_output_projection_dense() -> None:
    # Not a requirement of the loader -- `Flux` declares no skip patterns and nothing in its forward
    # reads `weight.dtype` -- but it is what the fixture documents, and a re-capture that quantized
    # them would make the loader suite's "dense remainder" assertions meaningless.
    dense = {
        key[: -len(".weight")] for key, (shape, dtype) in flux1_keys.items() if dtype == "BF16" and len(shape) == 2
    }

    for path in ("img_in", "txt_in", "time_in.in_layer", "vector_in.in_layer", "final_layer.linear"):
        assert path in dense, path
    assert "double_blocks.0.img_attn.proj" in dense, "this repack leaves attn.proj dense"
