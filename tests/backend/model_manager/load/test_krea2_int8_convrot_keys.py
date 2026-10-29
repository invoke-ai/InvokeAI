"""What the released `int8_tensorwise` Krea-2 Turbo build actually contains.

The Krea-2 loader suite beside this one runs on synthetic tiny geometry, which can say whether the
loader handles the shape it was given and cannot say whether the released file still has that shape.
These cells read the captured layout and pin the properties the decode depends on -- each one a
thing a re-upload could change without a line of loader code moving.

Krea-2 is the richest of the captured layouts because its keys are native: nothing here reaches the
module tree without `_convert_krea2_native_to_diffusers` running first, so every property below has
to survive a rename as well as being true in the file.
"""

import json

import torch

from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT8_TENSORWISE_FORMAT,
    check_int8_scale_layout,
    parse_comfy_quant_bytes,
)
from tests.backend.model_manager.load.state_dicts import krea2_turbo_int8_convrot_keys as fixture

KEYS = fixture.state_dict_keys
QUANTIZED = sorted(
    # `.endswith`, because the bare `scaled_fp8` marker key carries the same dtype and is not a weight.
    key[: -len(".weight")]
    for key, (_shape, dtype) in KEYS.items()
    if dtype == "I8" and key.endswith(".weight")
)


def test_the_recorded_marker_is_the_rotated_one() -> None:
    """`convrot` and its group size live in the marker and in nothing else.

    A header entry cannot carry them -- measured: the header spelling is `{"format":
    "int8_tensorwise"}` with no room for a rotation flag. So a build that dropped `convrot` would
    still declare `int8_tensorwise`, still load, and derotate nothing. This is the cell that notices.
    """
    assert QUANTIZED
    expected = {"format": INT8_TENSORWISE_FORMAT, "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}

    assert set(fixture.markers) == set(QUANTIZED)
    assert all(marker == expected for marker in fixture.markers.values())


def test_the_recorded_marker_is_what_the_production_parser_reads() -> None:
    """The fixture records the decoded marker; the loader decodes the bytes itself. Pinning the two
    together is what keeps a capture from drifting into a hand-written expectation."""
    blob = json.dumps(next(iter(fixture.markers.values()))).encode("utf-8")

    assert parse_comfy_quant_bytes(blob)["convrot_groupsize"] == CONVROT_GROUP_SIZE
    assert [shape for key, (shape, _dtype) in KEYS.items() if key.endswith(COMFY_QUANT_SUFFIX)][0] == [len(blob)]


def test_every_int8_weight_is_marked_and_scaled_and_nothing_else_is() -> None:
    """The two halves `reject_unmarked_int8_weights` and `reject_foreign_quantization_scales` exist
    to keep matched. An orphan either way is a file one of them would refuse."""
    marked = {key[: -len(COMFY_QUANT_SUFFIX)] for key in KEYS if key.endswith(COMFY_QUANT_SUFFIX)}
    scaled = {key[: -len(".weight_scale")] for key in KEYS if key.endswith(".weight_scale")}

    assert marked == set(QUANTIZED)
    assert scaled == set(QUANTIZED)


def test_the_scales_are_per_output_channel_and_shaped_to_multiply_rows() -> None:
    """`[out, 1]`, not `[out]`. Broadcasting aligns trailing dimensions, so a `[out]` scale on an
    `[out, in]` weight scales the input axis -- a shape error on a rectangular weight and, on a
    square one like `attn.wo` here, a model that loads and generates subtly wrong images.

    Run through `check_int8_scale_layout` rather than compared to a literal, so the cell answers the
    question the loader asks. The input extent is a stand-in: the layout check reads the row axis
    and the real widths here reach 16384, which would cost 100 MB of zeros to say the same thing.
    """
    for path in QUANTIZED:
        weight_shape, _dtype = KEYS[f"{path}.weight"]
        scale_shape, scale_dtype = KEYS[f"{path}.weight_scale"]

        assert scale_shape == [weight_shape[0], 1], path
        assert scale_dtype == "F32", path

        rows = min(weight_shape[0], 8)
        # The scale's trailing axes come from the file, so a recorded block grid would reach the
        # helper rather than being stopped by the literal above.
        check_int8_scale_layout(path, torch.zeros(rows, 4), torch.zeros(rows, *scale_shape[1:]))


def test_every_quantized_weight_is_a_whole_number_of_rotation_groups_wide() -> None:
    """`convrot` rotates along the input dim in groups of 256, and `build_regular_hadamard` refuses
    a width that is not a multiple of it. This is the property that decides loadability at all."""
    for path in QUANTIZED:
        weight_shape, _dtype = KEYS[f"{path}.weight"]

        assert len(weight_shape) == 2, path
        assert weight_shape[1] % CONVROT_GROUP_SIZE == 0, path


def test_the_two_keys_that_defeat_a_probe_are_here_and_are_not_quantized() -> None:
    """Both are real keys in this file, and both break the assumption that a module's parameter is
    called `weight`: the modulation table is stored at `mod.lin`, and the text-MLP norm at
    `txtmlp.0.scale`. A probe that asked the converter about `<module>.weight` answered with the
    module's own renamed prefix instead of its destination, and the scale was dropped at INFO.

    Neither is quantized in this build, which is why the defect stayed latent -- and why the cell
    pins their presence rather than their marker.
    """
    assert KEYS["blocks.0.mod.lin"][1] == "BF16"
    assert "blocks.0.mod.lin.weight" not in KEYS
    assert KEYS["txtmlp.0.scale"][1] == "F32"
    assert "txtmlp.0.weight" not in KEYS


def test_the_build_declares_its_scheme_only_per_tensor() -> None:
    """No `__metadata__` at all -- so for this file the per-tensor marker is not a second opinion,
    it is the only one. A loader reading the header alone finds nothing and folds the codes."""
    assert not hasattr(fixture, "layer_hints")
