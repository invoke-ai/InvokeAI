"""What the released MXFP8 Krea-2 build actually contains, and whether the synthetic payload matches.

The MXFP8 decode is built on four assumptions about the file: the scale is `uint8`, it carries one
entry per 32 weight elements, it is a whole number of cuBLAS tiles, and the layer says `mxfp8`
somewhere. Until this capture all four rested on the same synthetic construction that the tests
build, so the tests and the decode agreed without either being checked against a producer.

This is the one reachable MXFP8 file, and its weights cannot be checked in (`license: other`), so
what is recorded is the layout. Four cells read it directly; a fifth takes the block width out of it
and holds `MX_BLOCK_SIZE` -- the number every synthetic MXFP8 payload is built from -- against it.
"""

import torch

from invokeai.backend.quantization.block_scale_tiles import check_tile_layout
from invokeai.backend.quantization.fp8_scaled import MXFP8_FORMAT
from tests.backend.model_manager.load.state_dicts import krea2_turbo_mxfp8_keys as fixture
from tests.fixtures.quantized_payloads import MX_BLOCK_SIZE, mxfp8_tensors

KEYS = fixture.state_dict_keys
QUANTIZED = sorted(
    key[: -len(".weight")] for key, (_shape, dtype) in KEYS.items() if dtype == "F8_E4M3" and key.endswith(".weight")
)


def test_every_grid_is_one_uint8_entry_per_thirty_two_weight_elements() -> None:
    """The block width the decode derives from the shapes. A build that changed it would be decoded
    at the wrong granularity -- and `expand_weight_scale` would still widen the grid to the weight,
    so it would load and generate noise rather than raise."""
    assert QUANTIZED
    for path in QUANTIZED:
        (rows, columns), _dtype = KEYS[f"{path}.weight"]
        (scale_rows, blocks), scale_dtype = KEYS[f"{path}.weight_scale"]

        assert scale_dtype == "U8", path
        assert scale_rows == rows, path
        assert columns == blocks * MX_BLOCK_SIZE, path


def test_every_grid_is_a_whole_number_of_cublas_tiles() -> None:
    """`check_tile_layout` is what the de-swizzle requires, and off-tile it cannot be expressed --
    so this is the property that decides whether the released file is decodable at all."""
    for path in QUANTIZED:
        (rows, _columns), _dtype = KEYS[f"{path}.weight"]
        (_scale_rows, blocks), _scale_dtype = KEYS[f"{path}.weight_scale"]

        check_tile_layout(rows, blocks)


def test_the_header_names_every_quantized_layer_and_no_marker_does() -> None:
    """One transport, not two: the header names them and there is no `.comfy_quant` in the file.

    The decode refuses a `uint8` grid that nothing names, so for this build the header *is* the
    evidence -- a re-upload that dropped `_quantization_metadata` would make the file unloadable
    rather than mis-loaded, and this is the cell that would say why.
    """
    assert set(fixture.layer_hints) == set(QUANTIZED)
    assert {hint["format"] for hint in fixture.layer_hints.values()} == {MXFP8_FORMAT}
    assert not any(key.endswith(".comfy_quant") for key in KEYS)


def test_no_hint_carries_a_block_size() -> None:
    """Which is why the decode reads the width from the shapes and lets the marker only cross-check
    it. A decode that required the field would refuse this build outright."""
    assert not any("block_size" in hint for hint in fixture.layer_hints.values())


def test_the_block_width_the_builder_uses_is_the_one_the_released_build_stores() -> None:
    """The one number the synthetic payload and the real file have to agree on.

    `MX_BLOCK_SIZE` is what every MXFP8 cell builds its grid from, and `decode_mx_block_scales` infers
    the same width from the shapes it is given. Both sides being our own constant is how a scheme gets
    tested against itself, so the width is read out of the capture here and our constant is checked
    against it -- 40 real layers, five distinct geometries.

    The second half pins something separable: `mxfp8_tensors`' default `block_size` is a parameter that
    can be edited away from `MX_BLOCK_SIZE`, and then every synthetic grid would be built at a width
    the file does not use while the first assertion still passed.

    Deliberately not repeated here: that a distinct grid survives the de-swizzle. A constant grid is
    invariant under every permutation, so it cannot see a tile error at all; that is
    `TestMxfp8.test_the_exponents_are_decoded_and_unswizzled`, which uses varying exponents.
    """
    widths = set()
    for path in QUANTIZED:
        (_rows, columns), _dtype = KEYS[f"{path}.weight"]
        (_scale_rows, blocks), _scale_dtype = KEYS[f"{path}.weight_scale"]
        # Exact rather than floor division, which would let a remainder through: 6145 columns over
        # 192 blocks still floors to 32.
        assert columns % blocks == 0, path
        widths.add(columns // blocks)

    assert widths == {MX_BLOCK_SIZE}

    tensors, _expected = mxfp8_tensors("lin", torch.full((128, 4), 127))
    weight, scale = tensors["lin.weight"], tensors["lin.weight_scale"]

    assert weight.shape[1] // scale.shape[1] == MX_BLOCK_SIZE
