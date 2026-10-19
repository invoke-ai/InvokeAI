"""Do the shared checkpoint payloads say what the production decoders hear?

`tests/fixtures/quantized_payloads.py` is now the source of the quantized layers in a dozen test
files. That makes it a load-bearing statement about what ComfyUI writes, and a statement that is
wrong in a self-consistent way is the worst case: every test that builds a layer with it and asserts
against its own arithmetic stays green while agreeing about the wrong encoding.

So each payload is handed to the code that reads real checkpoints, and the weight that comes back is
compared with the one the payload says it stored. What that catches is the half the builders cannot
check themselves -- nibble order, which tensor the scale multiplies, the axis it is broadcast along,
the key spellings -- because that half is decided by the decoder, which was written from the format
rather than from these fixtures.

A round trip cannot see everything, though. It is invariant to the two constants the builders make
the most noise about, `INT8_LEVELS` and `FP8_E4M3_MAX`: halve either and the codes and the scale
move in opposite directions and cancel. Those are pinned separately below, against what torch says
the dtype's range is rather than against the fixture's own numbers.
"""

import pytest
import torch

from invokeai.backend.quantization.fp8_scaled import dequantize_fp8_scaled, extract_fp8_scaled_layers
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.quantized_payloads import mxfp8_tensors, nvfp4_signed_tensors, quantize_convrot, quantize_scaled_fp8

CPU = torch.device("cpu")


def test_the_convrot_payload_dequantizes_to_the_weight_it_claims() -> None:
    """Driven through `Int8ConvrotLinear`, which owns the un-rotation.

    The rotation is the part a builder cannot verify alone: codes and scale of the right shape and
    magnitude come out either way, and a payload that rotated in the wrong direction -- or not at
    all -- would still look like a plausible int8 layer to every shape check in the suite.
    """
    torch.manual_seed(0)
    original = torch.randn(8, CONVROT_GROUP_SIZE)
    payload = quantize_convrot(original)

    linear = Int8ConvrotLinear(payload.codes, payload.scale, convrot=True)
    decoded = linear._dequantized_weight(CPU, torch.float32)

    assert torch.allclose(decoded, payload.dequantized, atol=1e-4)
    # And close to the weight that went in, which is what makes the payload a useful stand-in at all.
    assert torch.corrcoef(torch.stack([decoded.flatten(), original.flatten()]))[0, 1] > 0.999


def test_the_convrot_codes_fill_the_int8_range_without_saturating_it() -> None:
    """What the round trip cannot see: the number of levels the scale is derived from.

    Halve `INT8_LEVELS` and every code halves while the scale doubles, so the decoded weight is
    unchanged and every other test here still passes -- the checkpoint would simply be throwing
    away a bit of precision per weight, silently. The statement that does pin it is about the range:
    the extreme weight lands on int8's largest magnitude, and nothing saturates against it.
    """
    torch.manual_seed(1)
    payload = quantize_convrot(torch.randn(8, CONVROT_GROUP_SIZE))
    codes = payload.codes.to(torch.int32)

    top = torch.iinfo(torch.int8).max
    # Per row, because the scale is per output channel: each row's own extreme lands on the top code.
    assert torch.equal(codes.abs().amax(dim=1), torch.full((codes.shape[0],), top)), codes.abs().amax(dim=1)
    # `-128` exists in int8 but the exporters leave it unused to keep the range symmetric. Too few
    # levels would pile many weights onto the extremes instead of only each row's largest.
    assert codes.min().item() > -128
    assert (codes.abs() == top).sum().item() <= 2 * codes.shape[0]


def test_the_convrot_scale_is_one_per_output_channel_and_reaches_its_own_row() -> None:
    """A `[out, 1]` scale is what `as_column_scale` broadcasts down the output axis. Built here with
    rows of deliberately different magnitudes and a non-square weight, so a scale taken over the
    wrong axis has the wrong shape, and one applied to the wrong axis puts each row's magnitude on
    another row. Driven through the decoder rather than asserted against the fixture's own arithmetic.
    """
    original = torch.zeros(4, CONVROT_GROUP_SIZE)
    for row in range(4):
        original[row] = torch.full((CONVROT_GROUP_SIZE,), 10.0**row)
    payload = quantize_convrot(original)

    assert payload.scale.shape == (4, 1)
    decoded = Int8ConvrotLinear(payload.codes, payload.scale, convrot=True)._dequantized_weight(CPU, torch.float32)

    magnitudes = decoded.abs().amax(dim=1)
    assert torch.all(magnitudes[:-1] < magnitudes[1:]), f"rows did not keep their own scale: {magnitudes}"


def test_the_scaled_fp8_payload_dequantizes_through_the_real_extraction() -> None:
    """Through `extract_fp8_scaled_layers` + `dequantize_fp8_scaled`, so the key spellings are part
    of the assertion: a payload whose scale is stored under a name the extraction does not look for
    would leave the weight un-scaled and the orphaned scale behind."""
    torch.manual_seed(2)
    original = torch.randn(4, 16)
    payload = quantize_scaled_fp8(original)
    state_dict = {"lin.weight": payload.codes, "lin.weight_scale": payload.scale}

    layers = extract_fp8_scaled_layers(state_dict)
    dequantize_fp8_scaled(state_dict, layers, torch.float32)

    assert set(state_dict) == {"lin.weight"}, "the scale was not consumed under the name it was stored as"
    assert torch.equal(state_dict["lin.weight"], payload.dequantized)
    assert torch.corrcoef(torch.stack([state_dict["lin.weight"].flatten(), original.flatten()]))[0, 1] > 0.999


def test_the_scaled_fp8_codes_reach_the_top_of_e4m3() -> None:
    """The other constant a round trip is blind to. `FP8_E4M3_MAX` set too low wastes range; set too
    high it overflows to infinity. Pinned against `torch.finfo`, which is the dtype's own answer
    rather than a restatement of the fixture's number."""
    torch.manual_seed(3)
    payload = quantize_scaled_fp8(torch.randn(4, 16))

    assert payload.codes.float().abs().max().item() == torch.finfo(torch.float8_e4m3fn).max
    assert payload.codes.float().isfinite().all(), "the scale let a weight overflow e4m3"


def test_the_nvfp4_payload_unpacks_to_the_exact_weights_it_claims() -> None:
    """Exact, not approximate: the signed payload is built from the two codes worth +-1.0 and scales
    that are powers of two, so every product is representable. A swapped nibble order transposes the
    sign pattern within each pair and a dropped `weight_scale_2` moves everything by 4x -- both
    exactly visible here, and both invisible to a magnitude check."""
    torch.manual_seed(4)
    # 128 rows by 64 columns is the smallest layer nvfp4 accepts: the scale grid has to be one whole
    # cuBLAS tile, 128 rows by 4 blocks of 16.
    positive = torch.randint(0, 2, (128, 64), dtype=torch.bool)
    tensors, expected = nvfp4_signed_tensors("lin", positive)

    linear = NVFP4Linear(tensors["lin.weight"], tensors["lin.weight_scale"], tensors["lin.weight_scale_2"])
    decoded = linear._dequantized_weight(CPU, torch.float32)

    assert linear.in_features == 64, "the packed weight's logical width was not doubled"
    assert torch.equal(decoded, expected)


def test_an_mx_grid_that_is_not_whole_tiles_is_refused_rather_than_silently_zeroed() -> None:
    """`stored_layout`'s index formula is a bijection only over whole tiles.

    Off-tile it leaves NaN in the positions it never writes, and `.to(torch.uint8)` turns those into
    zeros -- an exponent byte of 0 is `2**-127`, so the payload would look built and decode to
    nothing. The builder is shared now, so this would be wrong in every suite that used it.
    """
    with pytest.raises(ValueError, match="tile layout"):
        mxfp8_tensors("lin", torch.full((64, 4), 127))

    with pytest.raises(ValueError, match="tile layout"):
        mxfp8_tensors("lin", torch.full((128, 3), 127))

    tensors, _expected = mxfp8_tensors("lin", torch.full((128, 4), 127))
    assert tensors["lin.weight_scale"].shape == (128, 4)
