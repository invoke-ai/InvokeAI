"""A mixed fp8 FLUX.2 checkpoint must lose its scale bookkeeping and nothing else.

`flux2_dev_fp8mixed` quantizes only some Linears; the rest stay bf16 in the same file. It also
carries a calibrated `.input_scale` beside every `.weight_scale`, and — the reason this fixture
earns its place — real learned parameters whose names end in `.scale`
(`img_attn.norm.query_norm.scale`). A metadata filter that matches on "scale" anywhere, or on a
bare `.scale` suffix, deletes those weights. An all-fp8 fixture cannot catch either mistake.
"""

import torch

from invokeai.backend.model_manager.load.model_loaders.flux import Flux2CheckpointModel
from invokeai.backend.quantization.fp8_scaled import FP8_DTYPE, is_scale_metadata_key
from tests.backend.model_manager.load.state_dicts.flux2_transformer_fp8mixed_keys import (
    state_dict_keys as mixed_keys,
)

_DTYPES = {"F8_E4M3": FP8_DTYPE, "F32": torch.float32, "BF16": torch.bfloat16}

SCALE_VALUE = 0.25


def _mock_state_dict() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in mixed_keys.items():
        if key.endswith((".weight_scale", ".input_scale")):
            sd[key] = torch.tensor(SCALE_VALUE, dtype=torch.float32)
        else:
            sd[key] = torch.ones(shape, dtype=torch.float32).to(_DTYPES[dtype])
    return sd


def _quantized_weights() -> set[str]:
    return {k[: -len(".weight_scale")] + ".weight" for k in mixed_keys if k.endswith(".weight_scale")}


def test_the_fixture_is_really_mixed_and_carries_input_scales() -> None:
    """Guard the fixture: an all-fp8 or scale-free recapture makes the tests below vacuous."""
    dtypes = {dtype for _, dtype in mixed_keys.values()}
    assert "F8_E4M3" in dtypes and "BF16" in dtypes, "fixture is no longer mixed"
    assert any(k.endswith(".weight_scale") for k in mixed_keys)
    assert any(k.endswith(".input_scale") for k in mixed_keys)
    assert any(k.endswith(".scale") for k in mixed_keys), "no learned `.scale` parameter left to protect"


def test_learned_scale_parameters_are_not_treated_as_metadata() -> None:
    learned = [k for k in mixed_keys if k.endswith(".scale")]

    assert learned
    for key in learned:
        assert not is_scale_metadata_key(key), f"{key} is a weight, not quantization bookkeeping"


def test_dequantization_strips_every_scale_key() -> None:
    sd = Flux2CheckpointModel._dequantize_fp8_weights(None, _mock_state_dict())

    assert not [k for k in sd if k.endswith((".weight_scale", ".scale_weight", ".input_scale"))]


def test_unquantized_tensors_survive_untouched() -> None:
    before = _mock_state_dict()
    quantized = _quantized_weights()

    after = Flux2CheckpointModel._dequantize_fp8_weights(None, dict(before))

    for key, value in before.items():
        if key.endswith((".weight_scale", ".input_scale")) or key in quantized:
            continue
        assert key in after, f"{key} was deleted"
        assert torch.equal(after[key], value), f"{key} was modified"


def test_the_scale_is_folded_into_the_quantized_weights() -> None:
    sd = Flux2CheckpointModel._dequantize_fp8_weights(None, _mock_state_dict())

    quantized = _quantized_weights()
    assert quantized
    for key in quantized:
        # Each mock fp8 weight holds 1.0, so the folded result is exactly the scale.
        assert torch.allclose(sd[key].float(), torch.full_like(sd[key].float(), SCALE_VALUE))
        assert sd[key].dtype is torch.bfloat16


def test_the_scale_weight_spelling_is_folded_on_this_layout_too() -> None:
    """The regression guard: same real layout, scales spelled the other way.

    Reading only `.weight_scale` while stripping both spellings left each quantized weight at its
    raw fp8 codes — measured at a relative error of ~4700 on the real checkpoint, with nothing
    logged.
    """
    renamed = {
        (k[: -len(".weight_scale")] + ".scale_weight" if k.endswith(".weight_scale") else k): v
        for k, v in _mock_state_dict().items()
    }

    sd = Flux2CheckpointModel._dequantize_fp8_weights(None, renamed)

    quantized = _quantized_weights()
    assert quantized
    for key in quantized:
        assert torch.allclose(sd[key].float(), torch.full_like(sd[key].float(), SCALE_VALUE))
    assert not [k for k in sd if k.endswith((".weight_scale", ".scale_weight"))]


def test_a_per_channel_input_scale_is_stripped_by_name_not_by_rank() -> None:
    """A vector-valued `input_scale` must go too.

    Scalar scales were removed incidentally, by the "0-dimensional tensors are metadata" branch
    rather than by name. Anything with a shape slipped past it and reached
    `load_state_dict(..., strict=True)` as an unexpected key.
    """
    sd = _mock_state_dict()
    per_channel = [k for k in sd if k.endswith(".input_scale")]
    assert per_channel
    for key in per_channel:
        sd[key] = torch.full((8,), SCALE_VALUE, dtype=torch.float32)

    out = Flux2CheckpointModel._dequantize_fp8_weights(None, sd)

    assert not [k for k in out if k.endswith(".input_scale")]
