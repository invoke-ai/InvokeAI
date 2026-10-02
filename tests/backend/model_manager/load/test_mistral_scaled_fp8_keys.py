"""What the released scaled-fp8 Mistral-3-Small encoder actually contains.

Two properties of this build are not visible in any other captured layout, and both are ones the
loader depends on rather than merely observes: the header names its layers in the same namespace as
the tensors, which is what the hint re-key assumes, and the checkpoint carries the tokenizer as a
tensor, which is what a metadata sweep must not take with it.

Deliberately not asserted here: the scale spellings and the per-tensor scale shape. Both are already
pinned against `flux1_transformer_scaled_fp8_keys.py`, which carries a calibrated input scale too.
"""

from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import (
    MISTRAL_KEY_PREFIXES,
    _bare_mistral_path,
)
from invokeai.backend.quantization.fp8_scaled import is_scale_metadata_key
from tests.backend.model_manager.load.state_dicts import mistral_3_small_scaled_fp8_keys as fixture

KEYS = fixture.state_dict_keys
QUANTIZED = sorted(
    # `.endswith`, because the bare `scaled_fp8` marker key carries the same dtype and is not a weight.
    key[: -len(".weight")]
    for key, (_shape, dtype) in KEYS.items()
    if dtype == "F8_E4M3" and key.endswith(".weight")
)


def test_the_header_names_its_layers_in_the_tensors_own_namespace() -> None:
    """The hint re-key rests on this and cannot check it.

    `_quantization_metadata` is written before any prefix is stripped, so a header naming a
    namespace the tensors do not use matches nothing: the flags are dropped, `full_precision_matmul`
    comes out False where the producer asked for True, and the load still succeeds. Here the two
    agree exactly, which is why this file is usable as the reference for that path.
    """
    assert QUANTIZED
    assert set(fixture.layer_hints) == set(QUANTIZED)
    assert {hint["format"] for hint in fixture.layer_hints.values()} == {"float8_e4m3fn"}


def test_a_header_name_and_its_weight_survive_the_rename_together() -> None:
    """Driven through `_bare_mistral_path`, the one function that performs the rename, rather than
    through a restatement of it -- the drift this fixture family exists to catch is a side channel
    re-keyed by a *second* statement of the rule.

    `model.` comes off for bare `MistralModel`; the wrapper prefixes this build does not carry come
    off earlier, which is why they are named as a constant rather than spelled at the call site.
    """
    for path in QUANTIZED:
        assert path.startswith("model."), path
        assert not _bare_mistral_path(path).startswith("model."), path

    assert not any(key.startswith(MISTRAL_KEY_PREFIXES) for key in KEYS)


def test_the_tokenizer_rides_along_as_a_tensor_and_is_not_swept_up_as_metadata() -> None:
    """`tekken_model` is 19 MB of tokenizer sitting beside a zero-element `scaled_fp8` marker.

    One is metadata that has to go or `load_state_dict` sees an unexpected key; the other is a
    payload the encoder needs. A predicate loose enough to take the marker and the tokenizer both
    would fail much later and somewhere unrelated.
    """
    assert KEYS["tekken_model"][1] == "U8"
    assert not is_scale_metadata_key("tekken_model")

    assert KEYS["scaled_fp8"] == ([0], "F8_E4M3")
    assert is_scale_metadata_key("scaled_fp8")
