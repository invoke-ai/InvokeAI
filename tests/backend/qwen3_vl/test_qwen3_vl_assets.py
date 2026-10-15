"""Tests for the vendored Qwen3-VL tokenizer and architecture configs.

Single-file and GGUF Qwen3-VL encoders ship weights only. The tokenizer and config are vendored in
the package so those encoders work offline, and so a hub re-upload cannot change the architecture
constants the weights are folded into.
"""

import json

import pytest

from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType
from invokeai.backend.qwen3_vl.qwen3_vl_assets import (
    _CONFIG_BY_VARIANT,
    load_bundled_qwen3_vl_config_dict,
    load_bundled_qwen3_vl_tokenizer,
)

# Qwen3-VL's text vocabulary. Token ids stay below this; the visual tower's special tokens are
# inside it, not beyond it.
QWEN3_VL_VOCAB_SIZE = 151936


def test_bundled_tokenizer_known_ids() -> None:
    """Pinned ids, not just "non-empty": a wrong-but-working vocabulary encodes fine and conditions
    off-distribution. These are what ``Qwen/Qwen3-VL-8B-Instruct`` produces for this prompt.
    """
    tokenizer = load_bundled_qwen3_vl_tokenizer()

    assert tokenizer("A cinematic photo of a cat").input_ids == [32, 64665, 6548, 315, 264, 8251]


def test_bundled_tokenizer_roundtrip() -> None:
    tokenizer = load_bundled_qwen3_vl_tokenizer()
    prompt = "A cinematic photo of a cat"

    assert tokenizer.decode(tokenizer(prompt).input_ids) == prompt


def test_bundled_tokenizer_encodes_a_rare_prompt_within_the_vocabulary() -> None:
    ids = load_bundled_qwen3_vl_tokenizer()("a rare one: zxqwv 12345 <|vision_start|>").input_ids

    assert ids
    assert all(0 <= i < QWEN3_VL_VOCAB_SIZE for i in ids)


def test_bundled_tokenizer_carries_the_vision_special_tokens() -> None:
    """Krea-2's conditioning splices the vision markers as single ids. A tokenizer that shattered
    them into ordinary BPE pieces would still encode every prompt, just wrongly.
    """
    tokenizer = load_bundled_qwen3_vl_tokenizer()

    for marker in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
        assert len(tokenizer(marker, add_special_tokens=False).input_ids) == 1


def test_bundled_tokenizer_has_chat_template() -> None:
    assert load_bundled_qwen3_vl_tokenizer().chat_template


def test_bundled_tokenizer_is_cached() -> None:
    assert load_bundled_qwen3_vl_tokenizer() is load_bundled_qwen3_vl_tokenizer()


@pytest.mark.parametrize(
    ("variant", "hidden_size", "num_hidden_layers"),
    [
        (Qwen3VLVariantType.Qwen3VL_4B, 2560, 36),
        (Qwen3VLVariantType.Qwen3VL_8B, 4096, 36),
    ],
)
def test_bundled_config_carries_the_released_architecture(
    variant: Qwen3VLVariantType, hidden_size: int, num_hidden_layers: int
) -> None:
    """The constants a single-file checkpoint cannot supply.

    ``rope_theta`` is the sharpest of them: 1e6 rather than Qwen3-VL's 5e6 costs relative L2 0.1008
    against otherwise identical weights — as much as a full Q4 quantization, and just as silent.
    """
    text_config = load_bundled_qwen3_vl_config_dict(variant)["text_config"]

    assert text_config["hidden_size"] == hidden_size
    assert text_config["num_hidden_layers"] == num_hidden_layers
    assert text_config["rope_theta"] == 5000000


def test_every_variant_has_a_bundled_config() -> None:
    """A new variant must arrive with its config, not discover at load time that it has none."""
    assert set(_CONFIG_BY_VARIANT) == set(Qwen3VLVariantType)


def test_bundled_configs_are_distinct_per_variant() -> None:
    """Guards the mapping itself: pointing both variants at one file would build a 4B architecture
    from an 8B checkpoint, which only fails much later as a shape mismatch.
    """
    configs = [json.dumps(load_bundled_qwen3_vl_config_dict(v), sort_keys=True) for v in Qwen3VLVariantType]

    assert len(set(configs)) == len(configs)


def test_bundled_config_is_not_pre_normalized() -> None:
    """Normalization belongs in the load path, which tracks the installed transformers; freezing
    the mirrored ``rope_scaling`` into the vendored file would pin today's transformers instead.
    """
    for variant in Qwen3VLVariantType:
        assert "rope_parameters" not in load_bundled_qwen3_vl_config_dict(variant)["text_config"]


def test_config_dicts_are_not_shared_between_callers() -> None:
    """Callers mutate what they get (layer counts, dtype, tied embeddings) before building a config."""
    first = load_bundled_qwen3_vl_config_dict(Qwen3VLVariantType.Qwen3VL_4B)
    first["text_config"]["num_hidden_layers"] = 1

    assert load_bundled_qwen3_vl_config_dict(Qwen3VLVariantType.Qwen3VL_4B)["text_config"]["num_hidden_layers"] == 36


def test_an_unregistered_variant_is_refused_by_name() -> None:
    class _Unknown:
        value = "qwen3vl_99b"

    with pytest.raises(NotImplementedError, match="qwen3vl_99b"):
        load_bundled_qwen3_vl_config_dict(_Unknown())  # type: ignore[arg-type]
