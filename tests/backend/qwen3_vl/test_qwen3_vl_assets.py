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
    build_qwen3_vl_processor,
    load_bundled_qwen3_vl_config_dict,
    load_bundled_qwen3_vl_preprocessor_config_dict,
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


def test_bundled_preprocessor_config_is_the_released_one() -> None:
    """The values a reference image is cut into patches by. Qwen3-VL's vision tower takes 16px patches with
    0.5/0.5 normalization; transformers' class defaults are Qwen2-VL's 14px and CLIP mean/std, which run
    without error and feed the tower off-distribution pixels."""
    config = load_bundled_qwen3_vl_preprocessor_config_dict()

    assert config["patch_size"] == 16
    assert config["merge_size"] == 2
    assert config["temporal_patch_size"] == 2
    assert config["image_mean"] == [0.5, 0.5, 0.5]
    assert config["image_std"] == [0.5, 0.5, 0.5]
    assert config["size"] == {"longest_edge": 16777216, "shortest_edge": 65536}


def test_bundled_processor_expands_one_pad_per_merged_patch() -> None:
    """Built without any processor files, it turns one ``<|image_pad|>`` into the image's token grid, one pad
    per 2x2 merged 16px patches, and marks those pads as image tokens for the 3-D M-RoPE."""
    import torch

    processor = build_qwen3_vl_processor(load_bundled_qwen3_vl_tokenizer())
    # 384x256 is already a multiple of the 32px merged patch, so the grid is 12 x 8 merged patches.
    image = torch.rand(3, 256, 384)

    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>a cat"], images=[image], return_tensors="pt", do_rescale=False
    )

    assert inputs["image_grid_thw"].tolist() == [[1, 16, 24]]
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pads = inputs["input_ids"] == pad_id
    assert int(pads.sum()) == 16 * 24 // 4
    assert torch.equal(inputs["mm_token_type_ids"].bool(), pads)
    assert inputs["pixel_values"].shape == (16 * 24, 3 * 2 * 16 * 16)


def test_bundled_processor_normalizes_with_the_release_mean_and_std() -> None:
    """Mean and std 0.5: a uniform mid-grey reference must come out as zeros. CLIP's statistics (the class
    defaults) would shift and scale it into values the tower never saw."""
    import torch

    processor = build_qwen3_vl_processor(load_bundled_qwen3_vl_tokenizer())

    inputs = processor.image_processor(images=[torch.full((3, 256, 256), 0.5)], return_tensors="pt", do_rescale=False)

    torch.testing.assert_close(inputs["pixel_values"], torch.zeros_like(inputs["pixel_values"]), atol=1e-6, rtol=0)


def test_bundled_processor_raises_a_small_reference_to_the_pixel_floor() -> None:
    """The release's 65,536 px floor: a 100x100 reference is resized up to 256x256, an 8x8 merged grid."""
    import torch

    processor = build_qwen3_vl_processor(load_bundled_qwen3_vl_tokenizer())

    inputs = processor.image_processor(images=[torch.rand(3, 100, 100)], return_tensors="pt", do_rescale=False)

    assert inputs["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert int(inputs["image_grid_thw"].prod()) // processor.image_processor.merge_size**2 == 8 * 8
