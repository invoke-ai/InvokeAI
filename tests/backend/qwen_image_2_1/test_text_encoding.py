"""Qwen-Image-2.1's prompt encoding: the template the checkpoint was trained on, read before the final norm, and
the reference images an edit reads through the vision tower."""

import itertools

import pytest
import torch
from PIL import Image

from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
from invokeai.backend.qwen_image_2_1.text_encoding import (
    SYSTEM_BLOCK,
    encode_prompt,
    format_prompt,
    has_vision_tower,
    on_white,
    system_prefix_length,
)
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config


@pytest.fixture(scope="module")
def tokenizer():
    return load_bundled_qwen3_vl_tokenizer()


@pytest.fixture(scope="module")
def text_encoder(tokenizer):
    from transformers import Qwen3VLModel

    config = tiny_qwen3vl_config()
    config.text_config.vocab_size = len(tokenizer)
    torch.manual_seed(0)
    return Qwen3VLModel(config).float().eval()


def test_the_system_turn_is_fourteen_tokens_and_leads_every_prompt(tokenizer) -> None:
    # 14 is what the pipeline derives from the processor's chat template for the same system message.
    assert system_prefix_length(tokenizer) == 14
    prompt_ids = tokenizer(format_prompt("a fox"), add_special_tokens=False).input_ids
    assert prompt_ids[:14] == tokenizer(SYSTEM_BLOCK, add_special_tokens=False).input_ids


def test_the_embeddings_are_the_last_layer_before_the_final_norm(text_encoder, tokenizer) -> None:
    captured: dict[str, torch.Tensor] = {}
    language_model = text_encoder.language_model
    handle = language_model.norm.register_forward_hook(lambda m, args, out: captured.update(pre=args[0], post=out))
    try:
        embeds, image_slots, grids = encode_prompt(text_encoder, tokenizer, "a fox", torch.device("cpu"))
        ids = tokenizer([format_prompt("a fox")], return_tensors="pt").input_ids
        text_encoder(input_ids=ids)
    finally:
        handle.remove()

    drop = system_prefix_length(tokenizer)
    torch.testing.assert_close(embeds[0], captured["pre"][0, drop:])
    assert not torch.allclose(embeds[0], captured["post"][0, drop:])
    assert image_slots is None and grids == ()


def test_reference_placeholders_lead_the_prompt_as_the_pipeline_writes_them() -> None:
    assert format_prompt("swap them", 2) == (
        SYSTEM_BLOCK + "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> "
        "<image2><|vision_start|><|image_pad|><|vision_end|>swap them<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.fixture(scope="module")
def processor(tokenizer):
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    # The released processor's geometry: 16px patches merged 2x2, so one slot per 32x32 pixels.
    image_processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    return Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor()
    )


def _runs(mask: torch.Tensor) -> list[int]:
    return [len(list(run)) for is_slot, run in itertools.groupby(mask[0].tolist()) if is_slot]


def _encode(text_encoder, tokenizer, processor, references):
    return encode_prompt(
        text_encoder, tokenizer, "swap them", torch.device("cpu"), processor=processor, images=references
    )


def test_each_reference_fills_one_run_of_slots_per_32px_block(text_encoder, tokenizer, processor) -> None:
    references = [Image.new("RGBA", (64, 96), (255, 0, 0, 128)), Image.new("RGB", (128, 64), (0, 255, 0))]
    embeds, image_slots, grids = _encode(text_encoder, tokenizer, processor, references)
    assert image_slots is not None and image_slots.shape == embeds.shape[:2]
    # 64x96 is 3 rows of 2 blocks, 128x64 is 2 rows of 4, in the order given; the system turn is gone from both.
    assert grids == ((3, 2), (2, 4))
    assert _runs(image_slots) == [6, 8]


def test_the_vision_tower_reads_the_pixels(text_encoder, tokenizer, processor) -> None:
    red, green = Image.new("RGB", (64, 64), (255, 0, 0)), Image.new("RGB", (64, 64), (0, 255, 0))
    red_embeds, red_slots, _ = _encode(text_encoder, tokenizer, processor, [red])
    green_embeds, _, _ = _encode(text_encoder, tokenizer, processor, [green])
    # Same tokens either way; only the vision tower can tell the two apart.
    assert not torch.allclose(red_embeds[red_slots], green_embeds[red_slots])


def test_a_transparent_reference_reads_as_its_colors_over_white(text_encoder, tokenizer, processor) -> None:
    # Fully transparent black must read as plain white, not as the black its RGB channels hold.
    transparent = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    white = Image.new("RGB", (64, 64), (255, 255, 255))
    transparent_embeds, _, _ = _encode(text_encoder, tokenizer, processor, [transparent])
    white_embeds, _, _ = _encode(text_encoder, tokenizer, processor, [white])
    torch.testing.assert_close(transparent_embeds, white_embeds)


def test_references_need_the_vision_tower(tokenizer, processor) -> None:
    from transformers import Qwen3VLModel

    from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_visual_tower

    config = tiny_qwen3vl_config()
    config.text_config.vocab_size = len(tokenizer)
    encoder = Qwen3VLModel(config).float().eval()
    # As the standalone loaders leave it: an Identity where the tower was.
    drop_qwen3vl_visual_tower(encoder)
    assert not has_vision_tower(encoder)
    with pytest.raises(ValueError, match="vision tower"):
        encode_prompt(
            encoder, tokenizer, "a", torch.device("cpu"), processor=processor, images=[Image.new("RGB", (32, 32))]
        )


def test_transparency_is_flattened_over_white_for_the_vision_encoder() -> None:
    half_red = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
    assert on_white(half_red).getpixel((0, 0)) == (255, 127, 127)
    assert on_white(Image.new("RGBA", (1, 1), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
