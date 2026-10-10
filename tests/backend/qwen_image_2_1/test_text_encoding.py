"""Qwen-Image-2.1's prompt encoding: the template the checkpoint was trained on, read before the final norm."""

import pytest
import torch

from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
from invokeai.backend.qwen_image_2_1.text_encoding import (
    SYSTEM_BLOCK,
    encode_prompt,
    format_prompt,
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
        embeds = encode_prompt(text_encoder, tokenizer, "a fox", torch.device("cpu"))
        ids = tokenizer([format_prompt("a fox")], return_tensors="pt").input_ids
        text_encoder(input_ids=ids)
    finally:
        handle.remove()

    drop = system_prefix_length(tokenizer)
    torch.testing.assert_close(embeds[0], captured["pre"][0, drop:])
    assert not torch.allclose(embeds[0], captured["post"][0, drop:])
