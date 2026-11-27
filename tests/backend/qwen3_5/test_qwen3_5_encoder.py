"""`Qwen35Encoder` runs the `transformers` Qwen3.5 decoder layers in its own loop.

The loop exists so the encoder can stop at the deepest requested layer and return intermediate
outputs without the final norm, or run every layer and apply it. These tests pin that it computes
exactly what the transformers models compute for the same weights, and what the attention-only last
layer means. (Against the reference implementation and the real Anima-3.8B encoder it was checked
separately: fp32 relative error ~1e-6.)
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextModel

from invokeai.backend.model_manager.taxonomy import Qwen35VariantType
from invokeai.backend.qwen3_5.qwen3_5_encoder import (
    Qwen35Encoder,
    load_bundled_qwen3_5_tokenizer,
    qwen3_5_text_config,
    qwen3_5_text_config_from_config_json,
)


def _tiny_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    return config


@pytest.fixture
def models() -> tuple[Qwen3_5TextModel, Qwen35Encoder]:
    torch.manual_seed(0)
    config = _tiny_config()
    reference = Qwen3_5TextModel(config).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    encoder = Qwen35Encoder(config).eval()
    encoder.load_state_dict({k: v for k, v in reference.state_dict().items() if not k.startswith("norm.")}, strict=True)
    return reference, encoder


def _layer_outputs(reference: Qwen3_5TextModel, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Each decoder layer's output, captured by hook -- not through `output_hidden_states`, whose last entry is normed."""
    captured: list[torch.Tensor] = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: captured.append(out)) for layer in reference.layers]
    try:
        with torch.no_grad():
            reference(input_ids=input_ids)
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def test_layer_outputs_match_the_transformers_model(models) -> None:
    reference, encoder = models
    input_ids = torch.randint(0, 64, (1, 70))  # longer than one 64-token delta-rule chunk
    expected = _layer_outputs(reference, input_ids)

    outputs = encoder(input_ids, (1, 3, 7))

    for index, output in zip((1, 3, 7), outputs, strict=True):
        torch.testing.assert_close(output, expected[index])


def test_attention_only_last_layer_skips_exactly_its_mlp(models) -> None:
    reference, encoder = models
    input_ids = torch.randint(0, 64, (1, 9))
    expected = _layer_outputs(reference, input_ids)
    last = encoder.layers[7]
    with torch.no_grad():
        residual = expected[6]
        mlp_term = last.mlp(last.post_attention_layernorm(_post_attention(last, residual)))

    (output,) = encoder(input_ids, (7,), last_layer_attention_only=True)

    torch.testing.assert_close(output + mlp_term, expected[7])


def _post_attention(layer, hidden: torch.Tensor) -> torch.Tensor:
    rotary = Qwen35Encoder(_tiny_config()).rotary_emb
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0).unsqueeze(0).expand(3, -1, -1)
    return hidden + Qwen35Encoder._token_mixer(layer, hidden, rotary(hidden, position_ids))


def test_layers_past_the_deepest_requested_one_are_not_run(models) -> None:
    _, encoder = models
    calls: list[int] = []
    for index, layer in enumerate(encoder.layers):
        layer.register_forward_hook(lambda *_a, i=index: calls.append(i))

    encoder(torch.randint(0, 64, (1, 5)), (2,))

    assert calls == [0, 1, 2]


def test_last_hidden_state_is_what_the_multimodal_model_returns() -> None:
    """MOSS-Image reads `Qwen3_5Model`'s `last_hidden_state` -- the multimodal class, given text only.

    The final norm is Qwen3.5's own, `x * (1 + weight)`; random norm weights keep a generic RMSNorm
    from passing.
    """
    torch.manual_seed(0)
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
    )
    reference = Qwen3_5Model(Qwen3_5Config(text_config=_tiny_config(), vision_config=vision)).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    encoder = Qwen35Encoder(_tiny_config(), final_norm=True).eval()
    prefix = "language_model."
    encoder.load_state_dict(
        {k.removeprefix(prefix): v for k, v in reference.state_dict().items() if k.startswith(prefix)}, strict=True
    )
    input_ids = torch.randint(0, 64, (1, 70))  # longer than one 64-token delta-rule chunk

    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)

    torch.testing.assert_close(encoder.last_hidden_state(input_ids), expected.last_hidden_state)


@pytest.mark.parametrize(
    ("variant", "hidden_size", "intermediate_size"),
    [(Qwen35VariantType.Qwen35_4B, 2560, 9216), (Qwen35VariantType.Qwen35_9B, 4096, 12288)],
)
def test_the_vendored_configs_are_the_published_ones(
    variant: Qwen35VariantType, hidden_size: int, intermediate_size: int
) -> None:
    config = qwen3_5_text_config(variant)
    assert (config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.vocab_size) == (
        hidden_size,
        intermediate_size,
        32,
        248320,
    )
    assert [i for i, t in enumerate(config.layer_types) if t == "full_attention"] == [3, 7, 11, 15, 19, 23, 27, 31]
    assert config.rope_parameters["partial_rotary_factor"] == 0.25
    assert config._attn_implementation == "sdpa"


def test_a_model_config_json_yields_its_text_config() -> None:
    """A diffusers `text_encoder/config.json` describes the multimodal `Qwen3_5Model`; the encoder is its text part."""
    config = qwen3_5_text_config_from_config_json(
        {
            "architectures": ["Qwen3_5Model"],
            "model_type": "qwen3_5",
            "text_config": {"hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 128},
            "vision_config": {"depth": 27, "hidden_size": 1152},
        }
    )
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (64, 4, 128)
    assert config._attn_implementation == "sdpa"

    with pytest.raises(ValueError, match="text_config"):
        qwen3_5_text_config_from_config_json({"architectures": ["Qwen3_5Model"], "hidden_size": 64})


def test_the_vendored_tokenizer_encodes_offline() -> None:
    tokenizer = load_bundled_qwen3_5_tokenizer()
    # Token ids pinned from the tokenizer the reference extension ships (Qwen/Qwen3.5-4B @ 851bf6e).
    assert tokenizer.encode("1girl, Miku from Vocaloid", add_special_tokens=False) == [
        16, 27620, 11, 380, 36974, 494, 93948, 573,
    ]  # fmt: skip
    assert tokenizer.encode("", add_special_tokens=False) == []
    assert tokenizer.pad_token == "<|endoftext|>"


def test_the_vendored_tokenizer_splits_words_as_moss_image_does() -> None:
    """The 9B encodes with this tokenizer too, which holds while transformers loads both alike.

    The vendored `tokenizer.json` keeps combining marks inside a word and MOSS-Image's does not, but
    `Qwen2Tokenizer`, which both configs name, replaces either file's pre-tokenizer with its own. The
    ids below come from the tokenizer MOSS-Image ships; the vendored file's own pre-tokenizer would give
    [58069, 84237, 150104, 153348], so a transformers that honours it fails here.
    """
    tokenizer = load_bundled_qwen3_5_tokenizer()
    namaste = "नमस्ते"  # Devanagari, with a virama and a vowel sign
    assert tokenizer.encode(namaste, add_special_tokens=False) == [58069, 84237, 75681, 196965, 33228]
