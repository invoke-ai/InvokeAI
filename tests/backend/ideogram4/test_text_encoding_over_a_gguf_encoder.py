"""Ideogram 4 conditioning over a GGUF-loaded Qwen3-VL encoder.

Ideogram 4 does not call the encoder: `encode_qwen3vl_prompt` reaches inside it and drives
`language_model.{embed_tokens, layers, rotary_emb, config}` by hand, tapping 13 fixed layer indices
up to 35. The GGUF loader is free to reshape what it builds — it already replaces `visual` — and
nothing else would notice until an 8B user's first generation. Krea-2's node, which calls the model
normally, is covered in the loader's own tests.
"""

import torch

from invokeai.backend.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from invokeai.backend.ideogram4.text_encoding import encode_qwen3vl_prompt
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import HIDDEN_SIZE, load_tiny_gguf_encoder

_PROMPT_TOKENS = 6


class _StubTokenizer:
    """Stands in for the Qwen3-VL tokenizer, which is fetched from HuggingFace at load time.

    What is under test is the encoder side; the tokenizer only has to produce ids in range and the
    two calls the function makes of it.
    """

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return str(messages)

    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.arange(_PROMPT_TOKENS, dtype=torch.long).unsqueeze(0)}


def test_ideogram4_conditioning_runs_over_a_gguf_encoder(monkeypatch, tmp_path) -> None:
    """The full depth, because the taps are fixed indices into it, not a fraction of the layers."""
    encoder = load_tiny_gguf_encoder(monkeypatch, tmp_path, num_hidden_layers=36)
    apply_custom_layers_to_model(encoder)

    features = encode_qwen3vl_prompt("a cat", _StubTokenizer(), encoder)

    assert features.shape == (_PROMPT_TOKENS, HIDDEN_SIZE * len(QWEN3_VL_ACTIVATION_LAYERS))
    assert features.dtype is torch.float32
    assert torch.isfinite(features).all()
