"""Qwen3.5's text backbone, run as an encoder that returns hidden states.

Qwen3.5 interleaves Gated DeltaNet linear-attention layers with gated full attention (every fourth
layer). The decoder layers come from `transformers` (`qwen3_5`); this module only owns the loop, so
it can stop at the deepest layer a caller reads and hand back that layer's output without the final
norm `Qwen3_5TextModel.forward` applies -- or run every layer and apply that norm, for a caller that
reads the last hidden state.

Anima-3.8B reads layers 7, 15, 23 and 31 of the 4B model. Its encoder checkpoint
(`qwen35_4b.safetensors`) is not a stock export: it ships layer 31 without its MLP and replaces the
LM head with a 2560->1024 projection Anima never reads. The connector was trained on what that file
computes, so `forward` can run the deepest requested layer attention-only -- matching the file
exactly, and making a stock checkpoint's layer-31 MLP irrelevant rather than wrong.

MOSS-Image reads the 9B model's last hidden state, after the final norm, as transformers'
`Qwen3_5Model` returns it.

Single-file checkpoints carry no config, so the published text configs are vendored: the 4B from
`Qwen/Qwen3.5-4B`'s `config.json`, the 9B -- the `Qwen/Qwen3.5-9B` text config -- from the
`text_encoder/config.json` of `OpenMOSS-Team/MOSS-Image-Base` at revision
1ceede80badddd6a7812069619edaa5633587192.

The tokenizer is vendored from `Qwen/Qwen3.5-4B` (Apache-2.0) at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, the revision the reference ComfyUI extension ships, so a
single-file install encodes offline. It serves the 9B as well: as loaded, it is the tokenizer
MOSS-Image ships -- same vocabulary, merges, special tokens and chat template. The two
`tokenizer.json` files do differ in their pre-tokenizer regex: whether combining marks (Devanagari
vowel signs, Thai tone marks, emoji variation selectors) join a word, which changes token ids. Both
configs name `Qwen2Tokenizer`, though, and transformers replaces either file's regex with that
class's own.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from invokeai.backend.model_manager.taxonomy import Qwen35VariantType
from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir

_PACKAGE_DIR = Path(__file__).parent
_TOKENIZER_DIR = _PACKAGE_DIR / "tokenizer"
_TEXT_CONFIG_FILES = {
    Qwen35VariantType.Qwen35_4B: "qwen3_5_4b_text_config.json",
    Qwen35VariantType.Qwen35_9B: "qwen3_5_9b_text_config.json",
}


@lru_cache(maxsize=1)
def load_bundled_qwen3_5_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3.5 fast tokenizer. Result is cached for the process."""
    return load_gzipped_tokenizer_dir(_TOKENIZER_DIR)


def qwen3_5_text_config(variant: Qwen35VariantType) -> Qwen3_5TextConfig:
    """The vendored text config of a Qwen3.5 size, as published."""
    with open(_PACKAGE_DIR / _TEXT_CONFIG_FILES[variant], "r", encoding="utf-8") as f:
        return _text_config(json.load(f))


def qwen3_5_text_config_from_config_json(config: dict[str, Any]) -> Qwen3_5TextConfig:
    """The text config inside a transformers `Qwen3_5Model` `config.json`, such as a diffusers `text_encoder/`'s."""
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("This Qwen3.5 config.json has no text_config object.")
    return _text_config(text_config)


def _text_config(raw: dict[str, Any]) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(**raw)
    # The decoder layers dispatch attention through this; a config built by hand leaves it unset.
    config._attn_implementation = "sdpa"
    return config


class Qwen35Encoder(nn.Module):
    """Qwen3.5 decoder stack that returns the outputs of selected layers, or its last hidden state.

    Holds `embed_tokens`, `layers` and, with `final_norm`, `norm` under the names a checkpoint uses, so
    a bare-keyed state dict loads directly. Weights for layers past the deepest one any caller reads may
    be absent: they are never run.
    """

    def __init__(self, config: Qwen3_5TextConfig, final_norm: bool = False):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        # Qwen3.5's own norm, not a generic RMSNorm: it scales by `1 + weight` around a zero-initialized weight.
        self.norm: Qwen3_5RMSNorm | None = (
            Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if final_norm else None
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        layer_indices: tuple[int, ...],
        last_layer_attention_only: bool = False,
    ) -> list[torch.Tensor]:
        """Run the stack up to `max(layer_indices)` and return those layers' outputs, in order.

        Args:
            input_ids: Token IDs, unpadded. Shape: (batch, seq_len).
            layer_indices: Zero-based decoder layers whose output to return.
            last_layer_attention_only: Run the deepest requested layer without its MLP sub-block and
                return its post-attention residual. What Anima-3.8B's encoder checkpoint computes.

        Returns:
            One tensor of shape (batch, seq_len, hidden_size) per entry of `layer_indices`.
        """
        if not layer_indices:
            raise ValueError("layer_indices must not be empty")
        last = max(layer_indices)
        if last >= len(self.layers):
            raise ValueError(f"Layer {last} requested from a {len(self.layers)}-layer Qwen3.5 model")

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # Text-only: the three mRoPE axes share one position, which reduces to plain 1-D RoPE.
        position_embeddings = self.rotary_emb(hidden_states, position_ids.unsqueeze(0).expand(3, -1, -1))

        outputs: dict[int, torch.Tensor] = {}
        for index in range(last + 1):
            layer = self.layers[index]
            if index == last and last_layer_attention_only:
                hidden_states = hidden_states + self._token_mixer(layer, hidden_states, position_embeddings)
            else:
                # No mask: a single unpadded prompt, so full attention is plain causal attention and
                # the linear-attention layers have nothing to zero.
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            if index in layer_indices:
                outputs[index] = hidden_states
        return [outputs[index] for index in layer_indices]

    @torch.no_grad()
    def last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run every layer and apply the final norm: `Qwen3_5Model`'s `last_hidden_state` for a text-only prompt.

        Args:
            input_ids: Token IDs, unpadded. Shape: (batch, seq_len).

        Returns:
            Shape (batch, seq_len, hidden_size).
        """
        if self.norm is None:
            raise ValueError("This Qwen3.5 encoder was built without its final norm.")
        (hidden_states,) = self(input_ids, (len(self.layers) - 1,))
        return self.norm(hidden_states)

    @staticmethod
    def _token_mixer(
        layer: Qwen3_5DecoderLayer, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        normed = layer.input_layernorm(hidden_states)
        if layer.layer_type == "linear_attention":
            return layer.linear_attn(hidden_states=normed, cache_params=None, attention_mask=None)
        mixed, _ = layer.self_attn(hidden_states=normed, position_embeddings=position_embeddings, attention_mask=None)
        return mixed
