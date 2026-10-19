"""Qwen3.5's text backbone, run as an encoder that returns intermediate hidden states.

Qwen3.5 interleaves Gated DeltaNet linear-attention layers with gated full attention (every fourth
layer). The decoder layers come from `transformers` (`qwen3_5`); this module only owns the loop, so
it can stop at the deepest layer a caller reads and hand back that layer's output without the final
norm `Qwen3_5TextModel.forward` applies.

Anima-3.8B reads layers 7, 15, 23 and 31 of the 4B model. Its encoder checkpoint
(`qwen35_4b.safetensors`) is not a stock export: it ships layer 31 without its MLP and replaces the
LM head with a 2560->1024 projection Anima never reads. The connector was trained on what that file
computes, so `forward` can run the deepest requested layer attention-only -- matching the file
exactly, and making a stock checkpoint's layer-31 MLP irrelevant rather than wrong.

The tokenizer is vendored from `Qwen/Qwen3.5-4B` (Apache-2.0) at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, the revision the reference ComfyUI extension ships, so a
single-file install encodes offline.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir

_PACKAGE_DIR = Path(__file__).parent
_TOKENIZER_DIR = _PACKAGE_DIR / "tokenizer"
QWEN3_5_4B_TEXT_CONFIG_PATH = _PACKAGE_DIR / "qwen3_5_4b_text_config.json"


@lru_cache(maxsize=1)
def load_bundled_qwen3_5_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3.5 fast tokenizer. Result is cached for the process."""
    return load_gzipped_tokenizer_dir(_TOKENIZER_DIR)


def qwen3_5_4b_text_config() -> Qwen3_5TextConfig:
    """The Qwen3.5 4B text config, as published in `Qwen/Qwen3.5-4B`'s `config.json`."""
    with open(QWEN3_5_4B_TEXT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    config = Qwen3_5TextConfig(**raw)
    # The decoder layers dispatch attention through this; a config built by hand leaves it unset.
    config._attn_implementation = "sdpa"
    return config


class Qwen35Encoder(nn.Module):
    """Qwen3.5 decoder stack that returns the outputs of selected layers.

    Holds `embed_tokens` and `layers` under the names a checkpoint uses, so a bare-keyed state dict
    loads directly. Weights for layers past the deepest one any caller reads may be absent: they are
    never run.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

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

    @staticmethod
    def _token_mixer(
        layer: Qwen3_5DecoderLayer, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        normed = layer.input_layernorm(hidden_states)
        if layer.layer_type == "linear_attention":
            return layer.linear_attn(hidden_states=normed, cache_params=None, attention_mask=None)
        mixed, _ = layer.self_attn(hidden_states=normed, position_embeddings=position_embeddings, attention_mask=None)
        return mixed
