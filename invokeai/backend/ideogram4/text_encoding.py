"""Qwen3-VL text encoding for Ideogram 4.

Ideogram 4 conditions on a concatenation of hidden states taken from 13 specific
layers of the Qwen3-VL language model (see ``QWEN3_VL_ACTIVATION_LAYERS``), giving
a ``(seq_len, 4096 * 13) == (seq_len, 53248)`` feature tensor.

The reference pipeline runs the encoder over the full packed ``[text][image]``
sequence, but the text-token hidden states are independent of the image tokens
(attention is causal and gated to LLM-token positions), so we encode the text
tokens only. The denoise node assembles the full packed sequence.
"""

from __future__ import annotations

import torch

from invokeai.backend.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS

# Matches Ideogram4PipelineConfig.max_text_tokens.
MAX_TEXT_TOKENS = 2048


def encode_qwen3vl_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    *,
    max_text_tokens: int = MAX_TEXT_TOKENS,
) -> torch.Tensor:
    """Encode a single prompt into Ideogram 4 conditioning features.

    Returns a ``(num_text_tokens, 53248)`` float32 tensor (on the encoder's device;
    the caller is responsible for moving it to CPU for storage).
    """
    # Importing here keeps module import cheap and tolerant of transformers versions
    # that lay out the masking utilities differently.
    from transformers.masking_utils import create_causal_mask

    device = next(text_encoder.parameters()).device

    # Chat-format and tokenize, matching Ideogram4Pipeline._tokenize.
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"].to(device)  # (1, L)
    num_text_tokens = int(token_ids.shape[1])
    if num_text_tokens > max_text_tokens:
        raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}")

    # Text-only sequence: every position is a real LLM token.
    attention_mask = torch.ones((1, num_text_tokens), dtype=torch.long, device=device)
    pos_2d = torch.arange(num_text_tokens, device=device)[None, :]  # (1, L)

    language_model = text_encoder.language_model
    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, 1, num_text_tokens)
    text_position_ids = position_ids_4d[0]  # (1, L)
    mrope_position_ids = position_ids_4d[1:]  # (3, 1, L)

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    stacked = torch.stack(selected, dim=0)  # (num_taps, 1, L, H)
    stacked = stacked.permute(1, 2, 3, 0).reshape(1, num_text_tokens, -1)  # (1, L, H*num_taps)
    return stacked[0].to(torch.float32)  # (L, 53248)
