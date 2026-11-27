"""FLUX.3 prompt encoding with the Qwen3-VL 4B text encoder.

A port of diffusers' ``Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds`` (``modular_pipelines/flux3/encoders.py``
in diffusers 0.42.1), which matches the reference implementation's ``TextEncoder.encode``. Small deviations here
degrade images without failing, so each rule below is load-bearing:

- The prompt goes through the chat template as a single user turn, with the generation prompt appended.
- The token window is right-padded to a multiple of 80, capped at 8192. FLUX.3 attends to the padding
  tokens, so the window size is part of the model recipe.
- The padding tokens' position ids are 1 (as in the training environment), not what the current
  transformers release would derive for them; real tokens count up from 0.
- The attention mask is passed explicitly next to the position ids. Without it, transformers 5.10 reads the
  non-monotonic positions as packed sequences and masks the padding into segments of its own.
- Hidden states 4, 8, ..., 32 (index 0 is the embedding output; all are taken before the final norm) are
  stacked per token into a ``(L, 8 * 2560)`` sequence.
"""

import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

FLUX3_TEXT_ENCODER_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
FLUX3_TEXT_PAD_MULTIPLE = 80
FLUX3_MAX_TEXT_TOKENS = 8192
_PADDING_POSITION_ID = 1


def flux3_prompt_text(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    """The prompt in the chat template FLUX.3 was trained with."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def tokenize_flux3_prompt(
    tokenizer: PreTrainedTokenizerBase, prompt: str, max_tokens: int = FLUX3_MAX_TEXT_TOKENS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(input_ids, attention_mask, position_ids)``, each ``(1, L)``, with ``L`` padded to a multiple of 80.

    The padding side is passed per call: the tokenizer is a shared, cached instance.
    """
    text = flux3_prompt_text(tokenizer, prompt)
    real_length = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].shape[1]
    padded_length = min(math.ceil(real_length / FLUX3_TEXT_PAD_MULTIPLE) * FLUX3_TEXT_PAD_MULTIPLE, max_tokens)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=padded_length,
    )
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    position_ids = torch.arange(padded_length)[None].masked_fill(attention_mask == 0, _PADDING_POSITION_ID)
    return input_ids, attention_mask, position_ids


@torch.no_grad()
def encode_flux3_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
    device: torch.device,
    max_tokens: int = FLUX3_MAX_TEXT_TOKENS,
) -> torch.Tensor:
    """The FLUX.3 conditioning of ``prompt``: stacked Qwen3-VL hidden states, ``(1, L, 8 * hidden)``."""
    input_ids, attention_mask, position_ids = tokenize_flux3_prompt(tokenizer, prompt, max_tokens)
    output = text_encoder(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        position_ids=position_ids.to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = output.hidden_states
    if len(hidden_states) <= max(FLUX3_TEXT_ENCODER_LAYERS):
        raise ValueError(
            f"The text encoder returned {len(hidden_states) - 1} layers; FLUX.3 reads layer "
            f"{max(FLUX3_TEXT_ENCODER_LAYERS)}. Select the Qwen3-VL 4B encoder."
        )
    stacked = torch.stack([hidden_states[k] for k in FLUX3_TEXT_ENCODER_LAYERS], dim=1)
    batch_size, num_layers, seq_len, hidden_dim = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
