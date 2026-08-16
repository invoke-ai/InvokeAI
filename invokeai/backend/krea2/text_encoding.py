"""Qwen3-VL text encoding for Krea-2.

Extracted from the prompt node so alternative encoders (node packs, experiments) can reuse the exact
token layout instead of restating it. The prompt template is copied from diffusers
``Krea2Pipeline.get_text_hidden_states``: the prefix is a system turn instructing the model to describe
an image (the same "generate" template Qwen-Image uses), which is why the first ``KREA2_START_IDX``
tokens are dropped from the encoder output.
"""

from __future__ import annotations

from typing import Callable

import torch

from invokeai.backend.krea2.sampling_utils import (
    KREA2_MAX_SEQ_LEN,
    KREA2_NUM_SUFFIX_TOKENS,
    KREA2_SELECT_LAYERS,
    KREA2_START_IDX,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice

KREA2_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Reserve room for the suffix (diffusers: max_sequence_length + start_idx - num_suffix_tokens).
KREA2_BODY_MAX_LENGTH = KREA2_MAX_SEQ_LEN + KREA2_START_IDX - KREA2_NUM_SUFFIX_TOKENS

BuildTokenValues = Callable[[torch.Tensor], "torch.Tensor | None"]
"""Callback receiving the body's ``(body_len, 2)`` offset mapping and returning a ``(body_len,)`` vector."""


def encode_krea2_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    build_token_values: BuildTokenValues | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Encode a prompt into Krea-2 conditioning.

    Returns ``(prompt_embeds, prompt_mask, token_values)`` with shapes ``(1, 512, 12, hidden)``,
    ``(1, 512)`` and ``(1, 512)``.

    ``build_token_values`` is an optional extension point for callers that need a per-token vector
    aligned with the conditioning — for example per-token prompt weights. It is handed the tokenizer's
    offset mapping for the prompt body and returns one value per body token; the result is extended over
    the suffix with 1.0 and sliced by the same prefix drop the embeddings and mask get, so it stays
    aligned by construction. Requires a fast tokenizer; ``None`` is returned when the callback yields
    nothing. Callers that do not need this leave it unset, and the tokenizer call is unchanged.
    """
    device = get_effective_device(text_encoder)

    # diffusers tokenizes (prefix + prompt) and the assistant-turn suffix separately, then concatenates -
    # so the suffix always survives truncation. Building one string and truncating it (right-truncation)
    # drops the suffix for long (>~500-token) prompts, corrupting the trained token layout that the fixed
    # prefix-drop (KREA2_START_IDX) and suffix accounting depend on.
    body_text = KREA2_PREFIX + prompt

    want_values = build_token_values is not None
    # Only ask for offsets when they will be used, so the common path makes the exact same call it always did.
    offset_kwargs = {"return_offsets_mapping": True} if want_values else {}
    body_inputs = tokenizer(
        body_text,
        max_length=KREA2_BODY_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        **offset_kwargs,
    )
    # Append the suffix AFTER truncation so it can never be cut, matching the reference layout.
    suffix_inputs = tokenizer(KREA2_SUFFIX, return_tensors="pt")
    input_ids = torch.cat([body_inputs.input_ids, suffix_inputs.input_ids], dim=1).to(device=device)
    attention_mask = torch.cat([body_inputs.attention_mask, suffix_inputs.attention_mask], dim=1).to(
        device=device, dtype=torch.bool
    )
    # Padding sits between the prompt body and assistant suffix. Count only valid tokens when assigning
    # positions so the suffix receives the same mRoPE phase as it did during training.
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # Some VL models nest the language-model output; fall back to that if needed.
    hidden_states_tuple = getattr(outputs, "hidden_states", None)
    if hidden_states_tuple is None:
        lm_output = getattr(outputs, "language_model_outputs", None)
        hidden_states_tuple = getattr(lm_output, "hidden_states", None)
    if hidden_states_tuple is None:
        raise RuntimeError("Qwen3-VL encoder did not return hidden_states; cannot build Krea-2 conditioning.")

    # Stack the selected layers along a new layer axis: (B, seq, 12, hidden).
    stacked = torch.stack([hidden_states_tuple[i] for i in KREA2_SELECT_LAYERS], dim=2)

    # Drop the system-prompt prefix tokens.
    prompt_embeds = stacked[:, KREA2_START_IDX:]
    prompt_mask = attention_mask[:, KREA2_START_IDX:].bool()

    # Match the device-safe compute dtype used by the denoise loop (falls back from bf16 to fp16/fp32 on
    # devices without bf16 support) rather than forcing bfloat16.
    prompt_embeds = prompt_embeds.to(dtype=TorchDevice.choose_bfloat16_safe_dtype(device))

    token_values = None
    if want_values:
        assert build_token_values is not None
        body_values = build_token_values(body_inputs.offset_mapping[0])
        if body_values is not None:
            # The suffix is never weighted, so extend to the full 546-token layout before applying the
            # same prefix drop the embeddings and mask get.
            suffix_values = torch.ones(suffix_inputs.input_ids.shape[1], dtype=body_values.dtype)
            token_values = torch.cat([body_values, suffix_values])[KREA2_START_IDX:].unsqueeze(0)

    return prompt_embeds, prompt_mask, token_values
