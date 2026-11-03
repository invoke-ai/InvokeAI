"""MOSS-Image's prompt encoding: a chat-templated user turn, read from Qwen3.5-9B after its final norm.

Mirrors the reference pipeline's `encode_prompt` (OpenMOSS-Team/MOSS-Image-Base, revision
1ceede80badddd6a7812069619edaa5633587192): the prompt becomes the user turn of a chat with the
generation prompt and thinking enabled, so the formatted string ends in the assistant's opening
`<think>`, and its tokens are encoded in one unpadded forward.
"""

from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from invokeai.backend.util.logging import InvokeAILogger

if TYPE_CHECKING:
    # Not imported at runtime: transformers' qwen3_5 modeling module costs ~0.25 s, which every app
    # start would otherwise pay through the node module that imports this one.
    from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder

logger = InvokeAILogger.get_logger(__name__)


def encode_moss_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    encoder: "Qwen35Encoder",
    max_length: int = 4096,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode a prompt the way MOSS-Image's reference pipeline does.

    Args:
        prompt: The prompt. An empty one still encodes: the chat template alone yields tokens.
        tokenizer: The Qwen3.5 tokenizer.
        encoder: A Qwen3.5-9B encoder built with its final norm.
        max_length: Tokens kept of the formatted prompt; the reference pipeline's default. A longer
            prompt loses its end, the assistant turn included, exactly as in the reference.
        device: Where to run the encoder: its compute device, not where its weights happen to sit,
            since partial loading may have offloaded them all.

    Returns:
        Shape (seq_len, hidden_size), on `device`, in the encoder's dtype.
    """
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not isinstance(formatted, str):
        raise TypeError(f"Expected the chat template to render a string, got {type(formatted).__name__}.")
    input_ids = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    # A prompt of exactly `max_length` tokens fills the budget without losing anything; only count again then.
    if input_ids.shape[1] == max_length and len(tokenizer(formatted).input_ids) > max_length:
        logger.warning(
            f"The MOSS-Image prompt is longer than {max_length} tokens and was truncated, as the reference "
            "pipeline does. Shorten it for best results."
        )
    return encoder.last_hidden_state(input_ids.to(device))[0]
