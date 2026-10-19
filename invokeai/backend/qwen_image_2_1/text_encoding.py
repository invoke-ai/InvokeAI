"""Qwen-Image-2.1's prompt encoding, as `QwenImage21Pipeline._get_qwen_prompt_embeds` does it for text prompts."""

import torch

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_BLOCK = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
# A raw string handed to the tokenizer, not `apply_chat_template`: the two tokenize differently, and the
# checkpoint was trained on this one.
T2I_TEMPLATE = SYSTEM_BLOCK + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def format_prompt(prompt: str) -> str:
    # Qwen has no BOS token, so an empty prompt would leave the encoder nothing to read.
    return T2I_TEMPLATE.format(prompt or " ")


def system_prefix_length(tokenizer) -> int:
    """Leading tokens to drop from the hidden states: the system turn, which every prompt shares."""
    return len(tokenizer(SYSTEM_BLOCK, add_special_tokens=False).input_ids)


def _base_model(text_encoder: torch.nn.Module) -> torch.nn.Module:
    """The `Qwen3VLModel`: what loaders return, or the inner model of a `Qwen3VLForConditionalGeneration`.

    Called the way the pipeline calls it, so positions -- and the bf16 result -- match it bit for bit. (RoPE is
    relative, so the language model alone would differ only by a position offset under left padding: equal in
    exact arithmetic, not in bf16.)
    """
    return text_encoder if hasattr(text_encoder, "language_model") else text_encoder.model


@torch.no_grad()
def encode_prompt(text_encoder: torch.nn.Module, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Encode a prompt into `(1, L, hidden)` embeddings, as the pipeline encodes a single prompt.

    The transformer reads the last decoder layer's output BEFORE the final RMSNorm. transformers 5 ties
    the last hidden state to the normalized output, so a forward hook returning the norm's input
    neutralizes the norm for this call -- the pipeline's approach. The system turn every prompt shares is
    dropped from the result.
    """
    inputs = tokenizer([format_prompt(prompt)], return_tensors="pt").to(device)
    base = _base_model(text_encoder)
    handle = base.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        hidden = base(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
    finally:
        handle.remove()
    return hidden[:, system_prefix_length(tokenizer) :]
