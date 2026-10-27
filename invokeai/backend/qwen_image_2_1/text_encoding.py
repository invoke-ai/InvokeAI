"""Qwen-Image-2.1's prompt encoding, as `QwenImage21Pipeline._get_qwen_prompt_embeds` does it."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import NamedTuple

import torch
from PIL import Image

from invokeai.backend.util.attention import sdpa_score_matrix_bytes

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_BLOCK = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
# A raw string handed to the tokenizer, not `apply_chat_template`: the two tokenize differently, and the
# checkpoint was trained on this one.
T2I_TEMPLATE = SYSTEM_BLOCK + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PAD = "<|image_pad|>"
# Pixels per side of one vision token: 16px patches, merged 2x2.
_PIXELS_PER_TOKEN = 32
# The vision tower reads each reference's 16px patches before the 2x2 merge: four per vision token.
_PATCHES_PER_TOKEN = 4

# What one encode needs above the weights for a sequence of T tokens, measured on an RTX 4090 in bf16 (torch 2.13,
# transformers 5.10): ~286 B * T^2 + 224 KiB * T at peak. The quadratic term is the language model's attention,
# which SDPA runs on its math kernel for grouped-query heads; four 1-megapixel references (~4.1k tokens) need
# ~5.4 GiB. Both terms are activations, so they scale with the compute dtype's width. The allocator reserves about
# a fifth more than it hands out.
_BYTES_PER_TOKEN_PAIR = 286
_BYTES_PER_TOKEN = 224 * 1024
_RESERVED_OVER_ALLOCATED = 1.25


def format_prompt(prompt: str, num_images: int = 0) -> str:
    """The prompt as the checkpoint was trained on it, with one vision placeholder per reference image first."""
    if num_images and IMAGE_PAD in prompt:
        # The processor expands every placeholder into one image's tokens, so a literal one would claim an image
        # that is not there.
        raise ValueError(f"A prompt that reads reference images cannot contain {IMAGE_PAD}. Remove it from the prompt.")
    placeholders = " ".join(f"<image{i}><|vision_start|>{IMAGE_PAD}<|vision_end|>" for i in range(1, num_images + 1))
    # Qwen has no BOS token, so an empty prompt would leave the encoder nothing to read.
    return T2I_TEMPLATE.format(placeholders + (prompt or " "))


def sequence_length(tokenizer, prompt: str, images: Sequence[Image.Image] = ()) -> int:
    """Tokens the encoder reads: the formatted prompt, each reference's placeholder one token per 32x32 pixels.

    `images` are the references at `vae.reference_size`, which the image processor leaves at their size.
    """
    text = len(tokenizer(format_prompt(prompt, len(images))).input_ids) - len(images)
    return text + sum((w // _PIXELS_PER_TOKEN) * (h // _PIXELS_PER_TOKEN) for w, h in (i.size for i in images))


def vision_attention_bytes(
    tower: torch.nn.Module, images: Sequence[Image.Image], device: torch.device, dtype: torch.dtype
) -> int:
    """The score matrix the vision tower's attention materializes for the largest reference, 0 where it is fused.

    The tower attends within one image at a time, over its patches: four per vision token, ~4k for 1 megapixel.
    """
    if not images:
        return 0
    patches = max((w // _PIXELS_PER_TOKEN) * (h // _PIXELS_PER_TOKEN) for w, h in (i.size for i in images))
    return sdpa_score_matrix_bytes(
        device=device,
        dtype=dtype,
        num_heads=tower.config.num_heads,
        head_dim=tower.config.hidden_size // tower.config.num_heads,
        seq_len=_PATCHES_PER_TOKEN * patches,
    )


def working_memory_bytes(num_tokens: int, element_size: int, vision_attention: int = 0) -> int:
    """Device memory an encode of `num_tokens` tokens needs above the encoder's weights.

    `element_size` is the compute dtype's width. `vision_attention` is the tower's score matrix
    (`vision_attention_bytes`): the tower runs, and frees it, before the language model starts, so only the larger
    of the two attention peaks counts.
    """
    width = element_size / 2
    attention = max(_BYTES_PER_TOKEN_PAIR * num_tokens**2 * width, vision_attention)
    return int((attention + _BYTES_PER_TOKEN * num_tokens * width) * _RESERVED_OVER_ALLOCATED)


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


def has_vision_tower(text_encoder: torch.nn.Module) -> bool:
    """Whether this encoder can read reference images: whether a vision tower is attached (`attached_vision_tower`).

    Every encoder loads with an `Identity` where the tower was.
    """
    visual = getattr(_base_model(text_encoder), "visual", None)
    return isinstance(visual, torch.nn.Module) and not isinstance(visual, torch.nn.Identity)


@contextmanager
def attached_vision_tower(text_encoder: torch.nn.Module, tower: torch.nn.Module) -> Iterator[None]:
    """Attach the pipeline's vision tower to its cached encoder for one encode.

    The encoder and the tower are separate model-cache entries, so the language model is cached once for text
    prompts and edits. Both must be locked on the same device for the whole block. Whatever the block does, the
    encoder leaves it with the module it had, so the cache never holds an encoder with a tower it does not own.
    """
    base = _base_model(text_encoder)
    previous = base.visual
    base.visual = tower
    try:
        yield
    finally:
        base.visual = previous


def on_white(image: Image.Image) -> Image.Image:
    """A reference as the vision encoder reads it: alpha composited over white, as the checkpoint was trained."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


class PromptEncoding(NamedTuple):
    embeds: torch.Tensor
    """`(1, L, hidden)`, the system turn dropped."""
    image_pad_mask: torch.Tensor | None
    """`(1, L)`, True at each `<|image_pad|>` slot, where the transformer places 2x2 of a reference's latents."""
    reference_grids: tuple[tuple[int, int], ...]
    """Per reference, in order: its slots as (rows, columns), one slot per 32x32 pixels."""


@torch.no_grad()
def encode_prompt(
    text_encoder: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    processor=None,
    images: Sequence[Image.Image] = (),
) -> PromptEncoding:
    """Encode a prompt, and the reference images it edits.

    The transformer reads the last decoder layer's output BEFORE the final RMSNorm. transformers 5 ties
    the last hidden state to the normalized output, so a forward hook returning the norm's input
    neutralizes the norm for this call -- the pipeline's approach. The system turn every prompt shares is
    dropped from the result.

    `images` are the references, already at `vae.reference_size`; reading them needs the vision tower and the
    `processor` that cuts them into patches.
    """
    base = _base_model(text_encoder)
    if images:
        if processor is None or not has_vision_tower(text_encoder):
            raise ValueError("Reading reference images needs the Qwen3-VL vision tower and its image processor.")
        inputs = processor(
            text=[format_prompt(prompt, len(images))], images=[on_white(i) for i in images], return_tensors="pt"
        ).to(device)
        forward_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
        }
        if "mm_token_type_ids" in inputs:
            forward_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
    else:
        inputs = tokenizer([format_prompt(prompt)], return_tensors="pt").to(device)
        forward_kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

    handle = base.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        # One forward, so the KV cache the language model would build over the whole sequence is waste.
        hidden = base(**forward_kwargs, use_cache=False).last_hidden_state
    finally:
        handle.remove()

    drop = system_prefix_length(tokenizer)
    if not images:
        return PromptEncoding(hidden[:, drop:], None, ())
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    merge = processor.image_processor.merge_size
    grids = tuple((int(h) // merge, int(w) // merge) for _, h, w in inputs.image_grid_thw.tolist())
    return PromptEncoding(hidden[:, drop:], (inputs.input_ids == image_pad_id)[:, drop:], grids)
