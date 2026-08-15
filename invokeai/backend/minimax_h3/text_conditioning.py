"""MiniMax H3 text (and keyframe-vision) conditioning.

First-party port of the FL2VA half of ``modular_pipelines/minimax_h3/encoders.py`` (commit
recorded in ``__init__``). The presentation is the verbatim prompt, preceded by a
``"<Picture i>: "`` label and a vision block per keyframe — no chat template, no special
tokens. The conditioning is the *unnormalized* hidden state after the 50th of the Qwen3-VL
conditioner's 64 decoder layers; the language-model head never runs.
"""

import torch

from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)


def encode_prompt(
    text_encoder,
    tokenizer,
    processor,
    prompt: str,
    keyframe_images: list | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build MiniMax H3's presentation of a request and encode it.

    Args:
        text_encoder: A ``Qwen3VLForConditionalGeneration`` (the full released checkpoint).
        tokenizer: The Qwen2 fast tokenizer.
        processor: The ``Qwen3VLProcessor`` (needed even for text-only requests: it derives
            the token-type ids Qwen3-VL's 3D rotary layout keys off).
        prompt: The prompt, a single string (H3 packs one request into one sequence).
        keyframe_images: The keyframes already prepared onto the target canvas, in packed
            order (None or empty for text-to-video).
        device: The device to run the conditioner on.

    Returns:
        ``(prompt_embeds, text_token_tags)``: the ``(1, num_text_tokens, text_dim)`` hidden
        states and the per-row modality tags (vision-block rows are tagged as video).
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of its Qwen3-VL "
            f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers; the "
            f"selected text encoder has {num_layers}. A truncated stack's last hidden state is post-norm "
            "and is not the conditioning MiniMax H3 expects."
        )

    pixel_values, image_grid_thw = None, None
    token_ids: list[int] = []
    token_tags: list[int] = []
    if keyframe_images:
        vision = processor.image_processor(images=keyframe_images, return_tensors="pt")
        pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
        merge_size = processor.image_processor.merge_size**2
        for index in range(len(keyframe_images)):
            num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    # Call the language-model submodule directly: MiniMax H3 reads hidden_states[50] and never
    # uses the LM head, whose vocabulary-wide projection is all the top-level forward would add.
    # (InvokeAI's model cache places the whole module before this is called, so upstream's
    # accelerate-hook workaround is not needed here.)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None if pixel_values is None else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]
    return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)
