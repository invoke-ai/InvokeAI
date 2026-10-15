"""MiniMax H3 text (and reference/keyframe-vision) conditioning.

First-party port of the FL2VA and Ref2VA halves of ``modular_pipelines/minimax_h3/encoders.py``
(commit recorded in ``__init__``). The presentation is the verbatim prompt, preceded by
per-keyframe/per-reference labels numbered per modality — ``"<Picture i>: "`` plus a vision
block, ``"<Audio j>: "`` alone (a waveform never reaches the conditioner), ``"<Video k>: "``
plus one timestamped vision block per merged frame group — with no chat template and no
special tokens. The conditioning is the *unnormalized* hidden state after the 50th of the
Qwen3-VL conditioner's 64 decoder layers; the language-model head never runs.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from invokeai.backend.minimax_h3.reference_conditioning import sample_text_conditioning_frames


def validate_text_encoder_depth(text_encoder) -> None:
    """Reject encoders whose ``hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`` is not H3's conditioning.

    In a full stack that entry is mid-stack and always unnormalized. In a stack truncated to
    exactly that many layers, transformers appends the post-final-norm output at that index
    instead - which is only the right tensor when the final norm is an Identity, as in the
    purpose-truncated H3 single-file encoders (their files ship no final norm and the loader
    installs an Identity in its place).
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    truncated_ok = num_layers == MINIMAX_H3_TEXT_ENCODER_LAYER and isinstance(
        text_encoder.model.language_model.norm, torch.nn.Identity
    )
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER and not truncated_ok:
        raise ValueError(
            f"MiniMax H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of its Qwen3-VL "
            f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers; the "
            f"selected text encoder has {num_layers}. A truncated stack's last hidden state is post-norm "
            "and is not the conditioning MiniMax H3 expects (unless the final norm is an Identity, as in "
            "the H3-truncated single-file encoders)."
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
    validate_text_encoder_depth(text_encoder)

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


@dataclass
class MiniMaxH3TextReference:
    """What the text conditioner needs of one Ref2VA reference, in packed order.

    ``image`` is the normalized (2048-short-edge or area-matched) reference image;
    ``sampled_frames`` are the 2 fps frames sampled off the normalized 24 fps stack by
    :func:`~invokeai.backend.minimax_h3.reference_conditioning.sample_text_conditioning_frames`
    — pass the FULL normalized stack via ``frames`` instead to have :func:`encode_prompt_ref2va`
    sample them. Audio contributes only ``has_audio`` (its ``"<Audio j>: "`` label); a waveform
    never reaches the conditioner.
    """

    kind: str
    """`"image"`, `"video"` or `"audio"`."""
    has_audio: bool = False
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    """The full normalized 24 fps frame stack of a video reference, `(T, H, W, 3)` uint8."""


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3TextReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    video_block_timestamps: list[list[float]],
) -> tuple[list[int], list[int]]:
    r"""
    Tokenize MiniMax H3's presentation of a `ref2va` request.

    Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision
    block for an image, `"<Audio j>: "` alone for audio — a waveform never reaches the conditioner — and
    `"<Video k>: "` plus one timestamped vision block per merged frame group for a video. A video that carries
    sound is labelled `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The
    prompt follows verbatim, with no chat template and no special tokens.

    These labels are how a user's prompt addresses a reference, so the webv2 Video panel badges each reference
    card with the ones it earns here; `referencePromptLabels` in
    `invokeai/frontend/webv2/src/features/video/core/settings.ts` mirrors this numbering and must change with it.
    A badge that names a label the prompt cannot resolve fails silently -- the model simply ignores the mention.

    Returns:
        `tuple[list[int], list[int]]`: the token ids and their modality tags.
    """

    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def encode_prompt_ref2va(
    text_encoder,
    tokenizer,
    processor,
    prompt: str,
    references: list[MiniMaxH3TextReference],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build MiniMax H3's `ref2va` presentation and encode it in one conditioner call.

    The vision tensors are batched PER MODALITY while the presentation is tokenized in
    request order; the two agree because relative order is preserved within each modality and
    Qwen3-VL fills the n-th pad *run* of a modality with the n-th entry of that modality's
    batch. Audio contributes nothing beyond its label.
    """
    validate_text_encoder_depth(text_encoder)

    merge_size = processor.image_processor.merge_size**2
    vision_inputs: dict[str, torch.Tensor] = {}

    image_token_counts: list[int] = []
    images = [reference.image for reference in references if reference.kind == "image"]
    if any(image is None for image in images):
        raise ValueError("Every image reference must carry its normalized image.")
    if images:
        image_features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = image_features["pixel_values"]
        vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
        image_token_counts = [int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]]

    video_block_token_counts: list[int] = []
    video_block_timestamps: list[list[float]] = []
    videos = [reference for reference in references if reference.kind == "video"]
    if any(reference.frames is None for reference in videos):
        raise ValueError("Every video reference must carry its normalized frames.")
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            sample_text_conditioning_frames(reference.frames, temporal_patch)
            for reference in videos
            if reference.frames is not None
        ]
        video_block_timestamps = [timestamps for _, timestamps in sampled]
        video_features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
        )
        vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
        video_block_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
        ]
        for timestamps, grid in zip(video_block_timestamps, video_features["video_grid_thw"], strict=True):
            if int(grid[0]) != len(timestamps):
                raise ValueError(
                    f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax H3 "
                    f"labels {len(timestamps)} of them."
                )

    token_ids, token_tags = build_ref2va_presentation(
        tokenizer, prompt, references, image_token_counts, video_block_token_counts, video_block_timestamps
    )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    vision_kwargs = {
        name: value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
        for name, value in vision_inputs.items()
    }
    # Call the language-model submodule directly: MiniMax H3 reads hidden_states[50] and never
    # uses the LM head, whose vocabulary-wide projection is all the top-level forward would add.
    # (InvokeAI's model cache places the whole module before this is called, so upstream's
    # accelerate-hook workaround is not needed here.)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]
    return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)
