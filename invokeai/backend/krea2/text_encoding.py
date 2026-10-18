"""Qwen3-VL text encoding for Krea-2.

Extracted from the prompt node so alternative encoders (node packs, experiments) can reuse the exact
token layout instead of restating it. The prompt template is copied from diffusers
``Krea2Pipeline.get_text_hidden_states``: the prefix is a system turn instructing the model to describe
an image (the same "generate" template Qwen-Image uses), which is why the first ``KREA2_START_IDX``
tokens are dropped from the encoder output.

Edit mode (reference images read by the Qwen3-VL vision tower) follows ai-toolkit, which trains Krea-2 edit
LoRAs with ``model_kwargs.edit: true``, and ``ostris/ComfyUI-Krea2-Ostris-Edit``, which runs them:
``extensions_built_in/diffusion_models/krea2/src/text_encoder.py::encode_krea_prompt`` and
``krea2.py::_prep_vlm_images``. It encodes at natural length, so its conditioning is not 512 tokens long.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable

import numpy as np
import torch
from PIL import Image

from invokeai.backend.krea2.sampling_utils import (
    KREA2_MAX_SEQ_LEN,
    KREA2_NUM_SUFFIX_TOKENS,
    KREA2_SELECT_LAYERS,
    KREA2_START_IDX,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.util.qwen3_vl import attach_qwen3vl_visual_tower
from invokeai.backend.util.devices import TorchDevice

KREA2_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Reserve room for the suffix (diffusers: max_sequence_length + start_idx - num_suffix_tokens).
KREA2_BODY_MAX_LENGTH = KREA2_MAX_SEQ_LEN + KREA2_START_IDX - KREA2_NUM_SUFFIX_TOKENS

# Reference images: as many as the ComfyUI edit node takes, and the vision-pass budget ai-toolkit trains with
# (`vlm_max_pixels`, 384^2, the TextEncodeQwenImageEditPlus budget). The VLM only needs a coarse view; detail
# reaches the transformer through the VAE reference latents.
KREA2_MAX_REFERENCE_IMAGES = 3
KREA2_VLM_MAX_PIXELS = 384 * 384
KREA2_VLM_MIN_SIDE = 28

# Device memory an image encode needs above the weights, for T tokens. The attention and per-token terms are
# Qwen-Image-2.1's measurement on its Qwen3-VL-8B (RTX 4090, torch 2.13, transformers 5.10: ~286 B * T^2 +
# 224 KiB * T at peak, the allocator reserving about a fifth more), used here pending a calibration on the 4B
# (same 32 heads, narrower layers: 2560 hidden against 4096). Krea-2 also keeps every hidden state
# (`output_hidden_states`: embeddings + 36 layers) and stacks 12 of them, which that measurement did not
# include: 49 * 2560 bf16 values per token. With up to three references at 384^2 (~600 tokens) this comes to
# ~0.5 GB, which normally sits below the cache's `device_working_mem_gb` floor; it decides the reservation only
# with a lowered setting or a very long prompt. Still an estimate on 8B constants until measured on the 4B.
_BYTES_PER_TOKEN_PAIR = 286
_BYTES_PER_TOKEN = 224 * 1024
_RETAINED_HIDDEN_BYTES_PER_TOKEN = (37 + len(KREA2_SELECT_LAYERS)) * 2560 * 2
_RESERVED_OVER_ALLOCATED = 1.25

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
    prompt_embeds = _stacked_conditioning(outputs, device)
    prompt_mask = attention_mask[:, KREA2_START_IDX:].bool()

    token_values = None
    if want_values:
        assert build_token_values is not None
        body_values = build_token_values(body_inputs.offset_mapping[0])
        if body_values is not None:
            # The suffix is never weighted, so extend to the full 546-token layout before applying the
            # same prefix drop the embeddings and mask get. `new_ones` inherits the callback's device and
            # dtype - the callback is an extension seam, so it may well return a CUDA tensor, and a CPU
            # suffix would make the concat fail.
            suffix_values = body_values.new_ones(suffix_inputs.input_ids.shape[1])
            token_values = torch.cat([body_values, suffix_values])[KREA2_START_IDX:].unsqueeze(0)

    return prompt_embeds, prompt_mask, token_values


def _stacked_conditioning(outputs, device: torch.device) -> torch.Tensor:
    """The selected hidden layers stacked per token, ``(1, seq, 12, hidden)``, the system prefix dropped."""
    # Some VL models nest the language-model output; fall back to that if needed.
    hidden_states_tuple = getattr(outputs, "hidden_states", None)
    if hidden_states_tuple is None:
        lm_output = getattr(outputs, "language_model_outputs", None)
        hidden_states_tuple = getattr(lm_output, "hidden_states", None)
    if hidden_states_tuple is None:
        raise RuntimeError("Qwen3-VL encoder did not return hidden_states; cannot build Krea-2 conditioning.")

    # Stack the selected layers along a new layer axis: (B, seq, 12, hidden).
    stacked = torch.stack([hidden_states_tuple[i] for i in KREA2_SELECT_LAYERS], dim=2)

    # Drop the system-prompt prefix tokens, and match the device-safe compute dtype used by the denoise loop
    # (falls back from bf16 to fp16/fp32 on devices without bf16 support) rather than forcing bfloat16.
    return stacked[:, KREA2_START_IDX:].to(dtype=TorchDevice.choose_bfloat16_safe_dtype(device))


def prepare_vlm_reference(image: Image.Image) -> torch.Tensor:
    """A reference image as the Qwen3-VL vision pass reads it: ``(3, H, W)`` float32 in ``[0, 1]``.

    ai-toolkit's ``_prep_vlm_images``: alpha is dropped rather than composited (as ai-toolkit and ComfyUI's
    LoadImage both do), and the image is downscaled -- never upscaled, aspect kept -- to fit
    `KREA2_VLM_MAX_PIXELS`, each side at least `KREA2_VLM_MIN_SIDE`, bicubic with antialiasing on the float
    image. A tensor rather than a resized PIL image, because a uint8 round trip after the resize would round
    what the trained path keeps in float; the processor then gets it with ``do_rescale=False``.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pixels = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    height, width = pixels.shape[1:]
    scale = min(1.0, math.sqrt(KREA2_VLM_MAX_PIXELS / (height * width)))
    size = (max(round(height * scale), KREA2_VLM_MIN_SIDE), max(round(width * scale), KREA2_VLM_MIN_SIDE))
    if size == (height, width):
        return pixels
    resized = torch.nn.functional.interpolate(pixels.unsqueeze(0), size=size, mode="bicubic", antialias=True)
    return resized.squeeze(0).clamp(0, 1)


def image_placeholders(num_images: int) -> str:
    """One named vision placeholder per reference, the ComfyUI ``TextEncodeQwenImageEditPlus`` layout."""
    return "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(num_images))


def tokenize_krea2_prompt_with_images(
    prompt: str, images: Sequence[torch.Tensor], processor, tokenizer
) -> dict[str, torch.Tensor]:
    """The encoder inputs for a prompt with reference images, at natural length, on the CPU.

    `images` come from `prepare_vlm_reference`. The processor expands each ``<|image_pad|>`` into the image's
    token grid (``image_grid_thw.prod() / merge_size**2`` pads) and is called without truncation or padding,
    which would cut those runs apart. The suffix is tokenized separately and appended, as in the text-only
    path, and the processor's ``mm_token_type_ids`` -- what the 3-D M-RoPE is built from -- are extended over
    it as text. The 34-token system prefix is unchanged, so the prefix drop still keeps every image token.

    ``input_ids.shape[1]`` is the sequence length `working_memory_bytes` takes.
    """
    if not images:
        raise ValueError("tokenize_krea2_prompt_with_images needs at least one reference image.")
    try:
        body = processor(
            text=[KREA2_PREFIX + image_placeholders(len(images)) + prompt],
            images=list(images),
            return_tensors="pt",
            do_rescale=False,
        )
    except ValueError as error:
        raise ValueError(_reference_rejection(processor, images, error)) from error
    suffix = tokenizer(KREA2_SUFFIX, return_tensors="pt")
    return {
        "input_ids": torch.cat([body["input_ids"], suffix.input_ids], dim=1),
        "attention_mask": torch.cat([body["attention_mask"], suffix.attention_mask], dim=1).bool(),
        "mm_token_type_ids": torch.cat(
            [body["mm_token_type_ids"], torch.zeros_like(suffix.input_ids, dtype=body["mm_token_type_ids"].dtype)],
            dim=1,
        ),
        "pixel_values": body["pixel_values"],
        "image_grid_thw": body["image_grid_thw"],
    }


def _reference_rejection(processor, images: Sequence[torch.Tensor], error: ValueError) -> str:
    """Name the reference the image processor refused, and why. Only runs once the joint call has failed."""
    for index, image in enumerate(images, start=1):
        try:
            processor.image_processor(images=[image], return_tensors="pt", do_rescale=False)
        except ValueError as own_error:
            height, width = image.shape[1:]
            return (
                f"Krea-2 reference image {index} cannot be read by the Qwen3-VL vision tower "
                f"({width}x{height} px after the 384x384 downscale): {own_error}. Crop it to a less extreme shape."
            )
    return f"The Qwen3-VL processor refused the Krea-2 reference images: {error}"


def working_memory_bytes(num_tokens: int) -> int:
    """Device memory an image encode of `num_tokens` tokens needs above the encoder's weights (an estimate)."""
    per_token = _BYTES_PER_TOKEN + _RETAINED_HIDDEN_BYTES_PER_TOKEN
    return int((_BYTES_PER_TOKEN_PAIR * num_tokens**2 + per_token * num_tokens) * _RESERVED_OVER_ALLOCATED)


def encode_krea2_prompt_with_images(
    inputs: dict[str, torch.Tensor], text_encoder: torch.nn.Module, vision_tower: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode `tokenize_krea2_prompt_with_images` output into Krea-2 conditioning.

    Returns ``(prompt_embeds, prompt_mask)`` of shapes ``(1, L, 12, hidden)`` and ``(1, L)``, ``L`` the natural
    length less the system prefix, the mask all True.

    `vision_tower` is lent to `text_encoder` for this one forward (see `attach_qwen3vl_visual_tower`); both
    must already be locked on the device and stay locked until this returns. No ``position_ids`` are passed:
    the model builds its 3-D M-RoPE from ``image_grid_thw`` and ``mm_token_type_ids``, where the text-only path
    passes explicit 1-D positions because its padding sits inside the sequence.
    """
    device = get_effective_device(text_encoder)
    with attach_qwen3vl_visual_tower(text_encoder, vision_tower):
        outputs = text_encoder(
            **{name: tensor.to(device) for name, tensor in inputs.items()},
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    prompt_embeds = _stacked_conditioning(outputs, device)
    prompt_mask = inputs["attention_mask"][:, KREA2_START_IDX:].to(device)
    return prompt_embeds, prompt_mask
