"""The FLUX.3 prompt upsampler: a vision-language model that writes the caption FLUX.3 encodes.

Black Forest Labs' upsampler (a Qwen3.5-9B fine-tune) sees the request, the reference images of an edit and the
boxes of a layout. This module builds its chat as diffusers' ``upsample_prompt`` does (``modular_pipelines/flux3/
prompt_upsampler.py``, which adapts ``Upsampler.upsample`` of
github.com/black-forest-labs/flux-3-image-klein, Apache-2.0, see ``NOTICE.md``), runs one greedy generation and
reconciles the answer with the locked rows. It differs from
``upsample_prompt`` only in what a graph needs: generation can be cancelled and reports its progress.

The system prompts, the scaffolds, the box-table helpers and the reference preprocessing are diffusers' own, used
rather than copied so they cannot drift. Importing that module loads transformers' Qwen3.5 modeling, which every node
module would then pay for at server start, so ``_reference`` imports it on first use. That is the only place.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL.Image import Image
from transformers import PreTrainedModel, ProcessorMixin, StoppingCriteria, StoppingCriteriaList

# Equals diffusers' ``MAX_NEW_TOKENS``, the reference's budget for one caption; a test keeps them equal.
FLUX3_UPSAMPLER_MAX_NEW_TOKENS = 3584

# Tokens between two progress reports.
PROGRESS_TOKEN_INTERVAL = 16


class PromptUpsamplerCanceled(Exception):
    """Generation was stopped by the caller's cancel check; no caption was produced."""


def _reference():
    from diffusers.modular_pipelines.flux3 import prompt_upsampler

    return prompt_upsampler


def prepare_upsampler_references(images: Sequence[Image]) -> list[Image]:
    """The reference images as the upsampler sees them: over white, within the pixel budget for their count, re-encoded
    as JPEG. Raises ValueError for more than ten, or an aspect ratio above 200:1."""
    return _reference().prepare_references(list(images))


class _GenerationWatch(StoppingCriteria):
    """Polls the cancel check and reports the generated token count, once per generated token."""

    def __init__(
        self,
        prompt_length: int,
        is_canceled: Callable[[], bool] | None,
        on_progress: Callable[[int], None] | None,
    ) -> None:
        super().__init__()
        self._prompt_length = prompt_length
        self._is_canceled = is_canceled
        self._on_progress = on_progress
        self.canceled = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        if self._is_canceled is not None and self._is_canceled():
            self.canceled = True
        generated = input_ids.shape[1] - self._prompt_length
        if self._on_progress is not None and generated % PROGRESS_TOKEN_INTERVAL == 0:
            self._on_progress(generated)
        return torch.full((input_ids.shape[0],), self.canceled, dtype=torch.bool, device=input_ids.device)


def upsample_flux3_prompt(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    request: str,
    *,
    images: Sequence[Image] = (),
    locked_rows: Sequence[dict[str, Any]] = (),
    max_new_tokens: int = FLUX3_UPSAMPLER_MAX_NEW_TOKENS,
    device: torch.device,
    is_canceled: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    """The caption for ``request``, written from the prepared reference ``images`` (an edit) and around the
    ``locked_rows`` of a layout, which the answer is reconciled with.

    The mode, and so the system prompt, follows the arguments: t2i, i2i, t2i_bbox or i2i_bbox. One greedy generation
    of at most ``max_new_tokens``, ended by the model's own end-of-turn tokens.

    Args:
        model: The loaded upsampler, on ``device``.
        processor: Its processor, which carries the tokenizer and the chat template.
        request: The prompt. With locked rows, the caption they belong to, without its box table.
        images: The reference images, from ``prepare_upsampler_references``, in attachment order.
        locked_rows: The box-table rows to keep, as ``flux3_locked_rows`` returns them.
        max_new_tokens: The most tokens the caption may take.
        device: Where the inputs go; the model's compute device.
        is_canceled: Polled once per generated token.
        on_progress: Called with the generated token count every ``PROGRESS_TOKEN_INTERVAL`` tokens.

    Raises:
        PromptUpsamplerCanceled: ``is_canceled`` returned true; nothing is returned or reconciled.
        RuntimeError: The model answered with nothing.
    """
    reference = _reference()
    locked = [dict(row) for row in locked_rows]
    mode = ("i2i" if images else "t2i") + ("_bbox" if locked else "")
    if locked:
        request = reference.format_layout_request(request, locked)

    # Parts throughout: processor chat templates do not take plain strings.
    if images:
        scaffold = reference.I2I_SCAFFOLD[bool(locked)].format(instruction=request)
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": scaffold}]
    else:
        content = [{"type": "text", "text": request}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": reference.SYSTEM_PROMPTS[mode]}]},
        {"role": "user", "content": content},
    ]
    # The processor renders the template and expands the image placeholders in one call, for a transformers folder's
    # processor and the vendored single-file one alike. Token ids stay integral; pixels take the model's dtype.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device=device, dtype=model.dtype)

    prompt_length = inputs["input_ids"].shape[1]
    watch = _GenerationWatch(prompt_length, is_canceled, on_progress)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stopping_criteria=StoppingCriteriaList([watch]),
    )
    # The watch cut generation short: what came out is a partial caption.
    if watch.canceled:
        raise PromptUpsamplerCanceled
    caption = processor.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError("The FLUX.3 prompt upsampler returned an empty caption.")
    return reference.reconcile_boxes(caption, locked) if locked else caption
