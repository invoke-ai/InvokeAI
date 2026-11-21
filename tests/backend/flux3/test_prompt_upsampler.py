"""The FLUX.3 prompt upsampler's generation (`invokeai/backend/flux3/prompt_upsampler.py`).

The model is a recording fake that answers with canned tokens, so the inputs it is handed are the subject: diffusers'
own `upsample_prompt` is the oracle for them, called with the same fake, processor, images and boxes. The processor is
the vendored single-file Qwen3.5 one, which serves a transformers folder's identically.
"""

import json
from typing import Any

import pytest
import torch
from diffusers.modular_pipelines.flux3 import prompt_upsampler as diffusers_upsampler
from PIL import Image
from transformers import StoppingCriteriaList

from invokeai.backend.flux3.prompt_upsampler import (
    FLUX3_UPSAMPLER_MAX_NEW_TOKENS,
    PROGRESS_TOKEN_INTERVAL,
    PromptUpsamplerCanceled,
    prepare_upsampler_references,
    upsample_flux3_prompt,
)
from invokeai.backend.qwen3_5.qwen3_5_vl import build_qwen3_5_vl_processor

END_OF_TURN = "<|im_end|>"
CPU = torch.device("cpu")

OBJECT_ROWS = [
    {"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"},
    {"id": "obj_2", "bbox": [300, 600, 950, 900], "desc": "a lighthouse on a cliff"},
]
TARGET_ROWS = [
    {
        "id": "obj_1",
        "from": "target",
        "src_bbox": None,
        "tgt_bbox": [80, 100, 450, 420],
        "desc": "a red hot-air balloon",
    },
]


@pytest.fixture(scope="module")
def processor():
    return build_qwen3_5_vl_processor()


class _FakeUpsampler:
    """Records what `generate` received and answers with `answer`, a token at a time, as `generate` would: the stopping
    criteria are asked after every token and end the answer where one says so."""

    device = CPU
    dtype = torch.float32

    def __init__(self, processor: Any, answer: str) -> None:
        tokenizer = processor.tokenizer
        self.answer_ids = tokenizer.encode(answer, add_special_tokens=False) + tokenizer.encode(
            END_OF_TURN, add_special_tokens=False
        )
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        sequence = kwargs["input_ids"]
        criteria = kwargs.get("stopping_criteria", StoppingCriteriaList())
        for token in self.answer_ids:
            sequence = torch.cat([sequence, torch.tensor([[token]])], dim=1)
            if bool(criteria(sequence, None).any()):
                break
        return sequence


def _images(count: int) -> list[Image.Image]:
    sizes = [(70, 40), (50, 90)]
    return prepare_upsampler_references(
        [Image.new("RGB", sizes[i], (40 * i + 60, 90, 200 - 50 * i)) for i in range(count)]
    )


def _diffusers_boxes(rows: list[dict]) -> list[dict]:
    """The rows as the 0-1 boxes diffusers' `boxes_to_canvas` turns back into them."""
    return [
        {key: [v / 1000 for v in value] if key.endswith("bbox") and value else value for key, value in row.items()}
        for row in rows
    ]


CASES = {
    "t2i": ("A misty coast at dawn.", 0, []),
    "i2i with one image": ("Make the sky orange.", 1, []),
    "i2i with two images": ("Put the cat from the first image into the second.", 2, []),
    "t2i_bbox": ("A misty coast.", 0, OBJECT_ROWS),
    "i2i_bbox": ("Add a balloon.", 1, TARGET_ROWS),
}


@pytest.mark.parametrize("request_text, image_count, rows", CASES.values(), ids=CASES.keys())
def test_the_model_receives_what_diffusers_upsample_prompt_sends_and_the_caption_is_the_same(
    processor: Any, request_text: str, image_count: int, rows: list[dict]
) -> None:
    if rows:
        answer = "A calm sea under <obj_1>. " + json.dumps([{**row, "desc": "new wording"} for row in rows])
    else:
        answer = "A calm sea at dawn, soft light."
    images = _images(image_count)
    ours, theirs = _FakeUpsampler(processor, answer), _FakeUpsampler(processor, answer)

    caption = upsample_flux3_prompt(ours, processor, request_text, images=images, locked_rows=rows, device=CPU)
    reference = diffusers_upsampler.upsample_prompt(
        theirs, processor, request_text, images=images or None, boxes=_diffusers_boxes(rows) or None
    )

    assert caption == reference
    ours_in, theirs_in = ours.calls[0], theirs.calls[0]
    tensors = {key for key, value in theirs_in.items() if isinstance(value, torch.Tensor)}
    assert {"input_ids", "attention_mask"} <= tensors
    assert tensors == {key for key, value in ours_in.items() if isinstance(value, torch.Tensor)}
    for key in tensors:
        assert torch.equal(ours_in[key], theirs_in[key]), key
    # Greedy, the token budget, and nothing else: no sampling or stopping argument beyond diffusers' (ours adds the
    # cancel watch).
    settings = {key: value for key, value in ours_in.items() if key not in tensors | {"stopping_criteria"}}
    assert settings == {key: value for key, value in theirs_in.items() if key not in tensors}
    assert settings["do_sample"] is False
    # The request is part of what the model reads, not only its shape.
    prompt_text = processor.decode(ours_in["input_ids"][0])
    assert request_text in prompt_text
    assert prompt_text.count("<|vision_start|>") == image_count
    assert ("LOCKED BOUNDING-BOX ANNOTATIONS" in prompt_text) == bool(rows)


def test_generation_is_one_run_with_the_given_token_budget(processor: Any) -> None:
    model = _FakeUpsampler(processor, "A calm sea.")

    upsample_flux3_prompt(model, processor, "A sea.", max_new_tokens=200, device=CPU)

    (call,) = model.calls
    assert call["max_new_tokens"] == 200


def test_only_the_generated_tokens_are_decoded_and_stripped(processor: Any) -> None:
    model = _FakeUpsampler(processor, "  A calm sea.\n")

    assert upsample_flux3_prompt(model, processor, "A sea.", device=CPU) == "A calm sea."


def test_the_locked_rows_survive_an_answer_that_replans_them(processor: Any) -> None:
    answer = "A bay with <obj_1> over a <obj_2>, and a <gull>. " + json.dumps(
        [
            {"id": "obj_1", "bbox": [1, 1, 2, 2], "desc": "  a big balloon "},
            {"id": "obj_2", "bbox": [0, 0, 5, 5], "desc": "a tower"},
            {"id": "gull", "bbox": [10, 20, 60, 90], "desc": "a gull"},
        ]
    )
    model = _FakeUpsampler(processor, answer)

    caption = upsample_flux3_prompt(model, processor, "A bay.", locked_rows=OBJECT_ROWS, device=CPU)

    text, table = caption.split(" [", 1)
    assert text == "A bay with <obj_1> over a <obj_2>, and a <gull>."
    assert json.loads("[" + table) == [
        {"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a big balloon"},
        {"id": "obj_2", "bbox": [300, 600, 950, 900], "desc": "a tower"},
        {"id": "gull", "bbox": [10, 20, 60, 90], "desc": "a gull"},
    ]


def test_a_cancelled_generation_stops_at_once_and_returns_no_caption(processor: Any) -> None:
    model = _FakeUpsampler(processor, "A calm sea under a pale sky, with gulls over the water. " * 4)
    polls = 0

    def canceled_after_five_tokens() -> bool:
        nonlocal polls
        polls += 1
        return polls > 5

    with pytest.raises(PromptUpsamplerCanceled):
        upsample_flux3_prompt(
            model, processor, "A sea.", locked_rows=OBJECT_ROWS, device=CPU, is_canceled=canceled_after_five_tokens
        )

    assert polls == 6 < len(model.answer_ids)


def test_a_cancel_before_the_first_token_is_not_an_empty_caption(processor: Any) -> None:
    model = _FakeUpsampler(processor, "")

    with pytest.raises(PromptUpsamplerCanceled):
        upsample_flux3_prompt(model, processor, "A sea.", device=CPU, is_canceled=lambda: True)


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_an_empty_answer_is_an_error_that_names_the_prompt_upsampler(processor: Any, answer: str) -> None:
    model = _FakeUpsampler(processor, answer)

    with pytest.raises(RuntimeError, match="prompt upsampler"):
        upsample_flux3_prompt(model, processor, "A sea.", locked_rows=OBJECT_ROWS, device=CPU)


def test_progress_reports_the_token_count_every_interval_not_every_token(processor: Any) -> None:
    model = _FakeUpsampler(processor, "sea " * 40)
    reported: list[int] = []

    upsample_flux3_prompt(model, processor, "A sea.", device=CPU, on_progress=reported.append)

    generated = len(model.answer_ids)
    assert generated > 2 * PROGRESS_TOKEN_INTERVAL
    assert reported == list(range(PROGRESS_TOKEN_INTERVAL, generated + 1, PROGRESS_TOKEN_INTERVAL))


def test_the_default_token_budget_is_the_reference_one() -> None:
    assert FLUX3_UPSAMPLER_MAX_NEW_TOKENS == diffusers_upsampler.MAX_NEW_TOKENS
