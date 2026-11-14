"""The FLUX.3 prompt upsampler node, run through `invoke` with the real vendored Qwen3.5 processor and a fake model.

Only the model is faked: a recording stand-in that answers with canned tokens. The prompt upsampler's chat is checked
at the model boundary and against diffusers' system prompts; `tests/backend/flux3/test_prompt_upsampler.py` owns the
generation itself.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.modular_pipelines.flux3.prompt_upsampler import SYSTEM_PROMPTS
from PIL import Image
from transformers import StoppingCriteriaList

from invokeai.app.invocations.fields import FluxKontextConditioningField, ImageField
from invokeai.app.invocations.flux3.flux3_prompt_upsampler import Flux3PromptUpsamplerInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.flux3.layout import Flux3LayoutRegion
from invokeai.backend.model_manager.taxonomy import ModelType, SubModelType
from invokeai.backend.qwen3_5.qwen3_5_vl import build_qwen3_5_vl_processor

REGION = Flux3LayoutRegion.model_validate({"desc": "a red hot-air balloon", "bbox": [80, 100, 450, 420]})
MODEL = ModelIdentifierField(key="pup", hash="x", name="pup", base="any", type="text_llm")


@cache
def _processor() -> Any:
    # Building one costs about a second, and the node deep-copies what it loads: a copy of the 150k-vocabulary
    # tokenizer takes another. Nothing here changes the processor, so every copy is the instance itself.
    processor = build_qwen3_5_vl_processor()
    processor.__deepcopy__ = lambda memo: processor
    return processor


class _FakeUpsampler(torch.nn.Module):
    dtype = torch.float32

    def __init__(self, processor: Any, answer: str, fail: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        tokenizer = processor.tokenizer
        self.answer_ids = tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False)
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("out of memory")
        sequence = kwargs["input_ids"]
        criteria = kwargs.get("stopping_criteria", StoppingCriteriaList())
        for token in self.answer_ids:
            sequence = torch.cat([sequence, torch.tensor([[token]])], dim=1)
            if bool(criteria(sequence, None).any()):
                break
        return sequence


class _Harness:
    """A context whose model loads are recorded in order, with the upsampler locked on the device only while the
    `with` block of `model_on_device` is open."""

    def __init__(
        self,
        answer: str = "A calm sea at dawn.",
        *,
        fail: bool = False,
        config: Any = None,
        canceled: bool = False,
        images: dict[str, Image.Image] | None = None,
    ) -> None:
        self.processor = _processor()
        self.model = _FakeUpsampler(self.processor, answer, fail)
        self.events: list[str] = []
        self.images = images or {}
        self.context = MagicMock()
        self.context.models.get_config.return_value = config or SimpleNamespace(
            type=ModelType.TextLLM, name="pup", supports_images=True
        )
        self.context.models.load.side_effect = self._load
        self.context.models.offload_from_vram.side_effect = lambda identifier: self.events.append("offload")
        self.context.images.get_pil.side_effect = self._get_pil
        self.context.util.is_canceled.return_value = canceled

    def _get_pil(self, name: str) -> Image.Image:
        self.events.append(f"image:{name}")
        return self.images[name]

    def _load(self, identifier: ModelIdentifierField, submodel_type: SubModelType | None = None) -> Any:
        loaded = MagicMock()
        if submodel_type is SubModelType.Processor:
            loaded.model = self.processor
            self.events.append("processor")
        else:
            assert submodel_type is None

            @contextmanager
            def on_device() -> Iterator[tuple[None, _FakeUpsampler]]:
                self.events.append("lock")
                try:
                    yield None, self.model
                finally:
                    self.events.append("unlock")

            loaded.model_on_device = on_device
            loaded.compute_device = torch.device("cpu")
        return loaded

    def run(self, **fields: Any) -> str:
        node = Flux3PromptUpsamplerInvocation(id="upsampler", upsampler_model=MODEL, **fields)
        return node.invoke(self.context).value

    def prompt_text(self) -> str:
        return self.processor.decode(self.model.calls[0]["input_ids"][0])


def _reference(name: str) -> FluxKontextConditioningField:
    return FluxKontextConditioningField(image=ImageField(image_name=name))


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(type=ModelType.TextLLM, name="text-only", supports_images=False),
        SimpleNamespace(type=ModelType.Main, name="a checkpoint"),
    ],
    ids=["text-only LLM", "not a Text LLM"],
)
def test_a_model_that_cannot_read_images_is_refused_before_anything_loads(config: Any) -> None:
    harness = _Harness(config=config)

    with pytest.raises(ValueError, match="cannot be the FLUX.3 prompt upsampler"):
        harness.run(prompt="A sea.")

    harness.context.models.load.assert_not_called()
    assert harness.events == []


def test_more_than_ten_reference_images_are_refused_before_anything_loads() -> None:
    harness = _Harness()

    with pytest.raises(ValueError, match="at most 10 reference images"):
        harness.run(prompt="Edit.", reference_images=[_reference(f"img{i}") for i in range(11)])

    harness.context.models.load.assert_not_called()
    assert harness.events == []


@pytest.mark.parametrize(
    "fields, mode",
    [
        ({}, "t2i"),
        ({"reference_images": _reference("a")}, "i2i"),
        ({"regions": [REGION]}, "t2i_bbox"),
        ({"prompt": 'A sea. [{"id": "obj_1", "bbox": [10, 20, 300, 400], "desc": "a boat"}]'}, "t2i_bbox"),
        ({"reference_images": [_reference("a")], "regions": [REGION]}, "i2i_bbox"),
    ],
    ids=["plain", "reference", "regions", "prompt with a box table", "reference and regions"],
)
def test_the_mode_follows_the_inputs(fields: dict[str, Any], mode: str) -> None:
    harness = _Harness(images={"a": Image.new("RGB", (64, 64), (200, 30, 30))})
    fields = {"prompt": "A sea.", **fields}

    harness.run(**fields)

    assert SYSTEM_PROMPTS[mode] in harness.prompt_text()
    for other in {"t2i", "i2i", "t2i_bbox", "i2i_bbox"} - {mode}:
        assert SYSTEM_PROMPTS[other] not in harness.prompt_text()


def test_regions_become_object_rows_without_reference_images_and_target_rows_with_them() -> None:
    answer = "A sea with <obj_1>. []"
    plain = _Harness(answer)
    edit = _Harness(answer, images={"a": Image.new("RGB", (64, 64))})

    plain_caption = plain.run(prompt="A sea.", regions=[REGION])
    edit_caption = edit.run(prompt="Add a balloon.", regions=[REGION], reference_images=_reference("a"))

    assert plain_caption == (
        'A sea with <obj_1>. [{"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}]'
    )
    assert json.loads(edit_caption.split(". ", 1)[1]) == [
        {"id": "obj_1", "from": "target", "src_bbox": None, "tgt_bbox": [80, 100, 450, 420], "desc": REGION.desc}
    ]


def test_regions_follow_the_rows_of_the_prompts_own_box_table() -> None:
    harness = _Harness("A sea with <obj_1> and <obj_2>. []")
    table = [{"id": "obj_1", "bbox": [10, 20, 300, 400], "desc": "a boat"}]

    caption = harness.run(prompt=f"A sea. {json.dumps(table)}", regions=[REGION])

    assert [row["id"] for row in json.loads(caption.split(">. ", 1)[1])] == ["obj_1", "obj_2"]


def test_a_blank_prompt_with_regions_is_refused_before_anything_loads() -> None:
    harness = _Harness()

    with pytest.raises(ValueError, match="need a prompt"):
        harness.run(prompt=" ", regions=[REGION])

    harness.context.models.load.assert_not_called()


def test_reference_images_reach_the_upsampler_in_connection_order() -> None:
    harness = _Harness(images={"wide": Image.new("RGB", (192, 64)), "tall": Image.new("RGB", (64, 192))})

    harness.run(prompt="Combine.", reference_images=[_reference("wide"), _reference("tall")])

    assert [event for event in harness.events if event.startswith("image:")] == ["image:wide", "image:tall"]
    first, second = harness.model.calls[0]["image_grid_thw"].tolist()
    assert first[2] > first[1] and second[1] > second[2]


def test_the_upsampler_leaves_the_device_after_the_run_has_released_it() -> None:
    harness = _Harness("A calm sea.")

    assert harness.run(prompt="A sea.") == "A calm sea."

    assert harness.events == ["processor", "lock", "unlock", "offload"]


def test_the_upsampler_leaves_the_device_when_generation_fails() -> None:
    harness = _Harness(fail=True)

    with pytest.raises(RuntimeError, match="out of memory"):
        harness.run(prompt="A sea.")

    assert harness.events[-2:] == ["unlock", "offload"]


def test_a_cancel_raises_canceled_and_offloads_without_a_caption() -> None:
    harness = _Harness(canceled=True)

    with pytest.raises(CanceledException):
        harness.run(prompt="A sea.")

    assert harness.events[-2:] == ["unlock", "offload"]
