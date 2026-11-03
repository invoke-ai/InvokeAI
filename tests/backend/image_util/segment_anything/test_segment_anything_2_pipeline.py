from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from invokeai.backend.image_util.segment_anything.segment_anything_2_pipeline import SegmentAnything2Pipeline
from invokeai.backend.image_util.segment_anything.shared import BoundingBox, SAMInput, SAMPoint, SAMPointLabel


class FakeProcessedInputs(dict[str, torch.Tensor]):
    def __init__(self) -> None:
        original_sizes = torch.tensor([[5, 4]])
        super().__init__(pixel_values=torch.ones(1, 3, 5, 4), original_sizes=original_sizes)
        self.original_sizes = original_sizes
        self.to_device: torch.device | None = None

    def to(self, device: torch.device) -> "FakeProcessedInputs":
        self.to_device = device
        return self


class FakeProcessor:
    def __init__(self, processed_inputs: FakeProcessedInputs, masks: torch.Tensor) -> None:
        self.processed_inputs = processed_inputs
        self.masks = masks
        self.input_kwargs: dict[str, Any] | None = None
        self.post_process_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> FakeProcessedInputs:
        self.input_kwargs = kwargs
        return self.processed_inputs

    def post_process_masks(self, *, masks: torch.Tensor, original_sizes: torch.Tensor) -> list[torch.Tensor]:
        self.post_process_kwargs = {"masks": masks, "original_sizes": original_sizes}
        return [self.masks]


class FakeModel:
    def __init__(self, pred_masks: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self.pred_masks = pred_masks
        self.call_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs = kwargs
        return SimpleNamespace(pred_masks=self.pred_masks)


@pytest.mark.parametrize(
    ("sam_input", "expected_points", "expected_labels", "expected_boxes"),
    [
        (
            SAMInput(
                points=[
                    SAMPoint(x=1, y=2, label=SAMPointLabel.positive),
                    SAMPoint(x=3, y=4, label=SAMPointLabel.negative),
                ]
            ),
            [[[[1, 2], [3, 4]]]],
            [[[1, -1]]],
            None,
        ),
        (
            SAMInput(bounding_box=BoundingBox(x_min=1, y_min=2, x_max=3, y_max=4)),
            None,
            None,
            [[[1, 2, 3, 4]]],
        ),
    ],
)
def test_segment_uses_current_sam2_processor_contract(
    sam_input: SAMInput,
    expected_points: list[list[list[list[int]]]] | None,
    expected_labels: list[list[list[int]]] | None,
    expected_boxes: list[list[list[int]]] | None,
) -> None:
    processed_inputs = FakeProcessedInputs()
    pred_masks = torch.rand(1, 2, 3, 2, 2)
    expected_masks = torch.ones(2, 3, 5, 4, dtype=torch.bool)
    processor = FakeProcessor(processed_inputs, expected_masks)
    model = FakeModel(pred_masks)
    pipeline = SegmentAnything2Pipeline(model, processor)  # type: ignore[arg-type]
    image = Image.new("RGB", (4, 5))

    masks = pipeline.segment(image, [sam_input])

    assert processor.input_kwargs == {
        "images": image,
        "input_boxes": expected_boxes,
        "input_points": expected_points,
        "input_labels": expected_labels,
        "return_tensors": "pt",
    }
    assert processed_inputs.to_device == model.device
    assert model.call_kwargs == dict(processed_inputs)
    assert processor.post_process_kwargs == {
        "masks": pred_masks,
        "original_sizes": processed_inputs.original_sizes,
    }
    assert masks is expected_masks
    assert masks.dtype == torch.bool
    assert masks.shape == (2, 3, 5, 4)
