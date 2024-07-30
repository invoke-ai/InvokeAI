from typing import Optional

import torch
from PIL import Image
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.backend.grounded_sam.detection_result import DetectionResult


class SegmentAnythingModel:
    """A wrapper class for the transformers SAM model and processor that makes it compatible with the model manager."""

    def __init__(self, sam_model: SamModel, sam_processor: SamProcessor):
        self._sam_model = sam_model
        self._sam_processor = sam_processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "SegmentAnythingModel":
        self._sam_model.to(device=device, dtype=dtype)
        return self

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._sam_model)

    def segment(self, image: Image.Image, detection_results: list[DetectionResult]) -> torch.Tensor:
        boxes = self._to_box_array(detection_results)
        inputs = self._sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self._sam_model.device)
        outputs = self._sam_model(**inputs)
        masks = self._sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        masks = masks[0]
        return masks

    def _to_box_array(self, detection_results: list[DetectionResult]) -> list[list[list[int]]]:
        """Convert a list of DetectionResults to the bbox format expected by the Segment Anything model."""
        boxes = [result.box.to_box() for result in detection_results]
        return [boxes]
