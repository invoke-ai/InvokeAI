from typing import Optional

import torch
from PIL import Image

# Import SAM2 components - these should be available in transformers 4.56.0+
from transformers.models.sam2 import Sam2Model
from transformers.models.sam2.processing_sam2 import Sam2Processor

from invokeai.backend.image_util.segment_anything.shared import SAMInput
from invokeai.backend.raw_model import RawModel


class SegmentAnything2Pipeline(RawModel):
    """A wrapper class for the transformers SAM2 model and processor that makes it compatible with the model manager."""

    def __init__(self, sam2_model: Sam2Model, sam2_processor: Sam2Processor):
        """Initialize the SAM2 pipeline.

        Args:
            sam2_model: The SAM2 model
            sam2_processor: The SAM2 processor (can be Sam2Processor or Sam2VideoProcessor)
        """
        self._sam2_model = sam2_model
        self._sam2_processor = sam2_processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        # HACK: The SAM2 pipeline may not work on MPS devices. We only allow it to be moved to CPU or CUDA.
        if device is not None and device.type not in {"cpu", "cuda"}:
            device = None
        self._sam2_model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        # HACK: Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._sam2_model)

    def segment(
        self,
        image: Image.Image,
        inputs: list[SAMInput],
    ) -> torch.Tensor:
        """Segment the image using the provided inputs.

        Args:
            image: The image to segment.
            inputs: A list of SAMInput objects containing bounding boxes and/or point lists.

        Returns:
            torch.Tensor: The segmentation masks. dtype: torch.bool. shape: [num_masks, channels, height, width].
        """

        input_boxes: list[list[float]] = []
        input_points: list[list[list[float]]] = []
        input_labels: list[list[int]] = []

        for i in inputs:
            box: list[float] | None = None
            points: list[list[float]] | None = None
            labels: list[int] | None = None

            if i.bounding_box is not None:
                box: list[float] | None = [
                    i.bounding_box.x_min,
                    i.bounding_box.y_min,
                    i.bounding_box.x_max,
                    i.bounding_box.y_max,
                ]

            if i.points is not None:
                points = []
                labels = []
                for point in i.points:
                    points.append([point.x, point.y])
                    labels.append(point.label.value)

            if box is not None:
                input_boxes.append(box)
            if points is not None:
                input_points.append(points)
            if labels is not None:
                input_labels.append(labels)

        batched_input_boxes = [input_boxes] if input_boxes else None
        batched_input_points = [input_points] if input_points else None
        batched_input_labels = [input_labels] if input_labels else None

        processed_inputs = self._sam2_processor(
            images=image,
            input_boxes=batched_input_boxes,
            input_points=batched_input_points,
            input_labels=batched_input_labels,
            return_tensors="pt",
        ).to(self._sam2_model.device)

        # Generate masks using the SAM2 model
        outputs = self._sam2_model(**processed_inputs)

        # Post-process the masks to get the final segmentation
        masks = self._sam2_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=processed_inputs.original_sizes,
            reshaped_input_sizes=processed_inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        return masks[0]
