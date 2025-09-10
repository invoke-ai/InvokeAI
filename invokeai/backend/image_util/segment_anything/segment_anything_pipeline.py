from typing import Optional

import torch
from PIL import Image
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.backend.image_util.segment_anything.shared import SAMInput
from invokeai.backend.raw_model import RawModel


class SegmentAnythingPipeline(RawModel):
    """A wrapper class for the transformers SAM model and processor that makes it compatible with the model manager."""

    def __init__(self, sam_model: SamModel, sam_processor: SamProcessor):
        self._sam_model = sam_model
        self._sam_processor = sam_processor

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        # HACK(ryand): The SAM pipeline does not work on MPS devices. We only allow it to be moved to CPU or CUDA.
        if device is not None and device.type not in {"cpu", "cuda"}:
            device = None
        self._sam_model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._sam_model)

    def segment(
        self,
        image: Image.Image,
        inputs: list[SAMInput],
    ) -> torch.Tensor:
        """Run the SAM model.

        Either bounding_boxes or point_lists must be provided. If both are provided, bounding_boxes will be used and
        point_lists will be ignored.

        Args:
            image (Image.Image): The image to segment.
            bounding_boxes (list[list[int]]): The bounding box prompts. Each bounding box is in the format
                [xmin, ymin, xmax, ymax].
            point_lists (list[list[list[int]]]): The points prompts. Each point is in the format [x, y, label].
                `label` is an integer where -1 is background, 0 is neutral, and 1 is foreground.

        Returns:
            torch.Tensor: The segmentation masks. dtype: torch.bool. shape: [num_masks, channels, height, width].
        """

        input_boxes: list[list[list[float]]] = []
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
                input_boxes.append([box])
            if points is not None:
                input_points.append(points)
            if labels is not None:
                input_labels.append(labels)

        processed_inputs = self._sam_processor(
            images=image,
            input_boxes=input_boxes if input_boxes else None,
            input_points=input_points if input_points else None,
            input_labels=input_labels if input_labels else None,
            return_tensors="pt",
        ).to(self._sam_model.device)
        outputs = self._sam_model(**processed_inputs)
        masks = self._sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=processed_inputs.original_sizes,
            reshaped_input_sizes=processed_inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        return masks[0]
