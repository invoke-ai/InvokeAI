from typing import Optional, TypeAlias

import torch
from PIL import Image

# Import SAM2 components - these should be available in transformers 4.56.0+
from transformers.models.sam2 import Sam2Model

from invokeai.backend.raw_model import RawModel

# Type aliases for the inputs to the SAM2 model.
ListOfBoundingBoxes: TypeAlias = list[list[int]]
"""A list of bounding boxes. Each bounding box is in the format [xmin, ymin, xmax, ymax]."""
ListOfPoints: TypeAlias = list[list[list[list[int]]]]
"""A list of points in SAM2 4D format: [[[[x, y]]]] (image_dim, object_dim, point_per_object_dim, coordinates)"""
ListOfPointLabels: TypeAlias = list[list[list[int]]]
"""A list of SAM2 point labels in 3D format: [[[label]]] (image_dim, object_dim, point_label)"""


class SegmentAnything2Pipeline(RawModel):
    """A wrapper class for the transformers SAM2 model and processor that makes it compatible with the model manager."""

    def __init__(self, sam2_model: Sam2Model, sam2_processor):
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
        bounding_boxes: list[list[int]] | None = None,
        point_lists: list[list[list[int]]] | None = None,
        point_labels: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Segment an image using the SAM2 model.

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

        # Prep the inputs for SAM2:
        # - SAM2 expects 4D input format: [[[[x, y]]]] for points and [[[label]]] for labels
        # - Bounding boxes remain in 2D format: [[x_min, y_min, x_max, y_max]]
        if bounding_boxes:
            input_boxes: list[ListOfBoundingBoxes] | None = [bounding_boxes]
            input_points: list[ListOfPoints] | None = None
            input_labels: list[ListOfPointLabels] | None = None
        elif point_lists and point_labels:
            input_boxes: list[ListOfBoundingBoxes] | None = None
            input_points = point_lists
            input_labels = point_labels
        else:
            raise ValueError("Either bounding_boxes or (point_lists AND point_labels) must be provided.")

        # Process the inputs using the SAM2 processor
        inputs = self._sam2_processor(
            images=image,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self._sam2_model.device)

        # Generate masks using the SAM2 model
        outputs = self._sam2_model(**inputs)

        # Post-process the masks to get the final segmentation
        masks = self._sam2_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        return masks[0]
