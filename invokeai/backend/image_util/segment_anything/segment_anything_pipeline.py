from typing import Optional, TypeAlias

import torch
from PIL import Image
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.backend.raw_model import RawModel

# Type aliases for the inputs to the SAM model.
ListOfBoundingBoxes: TypeAlias = list[list[int]]
"""A list of bounding boxes. Each bounding box is in the format [xmin, ymin, xmax, ymax]."""
ListOfPoints: TypeAlias = list[list[int]]
"""A list of points. Each point is in the format [x, y]."""
ListOfPointLabels: TypeAlias = list[int]
"""A list of SAM point labels. Each label is an integer where -1 is background, 0 is neutral, and 1 is foreground."""


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
        bounding_boxes: list[list[int]] | None = None,
        point_lists: list[list[list[int]]] | None = None,
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

        # Prep the inputs:
        # - Create a list of bounding boxes or points and labels.
        # - Add a batch dimension of 1 to the inputs.
        if bounding_boxes:
            input_boxes: list[ListOfBoundingBoxes] | None = [bounding_boxes]
            input_points: list[ListOfPoints] | None = None
            input_labels: list[ListOfPointLabels] | None = None
        elif point_lists:
            input_boxes: list[ListOfBoundingBoxes] | None = None
            input_points: list[ListOfPoints] | None = []
            input_labels: list[ListOfPointLabels] | None = []
            for point_list in point_lists:
                input_points.append([[p[0], p[1]] for p in point_list])
                input_labels.append([p[2] for p in point_list])

        else:
            raise ValueError("Either bounding_boxes or points and labels must be provided.")

        inputs = self._sam_processor(
            images=image,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self._sam_model.device)
        outputs = self._sam_model(**inputs)
        masks = self._sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        return masks[0]
