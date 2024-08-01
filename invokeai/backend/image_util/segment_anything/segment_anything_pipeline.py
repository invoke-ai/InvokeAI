from typing import Optional

import torch
from PIL import Image
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

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

    def segment(self, image: Image.Image, bounding_boxes: list[list[int]]) -> torch.Tensor:
        """Run the SAM model.

        Args:
            image (Image.Image): The image to segment.
            bounding_boxes (list[list[int]]): The bounding box prompts. Each bounding box is in the format
                [xmin, ymin, xmax, ymax].

        Returns:
            torch.Tensor: The segmentation masks. dtype: torch.bool. shape: [num_masks, channels, height, width].
        """
        # Add batch dimension of 1 to the bounding boxes.
        boxes = [bounding_boxes]
        inputs = self._sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self._sam_model.device)
        outputs = self._sam_model(**inputs)
        masks = self._sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )

        # There should be only one batch.
        assert len(masks) == 1
        return masks[0]
